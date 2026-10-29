import unittest
from unittest.mock import patch
import ast
from pathlib import Path
from unittest.mock import AsyncMock

from tools.ppo_browser_focus import ensure_startup_focus, WindowsForeground


class Page:
    def __init__(self, focused=True):
        self.focused = focused
        self.current_title = 'Practice'
        self.front_calls = 0
        self.inputs = []

    async def bring_to_front(self):
        self.front_calls += 1

    async def title(self):
        return self.current_title

    async def evaluate(self, script, *args):
        if args:
            self.current_title = args[0]
        else:
            return self.focused


class Native:
    def __init__(self, foreground=True, found=True):
        self.foreground = foreground
        self.found = found
        self.attempts = 0
        self.marker = None

    def find(self, marker):
        self.marker = marker
        if not self.found:
            raise RuntimeError('ambiguous or missing')
        return 123

    def activate(self, hwnd):
        assert hwnd == 123
        self.attempts += 1

    def is_foreground(self, hwnd):
        return self.foreground

    def shell_activate(self, marker):
        assert marker == self.marker


class FocusTests(unittest.IsolatedAsyncioTestCase):
    async def run_focus(self, page, native, platform='win32'):
        with patch('tools.ppo_browser_focus.asyncio.sleep'):
            await ensure_startup_focus(page, native=native, platform=platform)

    async def test_owns_target_and_restores_title(self):
        page, native = Page(), Native()
        await self.run_focus(page, native)
        self.assertTrue(native.marker.startswith('RS-focus-'))
        self.assertEqual(page.current_title, 'Practice')
        self.assertEqual(page.front_calls, 1)
        self.assertEqual(native.attempts, 1)
        self.assertEqual(page.inputs, [])

    async def test_windows_denial_is_bounded(self):
        page, native = Page(), Native(foreground=False)
        with self.assertRaisesRegex(RuntimeError, 'Windows denied'):
            await self.run_focus(page, native)
        self.assertEqual(native.attempts, 3)
        self.assertEqual(page.current_title, 'Practice')

    async def test_dom_focus_alone_is_insufficient(self):
        with self.assertRaises(RuntimeError):
            await self.run_focus(Page(True), Native(False))

    async def test_os_focus_alone_is_insufficient(self):
        with self.assertRaises(RuntimeError):
            await self.run_focus(Page(False), Native(True))

    async def test_no_unique_target_never_activates(self):
        page, native = Page(), Native(found=False)
        with self.assertRaisesRegex(RuntimeError, 'uniquely identify'):
            await self.run_focus(page, native)
        self.assertEqual(native.attempts, 0)
        self.assertEqual(page.current_title, 'Practice')

    async def test_non_windows_uses_browser_verification(self):
        await self.run_focus(Page(), None, 'linux')
        with self.assertRaisesRegex(RuntimeError, 'focus could not'):
            await self.run_focus(Page(False), None, 'linux')

    async def test_shell_fallback_still_requires_both_checks(self):
        class ShellNative(Native):
            def shell_activate(self, marker):
                super().shell_activate(marker)
                self.foreground = True
        page, native = Page(), ShellNative(foreground=False)
        await self.run_focus(page, native)
        self.assertEqual(native.attempts, 3)
        self.assertEqual(page.current_title, 'Practice')
        with self.assertRaises(RuntimeError):
            await self.run_focus(Page(False), ShellNative(foreground=False))

    async def test_passive_manual_focus_recovery(self):
        class LaterNative(Native):
            checks = 0

            def is_foreground(self, hwnd):
                self.checks += 1
                return self.checks >= 7
        await self.run_focus(Page(), LaterNative(False))

    async def test_attached_fallback_requires_both_focus_checks(self):
        class AttachedNative(Native):
            def activate_attached(self, hwnd):
                self.foreground = True
        await self.run_focus(Page(), AttachedNative(False))
        with self.assertRaises(RuntimeError):
            await self.run_focus(Page(False), AttachedNative(False))


class QueueHandoffTests(unittest.TestCase):
    def test_input_queues_detach_even_if_activation_fails(self):
        calls = []
        class API:
            def GetForegroundWindow(self): return 20
            def GetWindowThreadProcessId(self, hwnd, _): return {10: 2, 20: 3}[hwnd]
            def AttachThreadInput(self, a, b, attach):
                calls.append((a, b, attach))
                return True
            def BringWindowToTop(self, hwnd): raise RuntimeError('injected failure')
        native = object.__new__(WindowsForeground)
        native.api = API()
        native.kernel = type('Kernel', (), {'GetCurrentThreadId': lambda self: 1})()
        with self.assertRaisesRegex(RuntimeError, 'injected'):
            native.activate_attached(10)
        self.assertEqual(calls, [(1, 2, True), (1, 3, True), (1, 3, False), (1, 2, False)])


class StartupIntegrationTests(unittest.IsolatedAsyncioTestCase):
    def actual(self):
        source = (Path(__file__).resolve().parents[1]/'train_rs_v612_ppo.py').read_text(encoding='utf8')
        tree = ast.parse(source)
        owner = next(n for n in ast.walk(tree) if isinstance(n, ast.ClassDef) and n.name == 'PPOV61Env')
        method = next(n for n in owner.body if isinstance(n, ast.AsyncFunctionDef) and n.name == '_open_browser')

        class Base:
            async def _open_browser(self):
                self.opened = True
        module = ast.fix_missing_locations(ast.Module(body=[ast.ClassDef(
            name='Actual', bases=[ast.Name(id='Base', ctx=ast.Load())],
            keywords=[], body=[method], decorator_list=[])], type_ignores=[]))
        ns = {'Base':Base, 'GuardUncertainError':RuntimeError}
        exec(compile(module, '<current PPO startup>', 'exec'), ns)
        instance = ns['Actual']()
        instance.page = Page()
        return instance

    async def test_actual_startup_calls_focus_after_open(self):
        instance = self.actual()

        async def verify(page):
            self.assertTrue(instance.opened)
            self.assertIs(page, instance.page)
        with patch('tools.ppo_browser_focus.ensure_startup_focus', side_effect=verify):
            await instance._open_browser()

    async def test_focus_failure_aborts_actual_startup(self):
        instance = self.actual()
        with patch('tools.ppo_browser_focus.ensure_startup_focus', new=AsyncMock(side_effect=RuntimeError('denied'))):
            with self.assertRaisesRegex(RuntimeError, 'Startup stopped: denied'):
                await instance._open_browser()
