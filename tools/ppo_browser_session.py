"""One owned browser/worker thread across sequential local Practice stages."""
import asyncio
import threading
import uuid
from urllib.parse import urlsplit


RESOURCES = ('playwright', 'browser', 'context', 'page', 'grabber')


class PersistentBrowserSession:
    def __init__(self):
        self.session_id = uuid.uuid4().hex
        self.loop = None
        self.thread = None
        self.owner = None
        self.resources = {}
        self.launches = self.reuses = 0
        self.closed = False

    def acquire(self, env):
        url = urlsplit(env.args.url)
        if url.scheme != 'http' or url.hostname not in ('localhost', '127.0.0.1') or url.port != 8000:
            raise ValueError('Persistent sessions support only local Practice on port 8000')
        if self.closed or self.owner is not None:
            raise RuntimeError('Browser session is closed or already leased')
        if self.loop is None:
            self.loop = asyncio.new_event_loop()

            def run():
                asyncio.set_event_loop(self.loop)
                self.loop.run_forever()

            self.thread = threading.Thread(target=run, daemon=True, name='practice-browser-session')
            self.thread.start()
        self.owner = env
        env.worker_loop, env.worker_thread = self.loop, self.thread

    def restore(self, env):
        if self.owner is not env:
            raise RuntimeError('Only the current lease may use the browser')
        if not self.resources.get('browser'):
            return False
        if (not self.resources['browser'].is_connected() or self.resources.get('page') is None
                or self.resources['page'].is_closed()):
            raise RuntimeError('Persistent browser disconnected; abort batch instead of replacing its window')
        for key in RESOURCES:
            setattr(env, key, self.resources.get(key))
        self.reuses += 1
        return True

    def remember(self, env):
        if self.owner is not env:
            raise RuntimeError('Only the current lease may register resources')
        if not self.resources.get('browser') and env.browser is not None:
            self.launches += 1
        for key in RESOURCES:
            if getattr(env, key, None) is not None:
                self.resources[key] = getattr(env, key)

    def release(self, env):
        if self.owner is not env:
            raise RuntimeError('Browser released by a different environment')
        self.remember(env)
        self.owner = None
        for key in (*RESOURCES, 'canvas'):
            setattr(env, key, None)

    def summary(self):
        return {'session_id': self.session_id, 'browser_launches': self.launches,
                'browser_reuses': self.reuses, 'same_worker_thread': True,
                'leased': self.owner is not None, 'closed': self.closed}

    def close(self):
        if self.closed:
            return
        if self.owner is not None:
            self.owner.close()
        self.closed = True
        if self.loop is None:
            return

        async def cleanup():
            errors = []
            for name in ('grabber', 'context', 'browser', 'playwright'):
                resource = self.resources.get(name)
                if resource is None:
                    continue
                try:
                    if name == 'grabber':
                        resource.close()
                    else:
                        await getattr(resource, 'stop' if name == 'playwright' else 'close')()
                except Exception as exc:
                    errors.append(f'{name}: {exc}')
            self.resources.clear()
            return errors

        try:
            errors = asyncio.run_coroutine_threadsafe(cleanup(), self.loop).result(timeout=25)
            if errors:
                print('[BROWSER SESSION] Cleanup warnings:', errors, flush=True)
        finally:
            self.loop.call_soon_threadsafe(self.loop.stop)
            self.thread.join(timeout=5)
            if not self.thread.is_alive():
                self.loop.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
