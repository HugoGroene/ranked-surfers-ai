"""Start local Practice only from the first reset, after policy warmup.

Visible evaluation only. A fresh Practice launch can already enter gameplay;
adopt it through the same two-vote startup rule instead of waiting for its death.
Later episodes retain the normal confirmed-menu/R restart path.
"""
import asyncio
import time
from rs_v612_gameplay_guard import GuardUncertainError


async def start_ready_practice(env, pending, restart):
    import rs_practice_controller as ctrl
    # Warm capture and classifier while the initial Practice control is held.
    await env._capture()
    env._read_screen_state()
    env.page._ppo_pending_practice = None
    env.page._ppo_initial_practice_released = True
    print('[VISIBLE START] Policy warmed; starting the first counted Practice run.',flush=True)
    await pending()
    await ctrl.focus_canvas(env.canvas)
    began = time.perf_counter()
    game_votes = menu_votes = 0
    while time.perf_counter()-began < env.startup_timeout_s:
        await asyncio.sleep(.035)
        frame, score, _ = await env._capture()
        state, probs = env._read_screen_state()
        elapsed = time.perf_counter()-began
        active = (state=='GAMEPLAY' and score < env.args.threshold*.55 and
                  elapsed >= max(.06,env.startup_grace_s))
        game_votes = game_votes+1 if active else 0
        menu = state=='MENU' and score >= env.args.threshold*.78
        menu_votes = menu_votes+1 if menu else 0
        if game_votes >= 2:
            env.startup_guard_ms.append(elapsed*1000)
            print(f'[VISIBLE START] First Practice GAMEPLAY confirmed in {elapsed*1000:.0f}ms.',flush=True)
            return frame, env._encode_last_frame()
        # Some local builds enter the menu on Practice. Only use the existing
        # guarded R path after that menu is independently established.
        if menu_votes >= 3 and elapsed >= .7:
            return await restart()
    env.startup_failures += 1
    raise GuardUncertainError('Initial Practice never reached confirmed gameplay/menu; no policy keys sent')


def install(trainer):
    build = trainer.build_env_class
    def visible_build():
        base = build()
        class ReadyVisibleEnv(base):
            async def _start_episode(self):
                pending = getattr(self.page, '_ppo_pending_practice', None)
                if pending is not None:
                    return await start_ready_practice(self, pending, super()._start_episode)
                return await super()._start_episode()
        return ReadyVisibleEnv
    trainer.build_env_class = visible_build
