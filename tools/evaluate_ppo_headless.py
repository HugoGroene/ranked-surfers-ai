"""Frozen evaluation through the validated Electron shared-memory adapter."""
import sys


def main(argv=None):
    import train_rs_v612_ppo as trainer
    from tools.ppo_headless_env import install
    args = trainer.parse_args([
        '--mode', 'eval', '--background-display', '1', '--canvas-policy-input',
        '--framing-profile', 'canvas_v1', '--detector-mode', 'off',
        '--reuse-observation-guard', '--press-delay-ms', '5',
        '--state-model', 'checkpoints/v612_snapshot_20261021/screen_state_candidate_v2.pt',
        *(sys.argv[1:] if argv is None else argv)])
    if args.mode != 'eval':
        raise ValueError('This entry point only evaluates frozen policies')
    install(trainer, 'electron', electron_transport='shared_memory')
    return 0 if trainer.evaluate(args) else 2


if __name__ == '__main__':
    raise SystemExit(main())
