"""Launch only after the active experiment has finished. Never sends gameplay inputs."""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))


def main():
    # Fixed safe baseline; arbitrary policy crop/resolution is an offline export choice.
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fps', type=int, default=20, choices=range(5, 41))
    parser.add_argument('--max-run-seconds', type=int, default=45)
    parser.add_argument('--root', type=Path, default=ROOT/'ai_data/demonstrations_native_v1')
    a = parser.parse_args()
    import recorder_snapshot
    sys.argv = ['recorder_snapshot', 'record', '--root', str(a.root), '--fps', str(a.fps),
                '--max-run-seconds', str(a.max_run_seconds), '--framing-profile', 'canvas_v1',
                '--detector-mode', 'off', '--url', 'http://localhost:8000']
    recorder_snapshot.main()


if __name__ == '__main__':
    main()
