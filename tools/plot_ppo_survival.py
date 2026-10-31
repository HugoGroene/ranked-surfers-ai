"""Plot saved parallel-run survival metrics after workers have stopped."""
import argparse
import json
import math
from pathlib import Path


def end_plot_mode(mode,preview_port,benchmark_only=False):
    if mode=='auto':
        return 'show' if preview_port is None and not benchmark_only else 'off'
    if mode not in ('show','save','off'):
        raise ValueError('Unknown end-plot mode')
    return mode


def build_figure(survival,runtime):
    # Imported only after training; importing the trainer never initializes a GUI.
    import matplotlib.pyplot as plt
    requested=float(survival['requested_seconds'])
    if not math.isfinite(requested) or requested<=0:
        raise ValueError('Invalid requested plot duration')
    points=survival.get('points',[])
    if any(not math.isfinite(float(p[key])) or float(p[key])<0
           for p in points for key in ('seconds','mean_seconds')):
        raise ValueError('Invalid survival plot point')
    if any(float(p['seconds'])>requested for p in points):
        raise ValueError('Plot point exceeds requested duration')
    figure,axis=plt.subplots(figsize=(11,5.5),layout='constrained')
    x=[float(p['seconds'])/60 for p in points]
    y=[float(p['mean_seconds']) for p in points]
    window=survival.get('window_episodes',100)
    if points:
        axis.plot(x,y,color='#238b72',linewidth=2,marker='o',markersize=3.5,
            markerfacecolor='#df4040',markeredgecolor='#df4040',
            label=f'Rolling mean (up to {window} completed episodes)')
        axis.legend(loc='best')
    else:
        axis.text(.5,.5,'No completed episodes at sampled times',
            ha='center',va='center',transform=axis.transAxes)
    axis.set_xlim(0,requested/60)
    axis.set_ylim(0,max(5.,max(y,default=0)*1.15))
    axis.set_xlabel('Training elapsed time (minutes; full requested duration)')
    axis.set_ylabel('Rolling mean active survival (seconds)')
    axis.grid(True,alpha=.25)
    mean=survival.get('overall_mean_seconds')
    mean_text=f'{mean:.2f}s' if mean is not None else 'n/a'
    completed=survival.get('completed_episodes',0)
    label='Frozen-policy diagnostic' if runtime.get('benchmark_only') else 'Training'
    axis.set_title(f'{label}: {completed} completed episodes | overall mean {mean_text}\n'
        f'{runtime.get("workers","?")} worker(s) | detector {runtime.get("detector_mode","?")} | '
        f'{runtime.get("status","finished")}')
    figure.supxlabel('Completed episodes pooled across workers; optimizer pauses excluded from survival. '
        f'Time-limit episodes included: {survival.get("time_limited_episodes",0)}.',fontsize=9)
    return figure


def plot_run(folder,show=True):
    folder=Path(folder)
    survival=json.loads((folder/'survival.json').read_text(encoding='utf8'))
    runtime_path=folder/'runtime.json'
    runtime=json.loads(runtime_path.read_text(encoding='utf8')) if runtime_path.exists() else {}
    figure=build_figure(survival,runtime)
    import matplotlib.pyplot as plt
    try:
        # Keep prior plots and results intact, including when replotting a run.
        number=1
        while True:
            output=folder/('survival_curve.png' if number==1 else f'survival_curve_{number}.png')
            try:
                handle=output.open('xb')
                break
            except FileExistsError:
                number+=1
        with handle:
            figure.savefig(handle,format='png',dpi=150)
        print(f'[PPO PLOT] Saved {output}',flush=True)
        if show:
            print('[PPO PLOT] Training has finished. Close the plot window to return to the terminal.',flush=True)
            plt.show(block=True)
        return output
    finally:
        plt.close(figure)


def finish_plot(folder,mode,preview_port,benchmark_only=False):
    selected=end_plot_mode(mode,preview_port,benchmark_only)
    if selected=='off':
        return None
    try:
        return plot_run(folder,show=selected=='show')
    except Exception as exc:
        # Model/report persistence and successful training must not depend on a GUI.
        print(f'[PPO PLOT] Could not display/save the curve: {exc}. Saved training results are preserved.',flush=True)
        return None


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report',type=Path,required=True)
    parser.add_argument('--save-only',action='store_true',help='Write a PNG without opening a window')
    args=parser.parse_args()
    plot_run(args.report,show=not args.save_only)


if __name__=='__main__':
    main()
