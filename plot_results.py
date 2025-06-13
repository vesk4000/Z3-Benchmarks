#!/usr/bin/env python3
# Generate SVG cactus, scatter, and critical-difference plots from a BenchExec CSV table
import os
import numpy as np
import matplotlib.pyplot as plt
import itertools
import pandas as pd
from pathlib import Path
import csv
from Orange.evaluation import compute_CD, graph_ranks

INPUT = Path("results/results.2025-06-09_08-27-08.table.csv")
OUTPUT = Path("../visualizations/")

def parse_results(filepath: Path):
    """
    Parse BenchExec CSV table: three-row multi-index header.
    Returns dict of configs -> {filename: {'status', 'wall_time'}}
    """
    if filepath.suffix.lower() != '.csv':
        raise ValueError(f"Unsupported input format: {filepath}")
    # read table: auto-detect delimiter
    with open(filepath, newline='') as f:
        sample = f.read(2048)
        delim = csv.Sniffer().sniff(sample).delimiter
    df = pd.read_csv(filepath, sep=delim, header=[0,1,2], index_col=0)
    configs = df.columns.get_level_values(1).unique()
    data = {}
    for config in df.columns.get_level_values(1).unique():
        sub = df.xs(config, axis=1, level=1)
        # drop tool-name level
        sub.columns = sub.columns.droplevel(0)
        data[config] = {}
        for fname, row in sub.iterrows():
            status = row.get('status')
            wall = row.get('walltime (s)', np.nan)
            try:
                wall = float(wall)
            except:
                wall = np.nan
            data[config][fname] = {'status': status, 'wall_time': wall}
    return data

def plot_quantile(data, outdir):
    plt.figure()
    for cfg, results in data.items():
        times = sorted([v['wall_time'] for v in results.values() if v['status'] != 'TIMEOUT'])
        if times:
            xs = np.arange(1, len(times)+1)
            plt.step(xs, times, where='post', label=cfg)
    plt.xlabel('Number of instances solved')
    plt.ylabel('Wall time (s)')
    plt.legend()
    plt.grid(True)
    p = os.path.join(outdir, 'quantile.svg')
    plt.savefig(p, format='svg')
    plt.close()

def plot_scatter(data, outdir):
    all_times = [v['wall_time'] for results in data.values() for v in results.values() if not np.isnan(v['wall_time'])]
    max_time = max(all_times) if all_times else 0
    for a, b in itertools.combinations(data.keys(), 2):
        x, y = [], []
        for fn in set(data[a]) & set(data[b]):
            ta = data[a][fn]['wall_time'] if data[a][fn]['status'] != 'TIMEOUT' else max_time
            tb = data[b][fn]['wall_time'] if data[b][fn]['status'] != 'TIMEOUT' else max_time
            x.append(ta)
            y.append(tb)
        plt.figure()
        plt.scatter(x, y, alpha=0.5)
        lim = max(max(x, default=0), max(y, default=0))
        plt.plot([0, lim], [0, lim], 'k--')
        plt.xlabel(f'Wall time ({a})')
        plt.ylabel(f'Wall time ({b})')
        plt.title(f'Scatter: {a} vs {b}')
        plt.grid(True)
        fname = f'scatter_{a.replace(" ","_")}_vs_{b.replace(" ","_")}.svg'
        plt.savefig(os.path.join(outdir, fname), format='svg')
        plt.close()

def plot_critical_difference(data, outdir):
    """Standalone CD plot: draw once and save without extra windows."""
    # use our draw_cd helper to render into a single figure
    fig, ax = plt.subplots(figsize=(8, 4))
    draw_critical_difference(ax, data)
    p = os.path.join(outdir, 'critical_difference.svg')
    fig.savefig(p, format='svg')

def draw_quantile(ax, data):
    """Draw quantile plot on given axes without showing or saving."""
    for cfg, results in data.items():
        times = sorted([v['wall_time'] for v in results.values() if v['status'] != 'TIMEOUT'])
        if times:
            xs = np.arange(1, len(times)+1)
            ax.step(xs, times, where='post', label=cfg)
    ax.set_xlabel('Number of instances solved')
    ax.set_ylabel('Wall time (s)')
    ax.legend()
    ax.grid(True)

def draw_scatter(ax, data, a, b):
    """Draw scatter plot for two configs on given axes."""
    all_times = [v['wall_time'] for results in data.values() for v in results.values() if not np.isnan(v['wall_time'])]
    max_time = max(all_times) if all_times else 0
    x, y = [], []
    for fn in set(data[a]) & set(data[b]):
        ta = data[a][fn]['wall_time'] if data[a][fn]['status'] != 'TIMEOUT' else max_time
        tb = data[b][fn]['wall_time'] if data[b][fn]['status'] != 'TIMEOUT' else max_time
        x.append(ta)
        y.append(tb)
    ax.scatter(x, y, alpha=0.5)
    lim = max(max(x, default=0), max(y, default=0))
    ax.plot([0, lim], [0, lim], 'k--')
    ax.set_xlabel(f'Wall time ({a})')
    ax.set_ylabel(f'Wall time ({b})')
    ax.grid(True)

def draw_critical_difference(ax, data):
    """Render critical-difference diagram into ax via an Agg canvas snapshot."""
    cfgs = list(data.keys())
    fns = sorted(set().union(*[set(d.keys()) for d in data.values()]))
    n = len(fns)
    mat = np.zeros((len(cfgs), n))
    all_times = [v['wall_time'] for results in data.values() for v in results.values() if not np.isnan(v['wall_time'])]
    max_time = max(all_times) if all_times else 0
    for i, cfg in enumerate(cfgs):
        for j, fn in enumerate(fns):
            mat[i, j] = data[cfg][fn]['wall_time'] if fn in data[cfg] and data[cfg][fn]['status'] != 'TIMEOUT' else max_time
    ranks = np.array([pd.Series(mat[:, j]).rank(method='average').values for j in range(n)]).T
    mean_ranks = np.mean(ranks, axis=1)
    cd = compute_CD(mean_ranks, n, alpha='0.05', test='nemenyi')
    # draw on a temporary Agg canvas
    from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
    import matplotlib.pyplot as _plt
    tmp_fig = _plt.Figure(figsize=(6, 2), dpi=100)
    canvas = FigureCanvas(tmp_fig)
    # ax_tmp = tmp_fig.add_subplot(111)
    graph_ranks(mean_ranks, cfgs, cd=cd, width=6, filename=os.path.join(OUTPUT, 'critical_difference.svg'))

    # canvas.draw()
    # # capture as RGB array
    # w, h = canvas.get_width_height()
    # # get RGBA buffer and drop alpha channel
    # arr = np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8)
    # buf = arr.reshape(h, w, 4)[..., :3]
    # # display in the target ax
    # ax.imshow(buf)
    # ax.axis('off')

def interactive_plots(data):
    """Show overview grid of all plots, then allow navigating to full-size zoomed views."""
    import matplotlib.pyplot as plt, matplotlib.gridspec as gridspec, numpy as np
    # prepare plot definitions
    plot_defs = []
    plot_defs.append(('Quantile Plot', lambda ax: draw_quantile(ax, data)))
    cfgs = list(data.keys())
    for a, b in itertools.combinations(cfgs, 2):
        plot_defs.append((f'Scatter: {a} vs {b}', lambda ax, a=a, b=b: draw_scatter(ax, data, a, b)))
    #plot_defs.append(('Critical Difference', lambda ax: draw_critical_difference(ax, data)))
    n = len(plot_defs)
    # determine grid size for overview
    cols = int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n / cols))
    fig = plt.figure(figsize=(12, 8))
    # create grid axes for overview
    gs = gridspec.GridSpec(rows, cols, figure=fig)
    overview_axes = []
    for i, (title, func) in enumerate(plot_defs):
        ax = fig.add_subplot(gs[i // cols, i % cols])
        func(ax)
        ax.set_title(title, fontsize=9)
        overview_axes.append(ax)
    # create full-size axes (initially hidden)
    full_axes = []
    for title, func in plot_defs:
        # create a full-figure axes for zoomed view
        axf = fig.add_axes([0.1, 0.1, 0.8, 0.8])
        func(axf)
        axf.set_title(title)
        axf.set_visible(False)
        full_axes.append(axf)
    # navigation state: 0 = overview, 1..n = full views
    state = {'i': 0}
    def show_state():
        # overview
        if state['i'] == 0:
            for ax in overview_axes: ax.set_visible(True)
            for ax in full_axes: ax.set_visible(False)
        else:
            for ax in overview_axes: ax.set_visible(False)
            for j, axf in enumerate(full_axes, start=1):
                axf.set_visible(j == state['i'])
        fig.canvas.draw_idle()
    def on_click(event):
        # click on overview thumbnail to zoom in, click anywhere in zoom to return overview
        if state['i'] == 0 and event.inaxes in overview_axes:
            idx = overview_axes.index(event.inaxes) + 1
            state['i'] = idx
        elif state['i'] != 0:
            state['i'] = 0
        show_state()
    fig.canvas.mpl_connect('button_press_event', on_click)
    show_state()
    plt.tight_layout()
    plt.show()

def main():
    os.makedirs(OUTPUT, exist_ok=True)
    data = parse_results(INPUT)
    plot_quantile(data, OUTPUT)
    plot_scatter(data, OUTPUT)
    #plot_critical_difference(data, OUTPUT)
    draw_critical_difference(data, data)
    #interactive_plots(data)

if __name__ == '__main__':
    main()
