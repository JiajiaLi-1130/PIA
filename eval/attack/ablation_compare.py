"""Plot comparison of four datasets.
This script plots several statistics from JSONL summary files.
"""

import argparse
import json
import logging
from pathlib import Path
import os
from typing import List, Tuple

import matplotlib
import matplotlib.pyplot as plt

# Global plotting style: prefer Calibri, fall back to common sans
# Use Times New Roman for all text; fall back to common serif fonts if unavailable
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['font.serif'] = ['Times New Roman', 'Times', 'DejaVu Serif']
matplotlib.rcParams['axes.unicode_minus'] = False
matplotlib.rcParams['font.size'] = 16
matplotlib.rcParams['axes.titlesize'] = 20
matplotlib.rcParams['axes.labelsize'] = 20
matplotlib.rcParams['xtick.labelsize'] = 16
matplotlib.rcParams['ytick.labelsize'] = 16
matplotlib.rcParams['legend.fontsize'] = 18
# Ensure fonts are embedded in vector outputs
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42


def resolve_path(p: str, base_dir: Path) -> Path:
    """Resolve a path: prefer the given path, then relative to script dir.

    Raises FileNotFoundError if the file cannot be located.
    """
    path = Path(p)
    if path.exists():
        return path
    candidate = base_dir / path
    if candidate.exists():
        return candidate
    raise FileNotFoundError(f"File not found: {p}")


def load_stats(file_path: Path) -> List[dict]:
    """Load a list of dicts from a JSONL file.

    Empty lines are ignored; parse errors are logged and skipped.
    """
    stats = []
    with file_path.open('r', encoding='utf-8') as f:
        for i, line in enumerate(f, start=1):
            s = line.strip()
            if not s:
                continue
            try:
                stats.append(json.loads(s))
            except Exception as e:  # pylint: disable=broad-except
                logging.warning("Failed to parse %s line %d: %s", file_path, i, e)
    return stats


def plot_comparison(datasets: List[Tuple[str, str]], output: Path, show: bool = True, dpi: int = 300) -> None:
    """Plot comparison charts for supplied datasets.

    Args:
        datasets: list of (file_path, display_name) pairs.
        output: output image path.
        show: whether to call ``plt.show()``.
        dpi: output resolution for raster images.
    """
    script_dir = Path(__file__).resolve().parent

    # Parse and load data
    data_list = []
    for file_path, name in datasets:
        p = resolve_path(file_path, script_dir)
        stats = load_stats(p)
        if not stats:
            logging.warning("File %s has no data or failed to parse", p)
        data_list.append((name, stats))
    # Use a clean white background style
    for style_try in ('default', 'ggplot', 'classic'):
        try:
            plt.style.use(style_try)
            break
        except Exception:
            continue
    # Ensure white figure/axes background
    matplotlib.rcParams['figure.facecolor'] = 'white'
    matplotlib.rcParams['axes.facecolor'] = 'white'
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.reshape(4)

    # Fields to plot and their English labels
    fields = [
        ("avg_rta", "Average RTA"),
        ("avg_asr", "Average ASR"),
        ("min_rta", "Minimum RTA"),
        ("max_asr", "Maximum ASR"),
    ]

    # Determine model and method for each dataset, map method -> color, model -> marker
    from matplotlib.lines import Line2D

    def extract_model_and_method(display_name: str):
        """Extract model and method from a display name string.

        The heuristic groups tokens so that model identifiers (e.g., containing
        '8B') are kept together; remaining tokens are treated as the method.
        This is tolerant of different naming conventions.
        """
        toks = display_name.split('-')
        # find token index that contains 'B' (e.g., '8B', '7B')
        model_end = None
        for i, t in enumerate(toks):
            if 'B' in t or 'b' in t:
                model_end = i
                break
        if model_end is None:
            # no obvious model token with 'B'; treat first token as model
            if len(toks) > 1 and toks[1].lower() == 'instruct':
                model = '-'.join(toks[:2])
                method_tokens = toks[2:]
            else:
                model = toks[0]
                method_tokens = toks[1:]
        else:
            next_idx = model_end + 1
            if next_idx < len(toks) and toks[next_idx].lower() == 'instruct':
                model = '-'.join(toks[: next_idx + 1])
                method_tokens = toks[next_idx + 1:]
            else:
                model = '-'.join(toks[: model_end + 1])
                method_tokens = toks[model_end + 1:]
        # join remaining tokens as method (so 'Persona-GA' stays intact)
        if method_tokens:
            method = '-'.join(method_tokens)
        else:
            method = toks[-1]
        return model, method

    # method -> color mapping (case-insensitive keys)
    method_color_map = {
        'ours': '#ff9a00',
        'w/o ucb': '#3490de',
        'w/o lineage': '#1fab89',
    }
    default_method_colors = ['#ff9a00', '#3490de', '#1fab89']

    # model -> marker mapping (consistent marker per model)
    model_marker_map = {}
    default_markers = ['o', 's', '^', 'D', 'v', 'P', 'X']

    # Precompute model/method for datasets
    parsed = []  # list of (name, stats, model, method)
    for name, stats in data_list:
        model, method = extract_model_and_method(name)
        parsed.append((name, stats, model, method))

    # Assign colors to methods (match keys case-insensitively)
    assigned_method_colors = {}
    for _name, _stats, _model, method in parsed:
        key = (method or '').strip().lower()
        if key in method_color_map:
            assigned_method_colors[method] = method_color_map[key]
        elif method not in assigned_method_colors:
            assigned_method_colors[method] = default_method_colors[len(assigned_method_colors) % len(default_method_colors)]

    # Assign markers to models: first model -> circle, second -> triangle, others use defaults
    for _name, _stats, model, _method in parsed:
        if model not in model_marker_map:
            if len(model_marker_map) == 0:
                model_marker_map[model] = 'o'  # circle for first model
            elif len(model_marker_map) == 1:
                model_marker_map[model] = '^'  # triangle for second model
            else:
                model_marker_map[model] = default_markers[len(model_marker_map) % len(default_markers)]

    # Plot each dataset using color(method) and marker(model)
    for idx, (field, title_en) in enumerate(fields):
        ax = axes[idx]
        for name, stats, model, method in parsed:
            color = assigned_method_colors.get(method, '#333333')
            marker = model_marker_map.get(model, 'o')
            gens = [s.get('generation') for s in stats]
            vals = [s.get(field, None) for s in stats]
            if not gens or not any(v is not None for v in vals):
                logging.debug("Skip missing %s data for %s", field, name)
                continue
            # For small-size embedding: use thinner lines and smaller markers
            ax.plot(gens, vals, color=color, linewidth=0.8, marker=marker, markersize=3, markeredgewidth=0.4, label=name)
        # Explicitly set fontsize to avoid style overrides
        ax.set_title('', fontsize=20)
        ax.set_xlabel('Generation', fontsize=20)
        ax.set_ylabel(title_en, fontsize=20)
        # Ensure tick labels also use 16pt
        ax.tick_params(axis='both', labelsize=16)
        ax.grid(True, alpha=0.3)

    # Compose legend: methods first, then models
    u_models = sorted(list(model_marker_map.keys()))
    u_methods = sorted(list(assigned_method_colors.keys()))

    # Model markers
    row1_h = [
        Line2D([0], [0], color='black', marker=model_marker_map[m], linestyle='', markersize=8, markeredgewidth=0.8)
        for m in u_models
    ]
    row1_l = u_models

    # Method lines
    row2_h = [
        Line2D([0], [0], color=assigned_method_colors[met], lw=3.0)
        for met in u_methods
    ]
    row2_l = u_methods

    # Combine into single legend: methods then models
    final_handles = row2_h + row1_h
    final_labels = row2_l + row1_l

    leg = axes[0].legend(
        final_handles,
        final_labels,
        loc='upper right',
        ncol=2,
        columnspacing=1.0,
        handletextpad=0.2,
        handlelength=1.0,
        fontsize=18,
        frameon=True,
    )

    # Left-align legend text
    for text in leg.get_texts():
        text.set_ha('left')
    # End legend adjustments
    
    
    plt.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    # Save vector formats for crisp rendering at small sizes, then raster versions
    out_svg = output.with_suffix('.svg')
    out_pdf = output.with_suffix('.pdf')
    # keep original raster filename (often .png)
    out_png = output if output.suffix.lower() == '.png' else output.with_suffix('.png')
    out_png_hd = out_png.with_name(out_png.stem + '_hd.png')

    try:
        fig.savefig(out_svg, bbox_inches='tight')
        fig.savefig(out_pdf, bbox_inches='tight')
        # high-DPI PNG for cases where raster is required
        fig.savefig(out_png_hd, dpi=max(dpi, 600), bbox_inches='tight')
        fig.savefig(out_png, dpi=dpi, bbox_inches='tight')
        logging.info("Saved images: %s, %s, %s, %s", out_svg, out_pdf, out_png_hd, out_png)
    except Exception:
        fig.savefig(output, dpi=dpi, bbox_inches='tight')
        logging.info("Saved image: %s", output)
    if show:
        plt.show()


def default_datasets() -> List[Tuple[str, str]]:
    # Base path is taken from environment to avoid embedding internal paths
    base = os.environ.get('ATTACK_BASE', 'persona/attack')
    # Anonymized model display names to avoid leaking internal model IDs
    return [
        (f'{base}/attack1/generation_elite_stats.jsonl', 'ModelA-Instruct-Ours'),
        (f'{base}/attack1_1/generation_elite_stats.jsonl', 'ModelA-Instruct-w/o UCB'),
        (f'{base}/attack1_2/generation_elite_stats.jsonl', 'ModelA-Instruct-w/o Lineage'),
        (f'{base}/attack2/generation_elite_stats.jsonl', 'ModelB-Instruct-Ours'),
        (f'{base}/attack2_1/generation_elite_stats.jsonl', 'ModelB-Instruct-w/o UCB'),
        (f'{base}/attack2_2/generation_elite_stats.jsonl', 'ModelB-Instruct-w/o Lineage'),
    ]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Plot comparison for four dataset groups')
    p.add_argument('--output', '-o', default='persona/attack/comparison_four_datasets.png', help='output image path')
    p.add_argument('--no-show', action='store_true', help='do not display the figure')
    p.add_argument('--dpi', type=int, default=300, help='output resolution in DPI')
    return p.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
    args = parse_args()
    datasets = default_datasets()
    out = Path(args.output)
    plot_comparison(datasets, out, show=(not args.no_show), dpi=args.dpi)


if __name__ == '__main__':
    main()