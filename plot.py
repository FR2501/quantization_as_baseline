import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import seaborn as sns

from synth_datasets import generate_ar, generate_piecewise_linear, generate_sinusoidal, generate_step

sns.set_theme(style="whitegrid", context="paper", font_scale=1.5, palette=sns.color_palette("colorblind"))

# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

CODER_ORDER = ["huffman", "gzip", "zstd", "bsc"]
PRED_ORDER  = ["quant", "lorenzo", "nlms-32"]

PRED_COLORS = {"quant": "steelblue", "lorenzo": "darkorange", "nlms-32": "green"}
PRED_LABELS = {"quant": "Quant",     "lorenzo": "Lorenzo",    "nlms-32": "NLMS"}

cleaned_datasets_root = "."  # update this path as needed


def comp_rate_boxplot(result_df):
    df_plot = result_df[(result_df['error_bound'] == 1e-1) & (result_df['dataset'].str.startswith('ucr_tsca'))]

    g = sns.catplot(
        data=df_plot,
        x="compressor_short",
        y="comp_ratio",
        col="entropy_coder",
        col_order=CODER_ORDER,
        order=PRED_ORDER,
        kind="box",
        col_wrap=4,
        height=5,
        aspect=1.1,
        linewidth=1,
        fliersize=5,
        sharex=True,
        hue='compressor_short',
        palette=PRED_COLORS,
    )
    g.set_titles("{col_name}")

    for ax in g.axes.flat:
        ax.set_xlabel("")

    g.figure.subplots_adjust(right=0.85, wspace=0.25, hspace=0.3)
    plt.yscale("log")

    plt.tight_layout()
    plt.savefig("comp_rate_boxplot.pdf")


def byte_entropy(result_df):
    df_bsc = result_df[(result_df["entropy_coder"] == "bsc") & (result_df['dataset'].str.startswith('ucr_tsca'))]

    g = sns.FacetGrid(
        df_bsc,
        col="compressor_short",
        col_order=PRED_ORDER,
        hue="compressor_short",
        palette=PRED_COLORS,
        height=4,
        aspect=1,
        row_order=PRED_ORDER,
    )

    g.map_dataframe(
        sns.scatterplot,
        x="entropy_b",
        y="comp_ratio",
        s=35,
    )

    g.set_axis_labels("", "Compression Ratio")
    g.set_titles("{col_name}")

    g.figure.supxlabel("Byte-wise entropy")

    plt.yscale("log")

    plt.tight_layout()
    plt.savefig("byte_entropy_vs_comp.pdf")


def int_entropy(result_df):
    df_bsc = result_df[(result_df["entropy_coder"] == "bsc") & (result_df['dataset'].str.startswith('ucr_tsca'))]

    g = sns.FacetGrid(
        df_bsc,
        col="compressor_short",
        col_order=PRED_ORDER,
        hue="compressor_short",
        palette=PRED_COLORS,
        height=4,
        aspect=1,
        row_order=PRED_ORDER,
    )

    g.map_dataframe(
        sns.scatterplot,
        x="int_entropy",
        y="comp_ratio",
        s=35,
    )

    g.set_axis_labels("", "Compression Ratio")
    g.set_titles("{col_name}")

    g.figure.supxlabel("Integer-wise entropy")

    plt.yscale("log")

    plt.tight_layout()
    plt.savefig("int_entropy_vs_comp.pdf")


def plot_entropy_vs_compression(df, error_bound=0.1, entropy_coder='bsc'):
    subset = df[
        (df['error_bound'] == error_bound) &
        (df['entropy_coder'] == entropy_coder) &
        (df['dataset'].str.startswith('ucr_tsca'))
    ].copy()

    entropy_cols = {
        'entropy_b':   'Byte-wise entropy',
        'cont_entr_1': 'Contextual entropy (k=1)',
        'cont_entr_2': 'Contextual entropy (k=2)',
        'cont_entr_4': 'Contextual entropy (k=4)',
    }

    # Melt to long form so FacetGrid can facet on both compressor and entropy type
    long = subset.melt(
        id_vars=['dataset', 'compressor_short', 'comp_ratio'],
        value_vars=list(entropy_cols.keys()),
        var_name='entropy_col',
        value_name='entropy_val',
    )
    long['entropy_label'] = long['entropy_col'].map(entropy_cols)
    long['compressor_short'] = pd.Categorical(long['compressor_short'], categories=PRED_ORDER, ordered=True)
    long['entropy_label'] = pd.Categorical(long['entropy_label'], categories=list(entropy_cols.values()), ordered=True)

    g = sns.FacetGrid(
        long,
        row='compressor_short',
        col='entropy_label',
        row_order=PRED_ORDER,
        col_order=list(entropy_cols.values()),
        sharey=True,
        sharex='col',
        height=2.5,
        aspect=1.1,
    )

    def draw_scatter(data, **kwargs):
        ax = plt.gca()
        comp = data['compressor_short'].iloc[0]
        ax.scatter(
            data['entropy_val'],
            data['comp_ratio'],
            alpha=0.6,
            s=20,
            color=PRED_COLORS[comp],
        )
        ax.set_yscale('log')

        x = data['entropy_val'].values
        y = np.log10(data['comp_ratio'].values)
        mask = np.isfinite(x) & np.isfinite(y)
        if mask.sum() > 2:
            r2 = np.corrcoef(x[mask], y[mask])[0, 1] ** 2
            ax.text(0.95, 0.95, f'R²={r2:.2f}',
                    transform=ax.transAxes,
                    ha='right', va='top', fontsize=9,
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

    g.map_dataframe(draw_scatter)

    g.set_titles(row_template='', col_template='{col_name}')
    g.set_axis_labels('Entropy (bits)', '')

    # Row labels (compressor name + y-axis label) on the leftmost axes
    for ax, comp in zip(g.axes[:, 0], PRED_ORDER):
        ax.set_ylabel(f'{PRED_LABELS[comp]}\nCompression Ratio', fontsize=9)

    g.figure.suptitle(
        f'Compression ratio vs. entropy measures (δ={error_bound}, {entropy_coder.upper()})',
        fontsize=12, y=1.02,
    )
    plt.tight_layout()
    g.figure.savefig(
        f'entropy_vs_compression_{error_bound}_{entropy_coder}.pdf',
        bbox_inches='tight',
    )
    plt.show()


def plot_error_bound_dependency(df, entropy_coder='bsc'):
    subset = df[(df['entropy_coder'] == entropy_coder) & (df['dataset'].str.startswith('ucr_tsca'))].copy()
    
    error_bounds = [1e-5, 1e-3, 1e-1]
    compressors = PRED_ORDER

    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True)

    for col, eb in enumerate(error_bounds):
        ax = axes[col]
        eb_data = subset[subset['error_bound'] == eb]

        plot_data   = [eb_data[eb_data['compressor_short'] == c]['comp_ratio'].dropna().values for c in compressors]
        print(plot_data)
        plot_colors = [PRED_COLORS[c] for c in compressors]
        plot_labels = [PRED_LABELS[c] for c in compressors]
        
        bp = ax.boxplot(
            plot_data,
            patch_artist=True,
            medianprops=dict(color='black', linewidth=1.5),
            flierprops=dict(marker='o', markersize=3, alpha=0.5),
            widths=0.5
        )
        
        for patch, color in zip(bp['boxes'], plot_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        for flier, color in zip(bp['fliers'], plot_colors):
            flier.set_markerfacecolor(color)
            flier.set_markeredgecolor(color)
        
        ax.set_yscale('log')
        ax.set_xticks([1, 2, 3])
        ax.set_xticklabels(plot_labels)
        ax.set_title(f'δ = {eb:.0e}', fontsize=11)
        
        if col == 0:
            ax.set_ylabel('Compression Ratio', fontsize=10)
        
    
    fig.suptitle(
        f'Compression ratio by error bound (BSC)',
        fontsize=12
    )
    plt.tight_layout()
    plt.savefig('error_bound_dependency_bsc.pdf', bbox_inches='tight')
    plt.show()


def plot_synthetic():
    n = 500

    examples = {
        'Step function\n(dwell=16, noise=0.1)': generate_step(n, 16, 0.1),
        'Piecewise linear\n(seg=64, noise=0.1)': generate_piecewise_linear(n, 64, 0.1),
        'Sinusoidal\n(freq=0.02, noise=0.1)': generate_sinusoidal(n, 0.02, 0.1),
        'AR process\n(coef=0.95, noise=0.1)': generate_ar(n, 0.95, 0.1),
    }

    fig, axes = plt.subplots(2, 2, figsize=(10, 5), sharex=True)
    for ax, (title, data) in zip(axes.flatten(), examples.items()):
        ax.plot(data, linewidth=0.8, color='steelblue')
        ax.set_title(title, fontsize=10)
        ax.set_xlabel('Time step')
        ax.set_ylabel('Value')
        ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig('synthetic_examples.pdf', bbox_inches='tight')
    plt.show()


def plot_dataset_examples():
    # Load one series from each dataset
    # Pick a representative series — first one is fine
    
    datasets = {
        'SmallKitchenAppliances\n(quantization wins)': 
            f'{cleaned_datasets_root}/ucr_tsca/SmallKitchenAppliances.npy',
        'StarLightCurves\n(prediction wins)': 
            f'{cleaned_datasets_root}/ucr_tsca/StarLightCurves.npy',
    }
    
    fig, axes = plt.subplots(2, 1, figsize=(5, 5), sharex=False)
    colors = ['steelblue', 'darkorange']
    
    for ax, (title, path), color in zip(axes, datasets.items(), colors):
        data = np.load(path)
        # Take first complete series — assuming column-major concatenation
        # Adjust slice length to match your series length
        i = 6
        series = data[i*720:(i+1)*720]  # SmallKitchenAppliances is length 720
        
        ax.plot(series, linewidth=0.8, color=color)
        ax.set_title(title, fontsize=10)
        ax.set_ylabel('Value (z-normalized)')
        ax.grid(alpha=0.3)
    
    axes[-1].set_xlabel('Time step')
    
    plt.tight_layout()
    plt.savefig('dataset_examples.pdf', bbox_inches='tight')
    plt.show()


def plot_synthetic_4panel(result_df, entropy_coder='bsc', noise_level=0.0):
    subset = result_df[(result_df['entropy_coder'] == entropy_coder) & (result_df['dataset'].str.startswith('synthetic'))].copy()
    print(subset)
    
    error_bounds = [1e-5, 1e-3, 1e-1]
    eb_colors    = {1e-5: 'maroon', 1e-3: 'crimson', 1e-1: 'lightsalmon'}
    linestyles   = {'lorenzo': '-',  'nlms-32': '--'}
    markers      = {'lorenzo': 'o',  'nlms-32': 's'}
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    panels = [
        {
            'ax': axes[0, 0],
            'filter': f'step.*noise{noise_level}',
            'param_regex': r'dwell(\d+)',
            'param_col': 'dwell',
            'xlabel': 'Mean dwell time (samples)',
            'title': 'Step functions',
            'xscale': 'log2',
        },
        {
            'ax': axes[0, 1],
            'filter': f'linear.*noise{noise_level}',
            'param_regex': r'seg(\d+)',
            'param_col': 'seg',
            'xlabel': 'Mean segment length (samples)',
            'title': 'Piecewise linear',
            'xscale': 'log2',
        },
        {
            'ax': axes[1, 0],
            'filter': f'sine.*noise{noise_level}',
            'param_regex': r'freq(\d+\.?\d*)',
            'param_col': 'freq',
            'xlabel': 'Frequency',
            'title': 'Sinusoidal',
            'xscale': 'log10',
        },
        {
            'ax': axes[1, 1],
            'filter': f'ar.*noise{noise_level}',
            'param_regex': r'coef(\d+\.?\d*)',
            'param_col': 'coef',
            'xlabel': 'AR coefficient',
            'title': 'AR process',
            'xscale': 'linear',
        },
    ]
    
    for panel in panels:
        ax = panel['ax']
        data = subset[
            subset['dataset'].str.contains(panel['filter'], regex=True)
        ].copy()
        
        data['param'] = data['dataset'].str.extract(
            panel['param_regex']
        ).astype(float)
        
        quant = data[data['compressor_short'] == 'quant']
        
        for comp in ['lorenzo', 'nlms-32']:
            comp_data = data[data['compressor_short'] == comp]
            merged = quant.merge(
                comp_data[['dataset', 'error_bound', 'comp_ratio', 'param']],
                on=['dataset', 'error_bound', 'param'],
                suffixes=('_quant', '_comp')
            )
            merged['ratio_factor'] = (
                merged['comp_ratio_quant'] / merged['comp_ratio_comp']
            )
            for eb in error_bounds:
                eb_data = merged[
                    merged['error_bound'] == eb
                ].groupby('param')['ratio_factor'].median()
                ax.plot(
                    eb_data.index, eb_data.values,
                    linestyle=linestyles[comp],
                    color=eb_colors[eb],
                    marker=markers[comp],
                    markersize=5,
                    label=f'{comp} δ={eb:.0e}'
                )
        
        ax.axhline(
            y=1.0, color='black', linestyle=':',
            linewidth=1.5, label='break-even'
        )
        
        if panel['xscale'] == 'log2':
            ax.set_xscale('log', base=2)
        elif panel['xscale'] == 'log10':
            ax.set_xscale('log', base=10)
        
        ax.set_xlabel(panel['xlabel'], fontsize=11)
        ax.set_ylabel('Compression ratio factor\n(quant / competitor)', fontsize=10)
        ax.set_title(panel['title'], fontsize=12)
        ax.legend(fontsize=9, ncol=2)
        ax.grid(alpha=0.3)
    
    fig.suptitle(
        f'Synthetic dataset analysis (BSC, noise σ={noise_level})',
        fontsize=13
    )
    plt.tight_layout()
    plt.savefig('synthetic_analysis_4panel.pdf', bbox_inches='tight')
    plt.show()

def make_plots(result_df, dataset_root):
    global cleaned_datasets_root
    cleaned_datasets_root = dataset_root

    comp_rate_boxplot(result_df)
    byte_entropy(result_df)
    int_entropy(result_df)
    plot_entropy_vs_compression(result_df)
    plot_error_bound_dependency(result_df)
    plot_synthetic()
    plot_dataset_examples()
    plot_synthetic_4panel(result_df, noise_level=0.1)