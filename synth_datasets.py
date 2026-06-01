import numpy as np
from pathlib import Path

rng = np.random.default_rng(42)

def generate_step(n, dwell_mean, noise_std, n_levels=2):
    levels = np.linspace(-1, 1, n_levels)
    signal = np.zeros(n)
    i = 0
    current_level = rng.choice(levels)
    while i < n:
        dwell = max(1, int(rng.geometric(1/dwell_mean)))
        signal[i:i+dwell] = current_level
        i += dwell
        current_level = rng.choice([l for l in levels if l != current_level])
    return signal + rng.normal(0, noise_std, n)

def generate_piecewise_linear(n, segment_length_mean, noise_std):
    signal = np.zeros(n)
    i = 0
    current_value = rng.uniform(-1, 1)
    while i < n:
        length = max(2, int(rng.geometric(1/segment_length_mean)))
        target = rng.uniform(-1, 1)
        segment = np.linspace(current_value, target, min(length, n-i))
        signal[i:i+len(segment)] = segment
        current_value = target
        i += len(segment)
    return signal + rng.normal(0, noise_std, n)

def generate_sinusoidal(n, frequency, noise_std):
    t = np.arange(n)
    signal = np.sin(2 * np.pi * frequency * t)
    return signal + rng.normal(0, noise_std, n)

def generate_ar(n, ar_coef, noise_std):
    # AR(1) for simplicity, ar_coef controls temporal dependence
    signal = np.zeros(n)
    signal[0] = rng.normal(0, noise_std)
    for i in range(1, n):
        signal[i] = ar_coef * signal[i-1] + rng.normal(0, noise_std)
    return signal

def generate_synthetic_datasets(n=1000, n_instances=50):
    """Generate all synthetic dataset classes with varying parameters."""
    datasets = {}
    
    # Class 1: Step functions — vary dwell time
    for dwell in [1, 2, 4, 8, 16, 32, 64]:
        for noise in [0.0, 0.01, 0.1, 1.0]:
            key = f'step_dwell{dwell}_noise{noise}'
            data = np.concatenate([
                generate_step(n, dwell, noise) 
                for _ in range(n_instances)
            ])
            datasets[key] = data.astype(np.float32)
    
    # Class 2: Piecewise linear — vary segment length
    for seg_len in [2, 4, 8, 16, 32, 64, 128]:
        for noise in [0.0, 0.01, 0.1, 1.0]:
            key = f'linear_seg{seg_len}_noise{noise}'
            data = np.concatenate([
                generate_piecewise_linear(n, seg_len, noise)
                for _ in range(n_instances)
            ])
            datasets[key] = data.astype(np.float32)
    
    # Class 3: Sinusoidal — vary frequency and noise
    for freq in [0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0]:
        for noise in [0.0, 0.01, 0.1, 1.0]:
            key = f'sine_freq{freq}_noise{noise}'
            data = np.concatenate([
                generate_sinusoidal(n, freq, noise)
                for _ in range(n_instances)
            ])
            datasets[key] = data.astype(np.float32)
    
    # Class 4: AR process — vary AR coefficient
    for ar_coef in [0.0, 0.5, 0.8, 0.9, 0.95, 0.99, 0.999]:
        for noise in [0.0, 0.01, 0.1, 1.0]:
            key = f'ar_coef{ar_coef}_noise{noise}'
            data = np.concatenate([
                generate_ar(n, ar_coef, noise)
                for _ in range(n_instances)
            ])
            datasets[key] = data.astype(np.float32)
    
    return datasets

def save_synthetic_datasets(output_dir='/run/media/fabianr/CORPUS_SSD/quantization_datasets/raw/synthetic_datasets'):
    Path(output_dir).mkdir(exist_ok=True)
    datasets = generate_synthetic_datasets()
    for name, data in datasets.items():
        np.save(f'{output_dir}/{name}.npy', data)
    print(f"Saved {len(datasets)} datasets to {output_dir}/")
    return datasets

# Generate and save
datasets = save_synthetic_datasets()



def parse_synthetic_name(name):
    """Parse dataset name back into signal class and parameters."""
    parts = name.replace('.npy', '').split('_')
    signal_class = parts[0]
    params = {}
    for part in parts[1:]:
        # e.g. 'dwell16' -> {'dwell': 16}
        key = ''.join(filter(str.isalpha, part))
        val = float(''.join(filter(lambda c: c.isdigit() or c == '.', part)))
        params[key] = val
    return signal_class, params

def plot_synthetic_results(results_df, entropy_coder='bsc'):
    import matplotlib.pyplot as plt
    
    subset = results_df[results_df['entropy_coder'] == entropy_coder].copy()
    
    # Parse signal class and parameters from dataset name
    subset['signal_class'] = subset['dataset'].apply(
        lambda x: x.split('/')[-1].split('_')[0]
    )
    
    signal_classes = {
        'step': ('dwell', 'Dwell time (samples)'),
        'linear': ('seg', 'Segment length (samples)'),
        'sine': ('freq', 'Frequency'),
        'ar': ('coef', 'AR coefficient'),
    }
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    error_bounds = [1e-5, 1e-3, 1e-1]
    colors = {1e-5: 'steelblue', 1e-3: 'darkorange', 1e-1: 'green'}
    compressors = ['lorenzo', 'mylfzip']
    linestyles = {'lorenzo': '-', 'mylfzip': '--'}
    
    for ax, (sig_class, (param_key, param_label)) in zip(
        axes.flatten(), signal_classes.items()
    ):
        class_data = subset[subset['signal_class'] == sig_class].copy()
        
        # Extract parameter value from dataset name
        class_data['param'] = class_data['dataset'].apply(
            lambda x: float(''.join(
                filter(lambda c: c.isdigit() or c == '.', 
                       [p for p in x.split('/')[-1].split('_') 
                        if param_key in p][0].replace(param_key, '')
                )))
        )
        
        # Focus on noise=0.01 for clarity, or average across noise levels
        # Use noise=0.01 as the representative case
        class_data = class_data[
            class_data['dataset'].str.contains('noise0.01')
        ]
        
        quant = class_data[class_data['compressor_short'] == 'quant']
        
        for comp in compressors:
            comp_data = class_data[class_data['compressor_short'] == comp]
            merged = quant.merge(
                comp_data[['dataset', 'error_bound', 'comp_ratio']],
                on=['dataset', 'error_bound'],
                suffixes=('_quant', '_comp')
            )
            merged['ratio_factor'] = merged['comp_ratio_quant'] / merged['comp_ratio_comp']
            
            for eb in error_bounds:
                eb_data = merged[merged['error_bound'] == eb].groupby('param')['ratio_factor'].median()
                ax.plot(
                    eb_data.index, eb_data.values,
                    linestyle=linestyles[comp],
                    color=colors[eb],
                    label=f'{comp} δ={eb:.0e}',
                    marker='o', markersize=4
                )
        
        ax.axhline(y=1.0, color='black', linestyle=':', linewidth=1, label='break-even')
        ax.set_xlabel(param_label)
        ax.set_ylabel('Compression ratio factor\n(quant / competitor)')
        ax.set_title(sig_class.capitalize())
        ax.set_xscale('log')
        if sig_class == 'ar':
            ax.set_xscale('linear')
        ax.legend(fontsize=7, ncol=2)
        ax.grid(alpha=0.3)
    
    fig.suptitle('Synthetic dataset analysis: quant vs prediction (BSC, noise=0.01)', fontsize=12)
    plt.tight_layout()
    plt.savefig('synthetic_analysis.pdf', bbox_inches='tight')
    plt.show()