import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from matplotlib.backends.backend_pdf import PdfPages
import pickle
import matplotlib.pyplot as plt
import os
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

def plot_time_iq_signal(signal: torch.Tensor, title: str, config: dict, length: int = None):
    """
    Plot the I/Q signal in time domain using configuration settings.

    Parameters:
    - signal: torch.Tensor with shape [1, 2, N] where 2 corresponds to [I, Q]
    - title: str, title for the plot
    - config: dict, plot configuration settings
    - length: int, optional length to plot from start (default = config['sig_length'])
    """
    signal_np = signal.squeeze(0).numpy()
    length = length or config['sig_length']

    signal_I = signal_np[0, :length]
    signal_Q = signal_np[1, :length]
    x_axis = np.arange(length)

    plt.figure(figsize=config['figure_size'], dpi=config['dpi'])

    if config.get('plot_as_scatter', False):
        plt.scatter(x_axis, signal_I, label='I', color=config['signal_color'],
                    s=config['scatter_size'], alpha=config['alpha'])
        plt.scatter(x_axis, signal_Q, label='Q', color='orange',
                    s=config['scatter_size'], alpha=config['alpha'])
    else:
        plt.plot(x_axis, signal_I, label='I', color=config['signal_color'],
                 linewidth=config['line_width'])
        plt.plot(x_axis, signal_Q, label='Q', color='orange',
                 linewidth=config['line_width'])

    plt.title(title, fontsize=config['title_fontsize'], weight=config['fontweight_title'])
    plt.xlabel('Time', fontsize=config['label_fontsize'], weight=config['fontweight_label'])
    plt.ylabel('Amplitude', fontsize=config['label_fontsize'], weight=config['fontweight_label'])

    plt.xticks(fontsize=config['tick_fontsize'], weight=config['fontweight_label'])
    plt.yticks(fontsize=config['tick_fontsize'], weight=config['fontweight_label'])
    plt.grid(True, linestyle='--', alpha=0.6)

    legend = plt.legend(
        fontsize=config['legend_fontsize'],
        loc=config.get('legend_loc', 'upper center'),
        bbox_to_anchor=config.get('legend_bbox', (0.5, 1.02)),
        ncol=config.get('legend_ncol', 2),
        frameon=False
    )
    for text in legend.get_texts():
        text.set_weight(config['fontweight_label'])

    plt.tight_layout()

    if config.get('save_path'):
        os.makedirs(config['save_path'], exist_ok=True)
        base_filename = os.path.join(config['save_path'], title.replace(" ", "_"))
        formats = config.get('save_formats', ['pdf'])

        for fmt in formats:
            dpi = config['dpi'] if fmt == 'pdf' else config.get('image_format_dpi', config['dpi'])
            filename = f"{base_filename}.{fmt}"
            plt.savefig(filename, bbox_inches='tight', transparent=True, dpi=dpi)
        
    if config.get('show', False):
        plt.show()
    else:
        plt.close()

def plot_iq_constellation(signal: torch.Tensor, title: str, config: dict, length: int = None):
    """
    Scatter plot of I vs Q (IQ constellation) using configuration settings.

    Parameters:
    - signal: torch.Tensor with shape [1, 2, N] where 2 corresponds to [I, Q]
    - title: str, title for the plot
    - config: dict, plot configuration settings
    - length: int, optional number of samples to plot (default = config['sig_length'])
    """
    signal_np = signal.squeeze(0).numpy()
    length = length or config['sig_length']

    signal_I = signal_np[0, :length]
    signal_Q = signal_np[1, :length]

    plt.figure(figsize=config['figure_size'], dpi=config['dpi'])
    plt.scatter(signal_I, signal_Q,
                color=config['signal_color'],
                s=config['scatter_size'],
                alpha=config['alpha'])

    plt.title(title, fontsize=config['title_fontsize'], weight=config['fontweight_title'])
    plt.xlabel('In-phase (I)', fontsize=config['label_fontsize'], weight=config['fontweight_label'])
    plt.ylabel('Quadrature (Q)', fontsize=config['label_fontsize'], weight=config['fontweight_label'])

    plt.xticks(fontsize=config['tick_fontsize'], weight=config['fontweight_label'])
    plt.yticks(fontsize=config['tick_fontsize'], weight=config['fontweight_label'])
    plt.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()

    if config.get('save_path'):
        os.makedirs(config['save_path'], exist_ok=True)
        base_filename = os.path.join(config['save_path'], title.replace(" ", "_"))
        formats = config.get('save_formats', ['pdf'])

        for fmt in formats:
            dpi = config['dpi'] if fmt == 'pdf' else config.get('image_format_dpi', config['dpi'])
            filename = f"{base_filename}.{fmt}"
            plt.savefig(filename, bbox_inches='tight', transparent=True, dpi=dpi)
    if config.get('show', False):
        plt.show()
    else:
        plt.close()
#-------------------------------------------------------------------------------
from scipy.signal import spectrogram

def plot_spectrogram(signal: torch.Tensor, title: str, config: dict, length: int = None):
    """
    Plot the spectrogram (magnitude in dB) of a complex I/Q signal.
    
    Parameters:
    - signal: torch.Tensor with shape [1, 2, N] where 2 is [I, Q]
    - title: str, title for the plot
    - config: dict, plot configuration settings
    - length: int, optional length of signal to use (default = config['sig_length'])
    """
    signal_np = signal.squeeze(0).numpy()
    length = length or config['sig_length']

    i = signal_np[0, :length]
    q = signal_np[1, :length]
    complex_signal = i + 1j * q

    plt.figure(figsize=config['figure_size'], dpi=config['dpi'])
    nfft = config.get('nfft', 256)
    noverlap = config.get('noverlap', nfft // 2)
    cmap = config.get('cmap', 'viridis')

    plt.specgram(complex_signal, NFFT=nfft, Fs=1.0, noverlap=noverlap, cmap=cmap, scale='dB')

    plt.title(f"{title} - Spectrogram", fontsize=config['title_fontsize'], weight=config['fontweight_title'])
    plt.xlabel('Time', fontsize=config['label_fontsize'], weight=config['fontweight_label'])
    plt.ylabel('Frequency', fontsize=config['label_fontsize'], weight=config['fontweight_label'])

    plt.xticks(fontsize=config['tick_fontsize'], weight=config['fontweight_label'])
    plt.yticks(fontsize=config['tick_fontsize'], weight=config['fontweight_label'])

    plt.tight_layout()

    if config.get('save_path'):
        os.makedirs(config['save_path'], exist_ok=True)
        base_filename = os.path.join(config['save_path'], title.replace(" ", "_"))
        formats = config.get('save_formats', ['pdf'])

        for fmt in formats:
            dpi = config['dpi'] if fmt == 'pdf' else config.get('image_format_dpi', config['dpi'])
            filename = f"{base_filename}.{fmt}"
            plt.savefig(filename, bbox_inches='tight', transparent=True, dpi=dpi)
    if config.get('show', False):
        plt.show()
    else:
        plt.close()

#------------------------------------------------------------------------------------------------------------------
def plot_rss_time_series(signal: torch.Tensor, title: str, config: dict, length: int = None):
    """
    Plot the RSS (received signal strength) as a time-series.
    
    Parameters:
    - signal: torch.Tensor with shape [1, 2, N]
    - title: str, title for the plot
    - config: dict, plot configuration settings
    - length: int, optional length to plot (default = config['sig_length'])
    """
    signal_np = signal.squeeze(0).numpy()
    length = length or config['sig_length']

    i = signal_np[0, :length]
    q = signal_np[1, :length]
    rss = np.sqrt(i**2 + q**2)
    x_axis = np.arange(length)

    plt.figure(figsize=config['figure_size'], dpi=config['dpi'])
    plt.plot(x_axis, rss, color=config['signal_color'], linewidth=config['line_width'])

    plt.title(f"{title} - RSS", fontsize=config['title_fontsize'], weight=config['fontweight_title'])
    plt.xlabel('Time', fontsize=config['label_fontsize'], weight=config['fontweight_label'])
    plt.ylabel('RSS (Amplitude)', fontsize=config['label_fontsize'], weight=config['fontweight_label'])

    plt.xticks(fontsize=config['tick_fontsize'], weight=config['fontweight_label'])
    plt.yticks(fontsize=config['tick_fontsize'], weight=config['fontweight_label'])
    plt.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()

    if config.get('save_path'):
        os.makedirs(config['save_path'], exist_ok=True)
        base_filename = os.path.join(config['save_path'], title.replace(" ", "_"))
        formats = config.get('save_formats', ['pdf'])

        for fmt in formats:
            dpi = config['dpi'] if fmt == 'pdf' else config.get('image_format_dpi', config['dpi'])
            filename = f"{base_filename}.{fmt}"
            plt.savefig(filename, bbox_inches='tight', transparent=True, dpi=dpi)
    if config.get('show', False):
        plt.show()
    else:
        plt.close()

#----------------------------------------------------------------------------------------------------------------------------
import pandas as pd

def plot_metrics_to_pdf(pkl_path, output_pdf_path, key_groups, config):
    import pickle
    import matplotlib.pyplot as plt
    import os
    import numpy as np
    import pandas as pd
    from itertools import cycle
    from matplotlib.backends.backend_pdf import PdfPages

    # Load metrics
    with open(pkl_path, 'rb') as f:
        metrics = pickle.load(f)

    os.makedirs(os.path.dirname(output_pdf_path), exist_ok=True)
    pdf = PdfPages(output_pdf_path)

    nan_strategy = config.get('nan_handling', 'ffill')  # 'ffill', 'zero', or 'ignore'

    # Generate consistent unique colors for all keys across all groups
    color_cycle = cycle(plt.cm.tab20.colors)
    key_colors = {key: next(color_cycle) for group in key_groups for key in group}

    for keys in key_groups:
        plt.figure(figsize=config['figure_size'], dpi=config['dpi'])
        ax1 = plt.gca()
        ax2 = None

        y_ranges = []
        for key in keys:
            if key in metrics:
                y = np.array(metrics[key])
                if nan_strategy == 'ffill':
                    y = pd.Series(y).fillna(method='ffill').fillna(0).to_numpy()
                elif nan_strategy == 'zero':
                    y = np.nan_to_num(y, nan=0.0)
                # if 'ignore', we just leave NaNs
                y_ranges.append((np.nanmin(y), np.nanmax(y)))

        use_second_axis = (
            len(keys) >= 2 and
            any(abs(r2[1] - r2[0]) > 5 * abs(r1[1] - r1[0]) for r1 in y_ranges for r2 in y_ranges)
        )

        for i, key in enumerate(keys):
            if key not in metrics:
                print(f"Warning: Key '{key}' not found in metrics file.")
                continue

            y = np.array(metrics[key])
            if np.isnan(y).any():
                print(f"Warning: NaN values found in '{key}', handled with '{nan_strategy}'")

            if nan_strategy == 'ffill':
                y = pd.Series(y).fillna(method='ffill').fillna(0).to_numpy()
            elif nan_strategy == 'zero':
                y = np.nan_to_num(y, nan=0.0)
            # if 'ignore', keep y as-is

            label = key.replace('train_', '').replace('_', ' ').title()
            color = key_colors.get(key, None)
            axis = ax1 if (i == 0 or not use_second_axis) else (ax2 or ax1.twinx())
            if axis == ax1:
                axis.plot(y, label=label, linewidth=config['line_width'], alpha=config['alpha'], color=color)
            else:
                ax2 = axis
                axis.plot(y, label=label, linestyle='--', linewidth=config['line_width'], alpha=config['alpha'], color=color)

        title = " vs. ".join([k.replace('_', ' ').title() for k in keys])
        plt.title("", fontsize=config['title_fontsize'], weight=config['fontweight_title'])
        ax1.set_xlabel('Epoch', fontsize=config['label_fontsize'], weight=config['fontweight_label'])
        ax1.set_ylabel('Value', fontsize=config['label_fontsize'], weight=config['fontweight_label'])
        ax1.tick_params(axis='both', labelsize=config['tick_fontsize'])

        if ax2:
            ax2.set_ylabel('Secondary Value', fontsize=config['label_fontsize'], weight=config['fontweight_label'])
            ax2.tick_params(axis='y', labelsize=config['tick_fontsize'])

        ax1.grid(True, linestyle='--', alpha=0.6)

        handles1, labels1 = ax1.get_legend_handles_labels()
        handles2, labels2 = ax2.get_legend_handles_labels() if ax2 else ([], [])
        legend = ax1.legend(
            handles1 + handles2,
            labels1 + labels2,
            fontsize=config['legend_fontsize'],
            loc=config.get('legend_loc', 'upper right'),
            bbox_to_anchor=config.get('legend_bbox', (1, 1)),
            ncol=config.get('legend_ncol', 1),
            frameon=False
        )
        for text in legend.get_texts():
            text.set_weight(config['fontweight_label'])

        plt.tight_layout()
        pdf.savefig()

        # Save as image if needed
        if config.get('save_path'):
            os.makedirs(config['save_path'], exist_ok=True)
            base_filename = os.path.join(config['save_path'], title.replace(" ", "_"))
            formats = config.get('save_formats', ['pdf'])

            for fmt in formats:
                dpi = config['dpi'] if fmt == 'pdf' else config.get('image_format_dpi', config['dpi'])
                filename = f"{base_filename}.{fmt}"
                plt.savefig(filename, bbox_inches='tight', transparent=True, dpi=dpi)

        if config.get('show', False):
            plt.show()
        else:
            plt.close()

    pdf.close()
    print(f"Metrics plots saved to {output_pdf_path}")

#------------------------------------------------------------------------------------------------------------------------------
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
import os

from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt
import os

def plot_tsne_embeddings(
    embeddings: np.ndarray,
    labels: np.ndarray,
    class_names: list,
    config: dict,
    title: str = "t-SNE Embedding",
    visible_classes: list = None
):
    """
    Plot t-SNE visualization of embeddings with class-based coloring.
    """
    import matplotlib.colors as mcolors

    # Apply t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
    tsne_result = tsne.fit_transform(embeddings)

    # High-contrast distinct colors for up to 16 classes
    custom_colors = [
        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
        '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
        '#bcbd22', '#17becf', '#aec7e8', '#ffbb78',
        '#98df8a', '#ff9896', '#c5b0d5', '#c49c94'
    ]

    # Create figure
    plt.figure(figsize=config['figure_size'], dpi=config['dpi'])

    # Plot each class with a distinct color
    unique_labels = np.unique(labels)
    for i, class_idx in enumerate(unique_labels):
        class_name = class_names[class_idx]
        if visible_classes is not None and class_name not in visible_classes:
            continue

        idx = labels == class_idx
        color = custom_colors[i % len(custom_colors)]  # use custom colors

        plt.scatter(
            tsne_result[idx, 0],
            tsne_result[idx, 1],
            label=class_name,
            alpha=config['alpha'],
            s=config['scatter_size'],
            color=color
        )

    # Title and labels
    plt.title("", fontsize=config['title_fontsize'], weight=config['fontweight_title'])
    plt.xlabel("t-SNE Dim 1", fontsize=config['label_fontsize'], weight=config['fontweight_label'])
    plt.ylabel("t-SNE Dim 2", fontsize=config['label_fontsize'], weight=config['fontweight_label'])
    plt.xticks(fontsize=config['tick_fontsize'], weight=config['fontweight_label'])
    plt.yticks(fontsize=config['tick_fontsize'], weight=config['fontweight_label'])
    plt.grid(True, linestyle='--', alpha=0.6)

    # Legend outside
    legend = plt.legend(
        fontsize=config['legend_fontsize'],
        loc='center left',
        bbox_to_anchor=(1.02, 0.5),
        ncol=config.get('legend_ncol', 1),
        frameon=False
    )
    for text in legend.get_texts():
        text.set_weight(config['fontweight_label'])

    plt.subplots_adjust(right=0.78)

    # Save
    if config.get('save_path'):
        os.makedirs(config['save_path'], exist_ok=True)
        base_filename = os.path.join(config['save_path'], title.replace(" ", "_"))
        formats = config.get('save_formats', ['pdf'])

        for fmt in formats:
            dpi = config['dpi'] if fmt == 'pdf' else config.get('image_format_dpi', config['dpi'])
            filename = f"{base_filename}.{fmt}"
            plt.savefig(filename, bbox_inches='tight', transparent=True, dpi=dpi)

    if config.get('show', False):
        plt.show()
    else:
        plt.close()

#-------------------------------------------------------------------------------------------------------------------------
def plot_original_and_modified_iq_data(
    signals_original, signals_modified, old_targets,
    config: dict, save_path: str,
    start_trigger: int = 0, end_trigger: int = 512
):
    """
    Plots and saves time-domain and frequency-domain (magnitude & phase) representations
    of original vs. modified IQ signals using shared config settings.

    Parameters:
    - signals_original: torch.Tensor [B, 1, 2, 512]
    - signals_modified: torch.Tensor [B, 1, 2, 512]
    - old_targets: torch.Tensor or list of target indices
    - config: dict with plotting configuration (shared across all functions)
    - save_path: str, path to save the final multi-page PDF
    - start_trigger, end_trigger: int, region to highlight in time/frequency
    """

    signal_names = config.get("signal_names", [
        "BPSK", "QPSK", "8PSK", "16APSK", "32APSK", "64APSK", "128APSK",
        "16QAM", "32QAM", "64QAM", "128QAM", "256QAM",
        "AM-DSB-SC", "AM-DSB-WC", "FM", "GMSK"
    ])

    # Prepare signals
    signals_original = signals_original.detach().cpu().numpy()
    signals_modified = signals_modified.detach().cpu().numpy()
    signal_orig = signals_original[0, 0]  # [2, 512]
    signal_mod = signals_modified[0, 0]   # [2, 512]
    I_orig, Q_orig = signal_orig[0], signal_orig[1]
    I_mod, Q_mod = signal_mod[0], signal_mod[1]
    class_name = signal_names[old_targets[0]]
    num_samples = signal_orig.shape[-1]

    # Compute FFT for freq domain
    complex_orig = I_orig + 1j * Q_orig
    complex_mod = I_mod + 1j * Q_mod
    freq_axis = np.fft.fftfreq(num_samples)[start_trigger:end_trigger]

    mag_orig = np.abs(complex_orig)
    mag_mod = np.abs(complex_mod)
    phase_orig = np.angle(complex_orig)
    phase_mod = np.angle(complex_mod)

    fft_mag_orig = np.abs(np.fft.fft(complex_orig))[start_trigger:end_trigger]
    fft_mag_mod = np.abs(np.fft.fft(complex_mod))[start_trigger:end_trigger]
    fft_phase_orig = np.angle(np.fft.fft(complex_orig))[start_trigger:end_trigger]
    fft_phase_mod = np.angle(np.fft.fft(complex_mod))[start_trigger:end_trigger]

    # Plotting
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with PdfPages(save_path) as pdf:

        def styled_plot(x, y_list, labels, title, xlabel, ylabel, highlight=None):
            plt.figure(figsize=config['figure_size'], dpi=config['dpi'])
            for y, label in zip(y_list, labels):
                plt.plot(x, y, label=label, linewidth=config['line_width'])
            if highlight:
                plt.axvspan(*highlight, color='yellow', alpha=0.3, label="Selected Region")
            plt.title(title, fontsize=config['title_fontsize'], weight=config['fontweight_title'])
            plt.xlabel(xlabel, fontsize=config['label_fontsize'], weight=config['fontweight_label'])
            plt.ylabel(ylabel, fontsize=config['label_fontsize'], weight=config['fontweight_label'])
            plt.xticks(fontsize=config['tick_fontsize'], weight=config['fontweight_label'])
            plt.yticks(fontsize=config['tick_fontsize'], weight=config['fontweight_label'])
            plt.grid(True, linestyle='--', alpha=0.6)
            plt.legend(fontsize=config['legend_fontsize'], frameon=False)
            plt.tight_layout()
            pdf.savefig()
            plt.close()

        # Time-domain plots
        styled_plot(
            np.arange(num_samples),
            [I_orig, Q_orig],
            ['Original I', 'Original Q'],
            f"Original Time-Domain Signal ({class_name})",
            'Time', 'Amplitude',
            highlight=(start_trigger, end_trigger)
        )

        styled_plot(
            np.arange(num_samples),
            [I_mod, Q_mod],
            ['Modified I', 'Modified Q'],
            f"Modified Time-Domain Signal ({class_name})",
            'Time', 'Amplitude',
            highlight=(start_trigger, end_trigger)
        )

        # Frequency-domain plots (magnitude and phase)
        styled_plot(freq_axis, [fft_mag_orig], ["Original Magnitude"], "Original FFT Magnitude", "Frequency", "Magnitude")
        styled_plot(freq_axis, [fft_mag_mod], ["Modified Magnitude"], "Modified FFT Magnitude", "Frequency", "Magnitude")
        styled_plot(freq_axis, [fft_phase_orig], ["Original Phase"], "Original FFT Phase", "Frequency", "Phase (radians)")
        styled_plot(freq_axis, [fft_phase_mod], ["Modified Phase"], "Modified FFT Phase", "Frequency", "Phase (radians)")
        
import pickle
def plot_metrics_to_pdf22(pkl_path, output_pdf_path):
    """
    Load metrics from a .pkl file and plot training/validation losses and accuracies.
    Save the plots as a PDF with separate figures for different metrics.
    """
    # Load the metrics from the .pkl file
    with open(pkl_path, 'rb') as f:
        metrics = pickle.load(f)

    pdf = PdfPages(output_pdf_path)

    # Plot Training and Validation Loss (Total)
    plt.figure(figsize=(10, 6))
    plt.plot(metrics['train_loss'], label='Train Loss')
    plt.plot(metrics['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.tight_layout()
    pdf.savefig()
    plt.close()

    # Plot Training and Validation Accuracy
    plt.figure(figsize=(10, 6))
    plt.plot(metrics['train_accuracy'], label='Train Accuracy')
    plt.plot(metrics['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    plt.tight_layout()
    pdf.savefig()
    plt.close()

    # Plot Classification and Reconstruction Loss on Dual-Axis
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Primary Y-axis (left) for Classification Loss
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Classification Loss', color='tab:blue')
    ax1.plot(metrics['train_loss_classification'], label='Train Loss (Classification)', color='tab:blue')
    ax1.plot(metrics['val_loss_classification'], label='Validation Loss (Classification)', linestyle='dashed', color='tab:cyan')
    ax1.tick_params(axis='y', labelcolor='tab:blue')

    # Secondary Y-axis (right) for Reconstruction Loss
    ax2 = ax1.twinx()
    ax2.set_ylabel('Reconstruction L1 Loss', color='tab:red')
    ax2.plot(metrics['train_loss_reconstruction'], label='Train Loss (Reconstruction)', color='tab:red')
    ax2.plot(metrics['train_loss_constrative'], label='Train Loss (Constrative)', color='tab:orange')
    #ax2.plot(metrics['val_loss_reconstruction'], label='Validation Loss (Reconstruction)', linestyle='dashed', color='tab:orange')
    ax2.tick_params(axis='y', labelcolor='tab:red')

    # Title and Legends
    fig.suptitle('Classification vs. Reconstruction and Constrative vLoss')
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')

    plt.tight_layout()
    pdf.savefig()
    plt.close()

    pdf.close()
    print(f"Metrics plots saved to {output_pdf_path}")


# plotting_factory.py
import os
import matplotlib.pyplot as plt

def plot_accuracy_vs_snrs(
    all_metrics: dict,
    monitor_class_names=None,
    plot_config: dict = None,
    filename_prefix: str = "snr_accuracy",
    include_overall: bool = True,
    include_per_class: bool = True,
):
    """
    Plot accuracy vs SNR using a given config and save to disk.

    Parameters
    ----------
    all_metrics : dict
        snr -> {"overall": float, "per_class": {class_name: float}}
        (as produced by your sweep loop)
    monitor_class_names : list[str] or None
        Which class names to show in the per-class chart. If None or empty,
        the per-class plot is skipped.
    plot_config : dict
        Dictionary of style settings (must contain all keys used inside).
    filename_prefix : str
        Base name used for the saved files (without extension).
    include_overall : bool
        If True, makes/saves the overall accuracy vs SNR plot.
    include_per_class : bool
        If True, makes/saves the per-class accuracy vs SNR plot.

    Returns
    -------
    dict : {
      "overall_paths": [str, ...],
      "per_class_paths": [str, ...]
    }
    """
    if plot_config is None:
        raise ValueError("plot_config cannot be None")

    # Create save directory
    os.makedirs(plot_config['save_path'], exist_ok=True)

    # Prepare data series
    snr_values = sorted(all_metrics.keys())
    overall_series = [all_metrics[s]["overall"] for s in snr_values]

    per_class_series = {}
    if include_per_class and monitor_class_names:
        for cls in monitor_class_names:
            per_class_series[cls] = []
        for s in snr_values:
            per_acc = all_metrics[s]["per_class"]
            for cls in monitor_class_names:
                per_class_series[cls].append(per_acc.get(cls, float("nan")))

    saved_overall_paths = []
    saved_per_class_paths = []

    # ----------------------------
    # Plot Overall Accuracy vs SNR
    # ----------------------------
    if include_overall:
        fig, ax = plt.subplots(figsize=plot_config['figure_size'], dpi=plot_config['dpi'])
        ax.plot(snr_values, overall_series, marker="o",
                linewidth=plot_config['line_width'], alpha=plot_config['alpha'])
        ax.set_xlabel("SNR (dB)", fontsize=plot_config['label_fontsize'],
                      fontweight=plot_config['fontweight_label'])
        ax.set_ylabel("Overall Accuracy (%)", fontsize=plot_config['label_fontsize'],
                      fontweight=plot_config['fontweight_label'])
        ax.set_title("Overall Accuracy vs SNR", fontsize=plot_config['title_fontsize'],
                     fontweight=plot_config['fontweight_title'])
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis="both", labelsize=plot_config['tick_fontsize'])

        # Save in all requested formats
        for fmt in plot_config['save_formats']:
            out_path = os.path.join(plot_config['save_path'], f"{filename_prefix}_overall.{fmt}")
            fig.savefig(out_path, bbox_inches="tight", dpi=plot_config['image_format_dpi'])
            saved_overall_paths.append(out_path)

        if plot_config['show']:
            plt.show()
        else:
            plt.close(fig)

    # ----------------------------
    # Plot Per-Class Accuracy vs SNR
    # ----------------------------
    if include_per_class and monitor_class_names:
        fig, ax = plt.subplots(figsize=plot_config['figure_size'], dpi=plot_config['dpi'])
        for cls, series in per_class_series.items():
            ax.plot(snr_values, series, marker="o",
                    linewidth=plot_config['line_width'], alpha=plot_config['alpha'], label=cls)

        ax.set_xlabel("SNR (dB)", fontsize=plot_config['label_fontsize'],
                      fontweight=plot_config['fontweight_label'])
        ax.set_ylabel("Per-Class Accuracy (%)", fontsize=plot_config['label_fontsize'],
                      fontweight=plot_config['fontweight_label'])
        ax.set_title("Per-Class Accuracy vs SNR", fontsize=plot_config['title_fontsize'],
                     fontweight=plot_config['fontweight_title'])

        # Legend styling
        ax.legend(
            title="",
            fontsize=plot_config['legend_fontsize'],
            title_fontsize=plot_config['legend_fontsize'],
            bbox_to_anchor=plot_config['legend_bbox'],
            loc=plot_config['legend_loc'],
            ncol=plot_config['legend_ncol'],
            frameon=False
        )
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis="both", labelsize=plot_config['tick_fontsize'])

        # Save in all requested formats
        for fmt in plot_config['save_formats']:
            out_path = os.path.join(plot_config['save_path'], f"{filename_prefix}_per_class.{fmt}")
            fig.savefig(out_path, bbox_inches="tight", dpi=plot_config['image_format_dpi'])
            saved_per_class_paths.append(out_path)

        if plot_config['show']:
            plt.show()
        else:
            plt.close(fig)

    return {
        "overall_paths": saved_overall_paths,
        "per_class_paths": saved_per_class_paths,
    }
        