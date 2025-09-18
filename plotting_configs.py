
plot_config_quick_view = {
    'figure_size': (7, 3),
    'line_width': 1.5,
    'scatter_size': 30,
    'scatter_trigger_size': 40,
    'title_fontsize': 14,
    'label_fontsize': 16,
    'legend_fontsize': 12,
    'tick_fontsize': 12,
    'dpi': 100,
    'alpha': 0.9,
    'save_path': 'Quick_plot',
    'sig_length': 200,
    'plot_as_scatter': False,
    'scatter_iq_size': 30,
    'scatter_iq_trigger_size': 40,
    'signal_color': 'blue',
    'trigger_color': 'red',
    'fontweight_title': 'normal',
    'fontweight_label': 'normal',
    'legend_bbox': (1, 1),
    'legend_loc': 'upper right',
    'legend_ncol': 1,
    'show': True,
    'save_formats': ['pdf', 'png'],       # 👈 new field
    'image_format_dpi': 300               # 👈 PNG resolution
}
plot_config_quick_view_T_sine2 = {
    'figure_size': (20 / 2.54, 14 / 2.54),  # ~11 cm × 7.5 cm from paper quality config
    'line_width': 1.5,                     # from quick view
    'scatter_size': 30,                    # from quick view
    'scatter_trigger_size': 40,            # from quick view
    'title_fontsize': 20,                  # from paper quality
    'label_fontsize': 25,                  # from paper quality
    'legend_fontsize': 20,                 # from paper quality
    'tick_fontsize': 20,                   # from paper quality
    'dpi': 100,                            # from quick view
    'alpha': 0.9,                          # from quick view
    'save_path': 'Quick_plot',             # from quick view
    'sig_length': 200,                     # from quick view
    'plot_as_scatter': False,              # from quick view
    'scatter_iq_size': 30,                 # from quick view
    'scatter_iq_trigger_size': 40,         # from quick view
    'signal_color': 'blue',                # from quick view
    'trigger_color': 'red',                # from quick view
    'fontweight_title': 'bold',            # upgraded to bold for better visibility
    'fontweight_label': 'bold',            # upgraded to bold
    'legend_bbox': (1, 1),                 # from quick view
    'legend_loc': 'upper right',           # from quick view
    'legend_ncol': 1,                      # from quick view
    'show': False,                         # from quick view
    'save_formats': ['pdf'],              # from quick view
    'image_format_dpi': 300  ,
    
# from quick view
}
plot_config__paper_metrice = {
    'figure_size': (20 / 2.54, 14 / 2.54),  # ~11 cm × 7.5 cm from paper quality config
    'line_width': 1.5,                     # from quick view
    'scatter_size': 30,                    # from quick view
    'scatter_trigger_size': 40,            # from quick view
    'title_fontsize': 20,                  # from paper quality
    'label_fontsize': 25,                  # from paper quality
    'legend_fontsize': 20,                 # from paper quality
    'tick_fontsize': 20,                   # from paper quality
    'dpi': 100,                            # from quick view
    'alpha': 0.9,                          # from quick view
    'save_path': 'Quick_plot',             # from quick view
    'sig_length': 200,                     # from quick view
    'plot_as_scatter': False,              # from quick view
    'scatter_iq_size': 30,                 # from quick view
    'scatter_iq_trigger_size': 40,         # from quick view
    'signal_color': 'blue',                # from quick view
    'trigger_color': 'red',                # from quick view
    'fontweight_title': 'bold',            # upgraded to bold for better visibility
    'fontweight_label': 'bold',            # upgraded to bold
    'legend_bbox': (1, 1),                 # from quick view
    'legend_loc': 'upper right',           # from quick view
    'legend_ncol': 1,                      # from quick view
    'show': False,                         # from quick view
    'save_formats': ['pdf'],              # from quick view
    'image_format_dpi': 300  ,
    'nan_handling': 'ffill'  # options: 'ffill', 'zero', or 'ignore'
# from quick view
}

plot_config_quick_view_T_sine = {
    'figure_size': (12, 7),
    'line_width': 1.5,
    'scatter_size': 30,
    'scatter_trigger_size': 40,
    'title_fontsize': 14,
    'label_fontsize': 16,
    'legend_fontsize': 12,
    'tick_fontsize': 12,
    'dpi': 100,
    'alpha': 0.9,
    'save_path': 'Quick_plot',
    'sig_length': 200,
    'plot_as_scatter': False,
    'scatter_iq_size': 30,
    'scatter_iq_trigger_size': 40,
    'signal_color': 'blue',
    'trigger_color': 'red',
    'fontweight_title': 'normal',
    'fontweight_label': 'normal',
    'legend_bbox': (1, 1),
    'legend_loc': 'upper right',
    'legend_ncol': 1,
    'show': False,
    'save_formats': ['pdf' ],       # 👈 new field
    'image_format_dpi': 300               # 👈 PNG resolution
}
plot_config_paper_quality_time = {
    'figure_size': (20 / 2.54, 14 / 2.54),  # ~11 cm × 7.5 cm
    'line_width': 1.7,
    'scatter_size': 50,
    'scatter_trigger_size': 70,
    'title_fontsize': 20,
    'label_fontsize': 25,
    'legend_fontsize': 20,
    'tick_fontsize': 20,
    'dpi': 400,
    'alpha': 0.95,
    'save_path': 'Paper_plot',
    'sig_length': 200,
    'plot_as_scatter': False,
    'scatter_iq_size': 50,
    'scatter_iq_trigger_size': 70,
    'signal_color': 'blue',
    'trigger_color': 'black',
    'fontweight_title': 'bold',
    'fontweight_label': 'bold',
    'legend_bbox': (0.5, 1.05),
    'legend_loc': 'upper center',
    'legend_ncol': 2,
    'show': True,
    'save_formats': ['pdf'],       # 👈 new field
    'image_format_dpi': 400               # 👈 PNG resolution
}

plot_config_paper_quality_spectrogram = {
    'figure_size': (20 / 2.54, 14 / 2.54),  # Match others for consistency
    'dpi': 400,
    'title_fontsize': 20,
    'label_fontsize': 25,
    'tick_fontsize': 20,
    'fontweight_title': 'bold',
    'fontweight_label': 'bold',
    'save_path': 'Paper_plot',
    'sig_length': 200,
    'nfft': 256,               # STFT window size
    'noverlap': 128,           # 50% overlap
    'cmap': 'inferno',         # Grayscale print-friendly
    'show': True,
    'save_formats': ['pdf', 'png'],       # ✅ Added format control
    'image_format_dpi': 400               # ✅ PNG resolution
}
plot_config_paper_quality_rss = {
    'figure_size': (20 / 2.54, 14 / 2.54),
    'line_width': 1.7,
    'title_fontsize': 20,
    'label_fontsize': 25,
    'tick_fontsize': 20,
    'dpi': 400,
    'alpha': 0.95,
    'signal_color': 'green',  # Distinct from I/Q and spectrogram
    'fontweight_title': 'bold',
    'fontweight_label': 'bold',
    'save_path': 'Paper_plot',
    'sig_length': 200,
    'save_formats': ['pdf', 'png'],       # ✅ Added format control
    'image_format_dpi': 400,              # ✅ PNG resolution
    'show': True
}


plot_config_paper_quality_iq = {
    'figure_size': (20 / 2.54, 14 / 2.54),  # Same as main signal plot
    'line_width': 1.7,
    'scatter_size': 50,
    'scatter_trigger_size': 70,
    'title_fontsize': 20,
    'label_fontsize': 25,
    'legend_fontsize': 20,
    'tick_fontsize': 20,
    'dpi': 400,
    'alpha': 0.95,
    'save_path': 'Paper_plot',
    'sig_length': 250,
    'plot_as_scatter': False,
    'scatter_iq_size': 50,
    'scatter_iq_trigger_size': 70,
    'signal_color': 'blue',
    'trigger_color': 'blue',
    'fontweight_title': 'bold',
    'fontweight_label': 'bold',
    'legend_bbox': (0.5, 1.05),
    'legend_loc': 'upper center',
    'legend_ncol': 3,
    'show': True,
    'save_formats': ['pdf', 'png'],     # ✅ Save as both formats
    'image_format_dpi': 400             # ✅ High-res PNG output
}







































# This file contains the plotting configurations for different types of plots.----------------------------------
plot_time_config_quick_view11= {
    # --- Figure Settings ---
    'figure_size': (8, 4),
    'dpi': 100,
    'save_path': None,

    # --- Font and Label Settings ---
    'title_fontsize': 12,
    'label_fontsize': 10,
    'legend_fontsize': 8,
    'tick_fontsize': 10,
    'fontweight_title': 'bold',
    'fontweight_label': 'bold',

    # --- Plot Appearance ---
    'line_width': 1,
    'alpha': 0.6,
    'plot_as_scatter': False,
    'signal_color': 'blue',

    # --- Signal Settings ---
    'sig_length': 500,
    'scatter_size': 25,
    'scatter_iq_size': 5,
}


plot_time_config_paper_quality11 = {
    # --- Figure Settings ---
    'figure_size': (160, 110),
    'dpi': 300,
    'save_path': 'Paper_plot',

    # --- Font and Label Settings ---
    'title_fontsize': 550,
    'label_fontsize': 550,
    'legend_fontsize': 450,
    'tick_fontsize': 400,
    'fontweight_title': 'bold',
    'fontweight_label': 'bold',

    # --- Plot Appearance ---
    'line_width': 5,
    'alpha': 0.9,
    'plot_as_scatter': False,
    'signal_color': 'blue',

    # --- Signal Settings ---
    'sig_length': 250,
    'scatter_size': 200,
    'scatter_iq_size': 15000,
}
