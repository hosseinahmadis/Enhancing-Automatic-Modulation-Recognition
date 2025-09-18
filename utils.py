import os
import yaml
import torch
import logging
import numpy as np
import pandas as pd
import random
import importlib
import inspect
import matplotlib.pyplot as plt
import torch.nn
from collections import Counter, defaultdict, deque
import time
import datetime
import pickle
from sklearn.manifold import TSNE
import torch
import yaml
from types import SimpleNamespace
import pickle
import matplotlib.pyplot as plt
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import torch.distributed as dist
from matplotlib.backends.backend_pdf import PdfPages
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import shutil
import logging

# Set up logger if not already configured
logger = logging.getLogger(__name__)
if not logger.hasHandlers():
    logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
#-------------------------------------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
#------------------------------------------------------------------------------------------------------
def plot_original_and_modified_iq_data(signals_original, signals_modified,old_targets,  save_path,start_trigger = 0, end_trigger = 512):
    """
    Plots and saves both time-domain and selected frequency-domain representations (magnitude and phase)
    of the first IQ data sample for both the original and modified signals, highlighting the selected region.

    Parameters:
    signals_original (numpy.ndarray): A batch of original input signals with shape (batch_size, 1, 2, 512).
    signals_modified (numpy.ndarray): A batch of modified input signals with shape (batch_size, 1, 2, 512).
    location (int): The starting index for frequency selection, used as location * 16.
    save_path (str): Path to save the PDF file.
    """
    config_file = 'configs.yaml'
    config = get_config(config_file)

    # Ensure tensors are on CPU before using NumPy
    signals_original = signals_original.detach().cpu().numpy()
    signals_modified = signals_modified.detach().cpu().numpy()
    # Extract first signal from both original and modified batches
    signal_orig = signals_original[0, 0]  # Extract first original signal with shape (2, 512)
    signal_mod = signals_modified[0, 0]  # Extract first modified signal with shape (2, 512)

    I_orig, Q_orig = signal_orig[0], signal_orig[1]  # Original I and Q components
    I_mod, Q_mod = signal_mod[0], signal_mod[1]  # Modified I and Q components
    
    signal_names=config["current_classes"]
    class_name=signal_names[old_targets[0]]
    num_samples = signals_original.shape[-1]

    # Compute FFT for frequency domain representation
    I_orig_fft = np.fft.fft(I_orig)
    Q_orig_fft = np.fft.fft(Q_orig)
    magnitude_spectrum_orig = np.abs(I_orig_fft + 1j * Q_orig_fft)  # Original magnitude
    phase_spectrum_orig = np.angle(I_orig_fft + 1j * Q_orig_fft)  # Original phase

    I_mod_fft = np.fft.fft(I_mod)
    Q_mod_fft = np.fft.fft(Q_mod)
    magnitude_spectrum_mod = np.abs(I_mod_fft + 1j * Q_mod_fft)  # Modified magnitude
    phase_spectrum_mod = np.angle(I_mod_fft + 1j * Q_mod_fft)  # Modified phase

    # Define selected frequency range
    start_idx = start_trigger
    end_idx = end_trigger
    freq_axis = np.fft.fftfreq(num_samples)[start_idx:end_idx]  # Extracting only the required range

    with PdfPages(save_path) as pdf:
        # Plot original time-domain signal with markers
        plt.figure(figsize=(10, 5))
        plt.plot(I_orig, label="Original In-phase (I)")
        plt.plot(Q_orig, label="Original Quadrature (Q)")
        plt.axvspan(start_idx, end_idx, color='yellow', alpha=0.3, label="Selected Region")  # Highlight region
        plt.xlabel("Samples")
        plt.ylabel("Amplitude")
        plt.title(f"Original Time-Domain Signal with Selected Region     {class_name}")
        plt.legend()
        plt.grid()
        pdf.savefig()
        plt.close()

        # Plot modified time-domain signal with markers
        plt.figure(figsize=(10, 5))
        plt.plot(I_mod, label="Modified In-phase (I)")
        plt.plot(Q_mod, label="Modified Quadrature (Q)")
        plt.axvspan(start_idx, end_idx, color='yellow', alpha=0.3, label="Selected Region")  # Highlight region
        plt.xlabel("Samples")
        plt.ylabel("Amplitude")
        plt.title(f"Modified Time-Domain Signal with Selected Region {class_name}")
        plt.legend()
        plt.grid()
        pdf.savefig()
        plt.close()
        location=22
        # Plot selected portion of original frequency-domain magnitude spectrum
        plt.figure(figsize=(10, 5))
        plt.plot(freq_axis, magnitude_spectrum_orig[start_idx:end_idx], marker='o', linestyle='-', label="Original Magnitude Spectrum")
        plt.xlabel("Frequency (Normalized)")
        plt.ylabel("Magnitude")
        plt.title(f"Original Frequency-Domain Magnitude (Location {location})")
        plt.legend()
        plt.grid()
        pdf.savefig()
        plt.close()

        # Plot selected portion of modified frequency-domain magnitude spectrum
        plt.figure(figsize=(10, 5))
        plt.plot(freq_axis, magnitude_spectrum_mod[start_idx:end_idx], marker='o', linestyle='-', label="Modified Magnitude Spectrum")
        plt.xlabel("Frequency (Normalized)")
        plt.ylabel("Magnitude")
        plt.title(f"Modified Frequency-Domain Magnitude (Location {location})")
        plt.legend()
        plt.grid()
        pdf.savefig()
        plt.close()

        # Plot selected portion of original frequency-domain phase spectrum
        plt.figure(figsize=(10, 5))
        plt.plot(freq_axis, phase_spectrum_orig[start_idx:end_idx], marker='o', linestyle='-', label="Original Phase Spectrum")
        plt.xlabel("Frequency (Normalized)")
        plt.ylabel("Phase (Radians)")
        plt.title(f"Original Frequency-Domain Phase (Location {location})")
        plt.legend()
        plt.grid()
        pdf.savefig()
        plt.close()

        # Plot selected portion of modified frequency-domain phase spectrum
        plt.figure(figsize=(10, 5))
        plt.plot(freq_axis, phase_spectrum_mod[start_idx:end_idx], marker='o', linestyle='-', label="Modified Phase Spectrum")
        plt.xlabel("Frequency (Normalized)")
        plt.ylabel("Phase (Radians)")
        plt.title(f"Modified Frequency-Domain Phase (Location {location})")
        plt.legend()
        plt.grid()
        pdf.savefig()
        plt.close()



def get_config(config_path="configs.yaml"):
    with open(config_path, 'r') as stream:
        config_dict = yaml.load(stream, Loader=yaml.FullLoader)
        
    return config_dict


def copy_files_to_path(file_list, destination_path):
    
    os.makedirs(destination_path, exist_ok=True)
    copied_paths = []

    for file_path in file_list:
        if os.path.exists(file_path):
            filename = os.path.basename(file_path)
            dst_path = os.path.join(destination_path, filename)
            shutil.copy2(file_path, dst_path)
            copied_paths.append(dst_path)
        else:
            print(f"⚠️ File not found: {file_path}")

    return copied_paths

def get_config_as_args(config_path="configs.yaml"):
    with open(config_path, 'r') as stream:
        config_dict = yaml.load(stream, Loader=yaml.FullLoader)
    
    # Convert to class-like object (SimpleNamespace allows dot-access)
    args = SimpleNamespace(**config_dict)
    
    return  args

def save_args_to_config(args, output_path="configs.yaml"):
    # Convert SimpleNamespace or other dot-access object to dictionary
    if isinstance(args, SimpleNamespace):
        args_dict = vars(args)
    else:
        args_dict = args.__dict__
    
    with open(output_path, 'w') as f:
        yaml.dump(args_dict, f, default_flow_style=False)

def get_device():
    config=get_config('configs.yaml')
    use_gpu = torch.cuda.is_available() if config['use_gpu'] == 'auto' else config['use_gpu']
    device = torch.device("cuda" if use_gpu else "cpu")
    return device    
#-----------------------------------------------------------------    
 
#------------------------------------------------------------    
def load_checkpoint(model,checkpoint_path,device):
               
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint)
        model=model.to(device)        
        print(f"Loaded checkpoint from epoch at {checkpoint_path}.")
        return model   

#-------------------------------------------------------------------------

def save_model(model, filename):
    torch.save(model.state_dict(), filename)
def load_model(model, model_path, device):
    """
    Load the model weights from a .pth file to the specified device.
    """
    # Load the model state dictionary
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()  # Set to evaluation mode

    print(f"Model loaded from {model_path} to {device}")
    return model
def prepare_dataloader(dataset: Dataset,config,batch_size=None ):
    
    return DataLoader(
            dataset,
            batch_size=config['batch_size'] if batch_size is None else batch_size,
            shuffle=True,
            num_workers=0
           )
    


#--------------------------------------------------------------------------------------

import torch.nn as nn

def get_criterion(loss_name):
    loss_name = loss_name.lower()
    if loss_name == "crossentropy":
        return nn.CrossEntropyLoss()
    elif loss_name == "mse":
        return nn.MSELoss()
    elif loss_name == "bce":
        return nn.BCELoss()
    else:
        raise ValueError(f"Unsupported loss function: {loss_name}")
import torch.optim as optim

def get_optimizer(optimizer_name, model_params, lr, weight_decay=0.0):
    optimizer_name = optimizer_name.lower()
    if optimizer_name == "adam":
        return optim.Adam(model_params, lr=lr, weight_decay=weight_decay)
    elif optimizer_name == "sgd":
        return optim.SGD(model_params, lr=lr, weight_decay=weight_decay)
    elif optimizer_name == "adamw":
        return optim.AdamW(model_params, lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")
    
    
class LogToFile:
    def __init__(self, filename='logs/output.txt', enable=False,log_to_screen=False):
        self.filename = filename
        self.enable = enable
        self.log_to_screen= log_to_screen
        self._ensure_log_directory()

    def __call__(self, text):
        if self.enable:
            self.write_to_file(text)
        if self.log_to_screen:     
            logger.info(text)
            #print(text)

    def write_to_file(self, text):
        with open(self.filename, 'a') as ff:
            ff.write(text + '\n')

    def clear_log_file(self):
        if self.enable:
            with open(self.filename, 'w') as ff:
                ff.write('')

    def _ensure_log_directory(self):
        directory = os.path.dirname(self.filename)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)




#--------------------------------------------------------------------------------
#--------------------------------------------------------------------------------
def get_model_predictions(model, data_loader):
    model.eval()
    device = get_device()

    outputs_collector = {}  # key -> list of tensors
    all_labels = []

    with torch.no_grad():
        for batch in data_loader:
            inputs = batch['iq'].to(device)
            labels = batch['label'].to(device)
            model_outputs = model(inputs)  # Should return a dict

            for key, value in model_outputs.items():
                if value is not None:
                    if key not in outputs_collector:
                        outputs_collector[key] = []
                    outputs_collector[key].append(value.cpu())

            all_labels.append(labels.cpu())

    # Concatenate outputs
    results = {}
    for key, tensor_list in outputs_collector.items():
        try:
            results[key] = torch.cat(tensor_list, dim=0)
        except Exception:
            results[key] = tensor_list  # fallback if tensor can't be concatenated

    results["labels"] = torch.cat(all_labels, dim=0)
    return results
