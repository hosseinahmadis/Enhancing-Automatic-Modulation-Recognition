import os
import h5py
import numpy as np
import torch
import torch.utils.data as data
from utils import get_device
from augementation_iq import *

device = get_device()

class RML2018_Dataset(data.Dataset):
    # Class variables to store split sizes once calculated
    total_samples_full = 0
    split_sizes = {'train': 0, 'val': 0, 'test': 0}

    def __init__(self, args=None, logger=None):
        self.args = args
        self.logger = logger
        self.transform = None

        # Dataset path
        root_path = self.args.dataset_path_server if device.type == 'cuda' else self.args.dataset_path_local
        data_file = self.args.rml_2018_dataset
        self.root_path = os.path.join(root_path, data_file)

        # Load data
        assert os.path.exists(self.root_path), f"Dataset not found at {self.root_path}"
        self.data = h5py.File(self.root_path, 'r')
        self.original_size = self.data['iq'].shape
        data_length = min(self.args.data_length, self.original_size[0])

        # Save full unfiltered version
        self.full_iq = self.data['iq'][:data_length]
        self.full_labels = self.data['label'][:data_length]
        self.full_snr = np.squeeze(self.data['snr'][:data_length])
        self.iq_min = self.data['iq_min'][:]
        self.iq_max = self.data['iq_max'][:]
        self.snr_min = self.data['snr_min'][()]
        self.snr_max = self.data['snr_max'][()]
        RML2018_Dataset.total_samples_full = data_length

        # Filter and split once
        self.shuffle_and_restart()
        self.generate_splits()

        self.input_size = (self.args.in_size[-2], self.args.in_size[-1])
        self.labels_name = self.args.total_class
        self.labels_index = self.args.train_class_indices
        self.uniq_class = np.unique(self.labels)

        self.log(f"Dataset loaded with {len(self.iq)} total filtered samples")
    def generate_splits(self):
        total_len = self.iq.shape[0]
        indices = np.arange(total_len)
        if self.args.shuffle_before_split:
            np.random.shuffle(indices)

        train_end = int(0.7 * total_len)
        val_end = train_end + int(0.1 * total_len)
        test_end = total_len

        RML2018_Dataset.split_sizes = {
            'train': train_end,
            'val': val_end - train_end,
            'test': test_end - val_end
        }

        # Save split indices
        self.train_indices = indices[:train_end]
        self.val_indices = indices[train_end:val_end]
        self.test_indices = indices[val_end:]

        # Create PyTorch subsets
        self.train_set = torch.utils.data.Subset(self, self.train_indices)
        self.val_set = torch.utils.data.Subset(self, self.val_indices)
        self.test_set = torch.utils.data.Subset(self, self.test_indices)

        self.log(f"Splits created: Train={train_end}, Val={val_end-train_end}, Test={test_end-val_end}")


    def shuffle_and_restart(self):
        self.apply_snr_filter_to_full()  # Reset filtered data

        total_len = self.iq.shape[0]
        indices = np.arange(total_len)

        if self.args.shuffle_before_split:
            np.random.shuffle(indices)
       
        train_end = int(0.7 * total_len)
        val_end = train_end + int(0.1 * total_len)
        RML2018_Dataset.split_sizes = {
            'train': train_end,
            'val': val_end - train_end,
            'test': total_len - val_end
        }

        # Save split indices
        self.train_indices = indices[:train_end]
        self.val_indices = indices[train_end:val_end]
        self.test_indices = indices[val_end:]

        # Wrap in PyTorch subsets
        self.train_set = torch.utils.data.Subset(self, self.train_indices)
        self.val_set = torch.utils.data.Subset(self, self.val_indices)
        self.test_set = torch.utils.data.Subset(self, self.test_indices)

        # Recreate labeled mask only if shuffle is enabled
        
        labeled_percent = getattr(self.args, "labeled_percent", 100)
        total_samples = len(self.labels)
        num_labeled = int((labeled_percent / 100.0) * total_samples)
        self.labeled_samples = np.zeros(total_samples, dtype=bool)
        if num_labeled > 0:
            labeled_indices = np.random.choice(total_samples, num_labeled, replace=False)
            self.labeled_samples[labeled_indices] = True
        self.log(f"[Shuffle & Restart] Splits set: train={train_end}, val={val_end - train_end}, test={total_len - val_end}")
    
        
                

    def filter_with_snr(self, snr_range_min, snr_range_max):
        mask = (self.snr >= snr_range_min) & (self.snr < snr_range_max)
        self.iq = self.iq[mask]
        self.labels = self.labels[mask]
        self.snr = self.snr[mask]
    def apply_snr_filter_to_full(self):
        snr_min, snr_max = self.args.snr_range
        mask = (self.full_snr >= snr_min) & (self.full_snr < snr_max)
        self.iq = self.full_iq[mask]
        self.labels = self.full_labels[mask]
        self.snr = self.full_snr[mask]

    def normalize_signal(self, iq):
        iq_min = self.iq_min[0]
        iq_max = self.iq_max[0]
        normalized_iq = (iq - iq_min) / (iq_max - iq_min)
        scaled_iq = normalized_iq * 2 - 1  # Scale to [-1, 1]
        return scaled_iq

    def __len__(self):
        return self.iq.shape[0]

    def __getitem__(self, index):
        y = int(self.labels[index])
        x = self.iq[index].transpose()         # [2, 512]
        x = np.expand_dims(x, axis=0)          # [1, 2, 512]
        x = x[:, :, 0:self.input_size[1]]
        x = self.normalize_signal(x)

        x_tensor = torch.tensor(x, dtype=torch.float32)
        x_base = x_tensor.clone()

        # Augmentations
        x_rotated = rotate_signal(x_base.clone(), random.choice([90, 270]))
        x_flipped  = add_gaussian_noise(x_base.clone())
        x_transformed = self.transform(x_base.clone()) if self.transform else x_base.clone()
       
        sample = {
            "iq": x_base,
            "iq_rotated": self.transform(x_base.clone()) if self.transform else x_base.clone(),
            "iq_flipped": self.transform(x_base.clone()) if self.transform else x_base.clone(),
            "iq_transformed": x_transformed,
            "label": y,
            "snr": self.snr[index],
            "status": 0,
            "has_labeled": self.labeled_samples[index],
        }
        return sample

    def summarize_dataset(self):
        output = []
        output.append(f"📦 Original Dataset Size: {self.original_size}")
        output.append(f"📦 Used Data Length (from config): {self.args.data_length}")
        output.append(f"🧪 Split Summary:")
        output.append(f"   - Train Samples: {RML2018_Dataset.split_sizes['train']}")
        output.append(f"   - Val Samples:   {RML2018_Dataset.split_sizes['val']}")
        output.append(f"   - Test Samples:  {RML2018_Dataset.split_sizes['test']}")

        # --- NEW: labeled vs unlabeled count per split ---
        def count_labeled(indices):
            labeled_mask = self.labeled_samples[indices]
            num_labeled = np.sum(labeled_mask)
            num_unlabeled = len(indices) - num_labeled
            return num_labeled, num_unlabeled

        train_labeled, train_unlabeled = count_labeled(self.train_indices)
        val_labeled, val_unlabeled = count_labeled(self.val_indices)
        test_labeled, test_unlabeled = count_labeled(self.test_indices)

        output.append("📝 Labeled vs Unlabeled per split:")
        output.append(f"   - Train: {train_labeled} labeled | {train_unlabeled} unlabeled")
        output.append(f"   - Val:   {val_labeled} labeled | {val_unlabeled} unlabeled")
        output.append(f"   - Test:  {test_labeled} labeled | {test_unlabeled} unlabeled")

        output.append(f"-----------------------------")
        output.append(f"IQ min: I = {self.iq_min[0]:.4f}, Q = {self.iq_min[1]:.4f}")
        output.append(f"IQ max: I = {self.iq_max[0]:.4f}, Q = {self.iq_max[1]:.4f}")
        output.append(f"SNR range: {self.snr_min} to {self.snr_max}")
        output.append(f"\n📊 Class distribution:")
        unique_labels, label_counts = np.unique(self.labels, return_counts=True)
        class_names = self.args.current_classes
        for label, count in zip(unique_labels, label_counts):
            output.append(f"  {class_names[label]}: {count} samples")

        output.append(f"\n📶 SNR distribution:")
        unique_snr, snr_counts = np.unique(self.snr, return_counts=True)
        for snr, count in zip(unique_snr, snr_counts):
            output.append(f"  SNR {snr}: {count} samples")

        summary = "\n".join(output)
        self.log(summary)
        return summary

    def log(self, msg):
        if self.logger:
            self.logger(msg)
        else:
            print(msg)
