# 📡 Semi-Supervised AMC with Vision Transformers

This repository contains the code, scripts, and resources for our paper on **semi-supervised automatic modulation classification (AMC)** using a **Vision Transformer (ViT)** framework.  
The project integrates **contrastive learning, reconstruction, and supervised classification** to improve performance on wireless signal datasets.

---

## 📁 Project Structure
```
├── Paper_plot/                 # Paper-related plots
├── Quick_plot/                 # Quick visualization scripts
├── pretrained/                 # Pretrained models (optional)
├── results/                    # Training logs, metrics, and errors
├── augmentation_iq.py          # Data augmentation functions
├── batch.sh                    # SLURM batch file for HPC cluster runs
├── configs.yaml                # Main configuration file
├── dataset.py                  # Dataset loader
├── dataset_finger.py           # Dataset with fingerprinting logic
├── evaluation.py               # Evaluation script
├── losses.py                   # Loss functions (classification, contrastive, reconstruction)
├── main_train_grid.py          # Main training script
├── plotting_configs.py         # Plotting utilities
├── plotting_factory.py         # Plotting functions
├── trainer.py                  # Training loop
├── trainer_ooool.py            # Alternate trainer
├── utils.py                    # Utility functions
├── Vit.py                      # Vision Transformer model definition
├── zz_dataset_visualize.ipynb  # Dataset visualization notebook
├── zz_tester.ipynb             # Testing notebook
```

---

## ⚙️ Configuration
All experiment parameters are defined in **`configs.yaml`**.

### Dataset paths
```yaml
dataset_path_local: "D:/datasets/wireless/"
dataset_path_server: "/home/ha168/datasets/wireless/"
```

### Dataset file
```yaml
rml_2018_dataset: "wireless_randomize_snr_0_22_20000.h5"
```
- This is a preprocessed version of the **RML2018 dataset** (~1.12 GB).  
- Due to GitHub file limits, the dataset is not included here.  
- You can download the original dataset from the [RML2018 repository](https://www.kaggle.com/datasets/pinxau1000/radioml2018).  
- *(For access to the preprocessed file, please contact the author by email.)*

### Training setup
- **Labeled percentage** (semi-supervised training):  
  ```yaml
  labeled_percent: 20
  ```
- **Loss weights**:  
  ```yaml
  w_classification: 1
  w_reconstruction: 0
  w_contrastive: 0
  ```
  - Pretraining: set `w_classification: 0` (reconstruction/contrastive only)  
  - Fine-tuning: set `w_classification: 1` and adjust `labeled_percent`

---

## 🚀 Running the Code

### 🔹 Local (CPU/GPU auto-detect)
Run:
```bash
python main_train_grid.py
```
or create a `main.py` to call training functions.

### 🔹 HPC Cluster (SLURM)
Use the provided **`batch.sh`**:
```bash
sbatch batch.sh
```

---

## 🏗️ Training Workflow
1. **Pretraining**
   - Train with reconstruction or contrastive learning (`w_classification=0`).  
   - Label percentage is ignored.

2. **Fine-tuning**
   - Add classification loss (`w_classification=1`).  
   - Choose `labeled_percent` (e.g., 5, 10, 20).  
   - Update `project_title` in config for each experiment.

3. **Evaluation**
   - Performed at the end of training.  
   - Metrics (accuracy, confusion matrices) saved in `results/`.

---

## 📊 Output
- Logs and error files → `results/`  
- Trained models → `pretrained/`  
- Metrics and plots → auto-saved during evaluation  

---

## 📌 Citation
If you use this code in your research, please cite our paper:

```bibtex
@article{ahmadi2025amc,
  title={Enhancing Automatic Modulation Recognition With a Reconstruction-Driven Vision Transformer Under Limited Labels},
  author={Ahmadi, Hossein and Saffari, Banafsheh and Emdadi Mandimahalleh, Sajjad and Safari, Mohammad Esmaeil and Ahmadi, Aria},
  journal={arXiv preprint arXiv:2508.20193},
  year={2025}
}

```

---

## ✉️ Contact
For questions about the code or dataset access, please contact:  
**Hossein Ahmadi** – [hossein_ahmadis@yahoo.com](mailto:hossein_ahmadis@yahoo.com)

---
