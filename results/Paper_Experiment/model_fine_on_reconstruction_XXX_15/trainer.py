import os
import math
import time
import torch
import pickle
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from losses import compute_total_loss
from evaluation import *
from Vit import get_device, get_model_predictions
from plotting_configs import *
from plotting_factory import plot_tsne_embeddings
from utils import *
from augementation_iq import build_transform
class Trainer:
    def __init__(self, model, config_model, args, logger_main, val_loader, train_loader, result_path=None,run_number=None):
        self.args = args
        self.model = model
        self.logger = logger_main
        self.device = get_device()
        self.model.to(self.device)
        self.run_number = run_number if run_number is not None else 0
        self.config_model = config_model
        self.train_loader = train_loader
        self.val_loader = val_loader

        self.project_title = args.project_title
        self.result_path = result_path if result_path else os.path.join(args.result_path, self.project_title)
        self.model_name = "Vit"
        self.epochs = args.max_epochs
        self.batch_size = args.batch_size
        self.lr = config_model['lr']
        self.lr_gamma = config_model['lr_gamma']
        self.step_size = config_model['step_size']
        self.weight_decay = config_model['weight_decay']
        self.w_classification = args.w_classification
        self.w_reconstruction = args.w_reconstruction
        self.w_contrastive = args.w_contrastive
        self.contrastive_temperature = args.contrastive_temperature
        self.patch_size = args.patch_size
        self.criterion = get_criterion(args.loss_function)
        self.optimizer = get_optimizer(config_model['optimizer'], self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=self.step_size, gamma=self.lr_gamma)

        

        self.metrics = {
            'train_loss': [], 'train_accuracy': [],
            'train_loss_classification': [], 'train_loss_reconstruction': [], 'train_loss_constrative': [],
            'val_loss': [], 'val_accuracy': [],
            'val_loss_classification': [], 'val_loss_reconstruction': [], 'val_loss_constrative': [],
            'epoch_times': [],
        }

    def train(self):
        self.logger(f"🚀 Start training {self.model_name}")

        # NEW: init best tracker once
        if not hasattr(self, "best_val_loss"):
            self.best_val_loss = float("inf")

        for epoch in range(self.epochs):
            start_time = time.time()
            self.current_epoch = epoch

            if self.device.type == 'cuda':
                torch.cuda.empty_cache()
                mem = torch.cuda.memory_allocated(self.device) / 1024
                self.logger(f"Epoch {epoch + 1}/{self.epochs}, memory allocated: {mem:.4f} MB")

            self.model.train()
            train_metrics = self.train_epoch()

            train_loss = train_metrics["avg_loss"]
            loss_cls = train_metrics["avg_cls"]
            loss_rec = train_metrics["avg_rec"]
            train_acc = train_metrics["accuracy"]
            loss_con = train_metrics["avg_con"]

            self.logger(
                f"✅ Epoch [{epoch + 1}/{self.epochs}] — Train Acc: {train_acc:.2f}%, "
                f"Loss: {train_loss:.4f}, CLS: {loss_cls:.4f}, REC: {loss_rec:.4f}, CONTR: {loss_con:.4f}"
            )

            self.metrics['train_loss'].append(train_loss)
            self.metrics['train_accuracy'].append(train_acc)
            self.metrics['train_loss_classification'].append(loss_cls)
            self.metrics['train_loss_reconstruction'].append(loss_rec)
            self.metrics['train_loss_constrative'].append(loss_con)

            self.args.print_acc_per_class = (epoch % 10 == 0)

            # 🔁 validate
            val_metrics = evaluate_model(
                model=self.model,
                dataloader=self.val_loader,
                criterion=self.criterion,
                args=self.args,
                device=self.device,
                logger=self.logger
            )

            val_loss = val_metrics["avg_loss"]  # NEW: keep a local for clarity
            self.metrics['val_loss'].append(val_loss)
            self.metrics['val_accuracy'].append(val_metrics["accuracy"])
            self.metrics['val_loss_classification'].append(val_metrics["avg_cls"])
            self.metrics['val_loss_reconstruction'].append(val_metrics["avg_rec"])
            self.metrics['val_loss_constrative'].append(val_metrics["avg_con"])

            # NEW: save when val_loss improves (with safety for NaN/Inf)
            if math.isfinite(val_loss) and (val_loss < self.best_val_loss):
                self.best_val_loss = val_loss
                save_path = os.path.join(self.args.result_path, f"last_good_model{self.run_number}.pth")
                save_model(self.model, save_path)  # uses your existing function
                self.logger(f"💾 Saved new best model @ val_loss={val_loss:.6f} -> {save_path}")
                self.plot_somethings()

            self.scheduler.step()
            epoch_time = time.time() - start_time
            self.metrics['epoch_times'].append(epoch_time)

            self.logger(f"⏱️ Epoch [{epoch + 1}] took {epoch_time:.2f} seconds")
            

        return {
            "metrics": self.metrics,
            "model": self.model
        }


    def train_epoch(self):
        self.model.train()

        total_loss = 0
        total_cls = 0
        total_rec = 0
        total_con = 0
        correct = 0
        samples = 0

        self.logger(f"🔧 Learning rate = {self.lr}")
        if self.args.Aug_prob>0 :
            self.train_loader.dataset.dataset.transform=build_transform(self.args,"full")
           
        '''  if self.current_epoch==40 : 
            self.args.Aug_prob=.3   
            self.args.w_contrastive=.4'''
            
        for data in self.train_loader:
            source = data['iq'].to(self.device)
            source_augment = data['iq_transformed'].to(self.device)
            source_1 = data['iq_rotated'].to(self.device)
            source_2 = data['iq_flipped'].to(self.device)
            targets = data['label'].to(self.device)
            has_labeled = data['has_labeled'].to(self.device)

            batch_outputs = self._run_batch(source,source_augment, source_1, source_2, targets, has_labeled)

            total_loss += batch_outputs["loss"]
            total_cls += batch_outputs["loss_cls"]
            total_rec += batch_outputs["loss_rec"]
            total_con += batch_outputs["loss_con"]
            correct += batch_outputs["correct_preds"]
            samples += batch_outputs["batch_samples"]

        avg_loss = total_loss / len(self.train_loader)
        avg_cls = total_cls / len(self.train_loader)
        avg_rec = total_rec / len(self.train_loader)
        avg_con = total_con / len(self.train_loader)
        acc = (correct / samples) * 100

        return {
            "avg_loss": avg_loss,
            "avg_cls": avg_cls,
            "avg_rec": avg_rec,
            "accuracy": acc,
            "avg_con": avg_con
        }

    def _run_batch(self, source,source_aug, source_1, source_2, targets, has_labeled):
        self.optimizer.zero_grad()

        output = self.model(source_aug)
        output1 = self.model(source_1)
        output2 = self.model(source_2)

        logits = output.get("logits", None)
        x_rec = output.get("x_rec", None)
        x_patch_average = output.get("average_x_patch", None)
        projection1 = output1.get("average_patch_proj", None)
        projection2 = output2.get("average_patch_proj", None)

        weights = {
            'classification': self.args.w_classification,
            'reconstruction': self.args.w_reconstruction,
            'contrastive': self.args.w_contrastive,
            "pseudo_label_weight": self.args.pseudo_label_weight,
            "selftraining_confidence_threshold": self.args.selftraining_confidence_threshold
        }

        loss, loss_cls, loss_rec, loss_con = compute_total_loss(
            logits, x_rec, source,source_aug, has_labeled, targets,
            projection1, projection2,
            self.criterion,
            weights,
            self.contrastive_temperature
        )

        loss.backward()
        self.optimizer.step()

        _, predicted = torch.max(logits, dim=1)
        correct_preds = (predicted == targets).sum().item()
        batch_samples = targets.size(0)

        return {
            "loss": loss.item(),
            "loss_cls": loss_cls.item(),
            "loss_rec": loss_rec.item() if loss_rec is not None else 0.0,
            "loss_con": loss_con.item() if loss_con is not None else 0.0,
            "x_patch_average": x_patch_average,
            "correct_preds": correct_preds,
            "batch_samples": batch_samples,
            "x_rec": x_rec,
        }

    def plot_somethings(self):
        class_names = self.args.current_classes

        if self.current_epoch >60 :
            
            predictions = get_model_predictions(self.model, data_loader=self.val_loader)
          
            all_avg_x_patch = predictions["average_x_patch"]
            all_labels = predictions["labels"]
            selected_classes = ["BPSK", "QPSK", "8PSK", "16APSK", "32APSK","64APSK", "16QAM","32QAM", "FM", "GMSK"]

            plot_config_quick_view_T_sine2["save_path"] = self.result_path
            plot_tsne_embeddings(
                embeddings=all_avg_x_patch,
                labels=all_labels,
                class_names=class_names,
                config=plot_config_quick_view_T_sine2,
                title=f"t-SNE of {self.project_title}",
                visible_classes=selected_classes
            )
