"""
PyTorch Lightning wrapper for model training and evaluation
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchmetrics import Accuracy, F1Score, Precision, Recall
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from sklearn.metrics import confusion_matrix

class Lightning_Wrapper(pl.LightningModule):
    def __init__(self, model, num_classes, optimizer=None, learning_rate=1e-3, 
                 scheduler=None, criterion=None, log_dir=None, label_names=None, freeze_nfp=False, unfreeze_epoch=5):
        """
        Lightning wrapper for model training and evaluation
        Args:
            model: PyTorch model to wrap
            num_classes: Number of classes
            optimizer: Optimizer to use (default: Adam)
            learning_rate: Learning rate (default: 1e-3)
            scheduler: Learning rate scheduler (optional)
            criterion: Loss function (default: CrossEntropyLoss)
            log_dir: Directory to save logs and outputs
            label_names: Names of the classes (for confusion matrix)
        """
        super().__init__()
        self.model = model
        self.num_classes = num_classes 
        self.learning_rate = learning_rate
        self.scheduler = scheduler
        self.criterion = criterion if criterion is not None else nn.CrossEntropyLoss(label_smoothing=0.05)
        self.log_dir = log_dir
        self.label_names = label_names
        self.freeze_nfp = freeze_nfp
        self.unfreeze_epoch = unfreeze_epoch
        # Initialize metrics
        self.train_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_acc   = Accuracy(task="multiclass", num_classes=num_classes)
        self.test_acc  = Accuracy(task="multiclass", num_classes=num_classes)

        self.train_f1 = F1Score(task="multiclass", num_classes=num_classes, average="macro")
        self.val_f1   = F1Score(task="multiclass", num_classes=num_classes, average="macro")
        self.test_f1  = F1Score(task="multiclass", num_classes=num_classes, average="macro")

        self.train_precision = Precision(task="multiclass", num_classes=num_classes, average="macro")
        self.val_precision   = Precision(task="multiclass", num_classes=num_classes, average="macro")
        self.test_precision  = Precision(task="multiclass", num_classes=num_classes, average="macro")

        self.train_recall = Recall(task="multiclass", num_classes=num_classes, average="macro")
        self.val_recall   = Recall(task="multiclass", num_classes=num_classes, average="macro")
        self.test_recall  = Recall(task="multiclass", num_classes=num_classes, average="macro")


        # Store predictions for confusion matrix
        self.test_predictions = []
        self.test_targets = []

    def forward(self, batch):
        # If batch is a dict (as with torchgeo), extract the image tensor
        if isinstance(batch, dict):
            batch = batch['image']
        return self.model(batch)


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        if self.scheduler is not None:
            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': self.scheduler(optimizer),
                    'monitor': 'val_loss'
                }
            }
        return optimizer

    def training_step(self, batch, batch_idx):
        # Debug: print batch type and keys
        print('TRAINING STEP BATCH TYPE:', type(batch))
        if isinstance(batch, dict):
            print('TRAINING STEP BATCH KEYS:', batch.keys())
            x = batch['image']
            y = batch['label']
        else:
            x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        
        # Log metrics
        self.train_acc(y_hat.softmax(dim=-1), y)
        self.train_f1(y_hat.softmax(dim=-1), y)
        self.train_precision(y_hat.softmax(dim=-1), y)
        self.train_recall(y_hat.softmax(dim=-1), y)
        
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_acc', self.train_acc, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_f1', self.train_f1, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_precision', self.train_precision, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_recall', self.train_recall, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        
        return loss

    def validation_step(self, batch, batch_idx):
        # Debug: print batch type and keys
        print('VALIDATION STEP BATCH TYPE:', type(batch))
        if isinstance(batch, dict):
            print('VALIDATION STEP BATCH KEYS:', batch.keys())
            x = batch['image']
            y = batch['label']
        else:
            x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        
        # Log metrics
        self.val_acc(y_hat.softmax(dim=-1), y)
        self.val_f1(y_hat.softmax(dim=-1), y)
        self.val_precision(y_hat.softmax(dim=-1), y)
        self.val_recall(y_hat.softmax(dim=-1), y)
        
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_acc', self.val_acc, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_f1', self.val_f1, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_precision', self.val_precision, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_recall', self.val_recall, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        
        return loss

    def test_step(self, batch, batch_idx):
        # Debug: print batch type and keys
        print('TEST STEP BATCH TYPE:', type(batch))
        if isinstance(batch, dict):
            print('TEST STEP BATCH KEYS:', batch.keys())
            x = batch['image']
            y = batch['label']
        else:
            x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        
        # Store predictions for confusion matrix
        preds = torch.argmax(y_hat, dim=1)
        self.test_predictions.extend(preds.cpu().numpy())
        self.test_targets.extend(y.cpu().numpy())
        
        # Log metrics
        self.test_acc(y_hat.softmax(dim=-1), y)
        self.test_f1(y_hat.softmax(dim=-1), y)
        self.test_precision(y_hat.softmax(dim=-1), y)
        self.test_recall(y_hat.softmax(dim=-1), y)
        
        self.log('test_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('test_acc', self.test_acc, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('test_f1', self.test_f1, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('test_precision', self.test_precision, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('test_recall', self.test_recall, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        
        return loss
    
    def on_test_epoch_end(self):
        if self.log_dir is not None:
            cm = confusion_matrix(self.test_targets, self.test_predictions)

            # Save confusion matrix as image
            plt.figure(figsize=(10, 8))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')

            if self.label_names is not None:
                plt.xticks(np.arange(len(self.label_names)) + 0.5, self.label_names, rotation=45)
                plt.yticks(np.arange(len(self.label_names)) + 0.5, self.label_names, rotation=0)

            plt.xlabel('Predicted')
            plt.ylabel('True')
            plt.title('Confusion Matrix')

            os.makedirs(os.path.join(self.log_dir, 'confusion_matrices'), exist_ok=True)
            plt.savefig(os.path.join(self.log_dir, 'confusion_matrices', 'confusion_matrix.png'))
            plt.close()

            self.test_confusion_matrix = cm

            # Clear prediction buffers
            self.test_predictions = []
            self.test_targets = []

    # ----------------------------------------------------------------------
    # 🔧 Freeze NFP Head at epoch 10
    # ----------------------------------------------------------------------
    def on_train_epoch_start(self):
        if self.freeze_nfp and self.current_epoch >= self.unfreeze_epoch:
            print(f"[INFO] Unfreezing NFP Head at epoch {self.current_epoch}")
            for name, param in self.model.named_parameters():
                if 'nfp_head' in name or 'se_gate' in name:
                    param.requires_grad = True
            self.freeze_nfp = False
        elif self.freeze_nfp:
            for name, param in self.model.named_parameters():
                if 'nfp_head' in name or 'se_gate' in name:
                    param.requires_grad = False

    def get_progress_bar_dict(self):
        # Don't show the version number
        items = super().get_progress_bar_dict()
        items.pop("v_num", None)
        return items

