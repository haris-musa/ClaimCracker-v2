"""Training pipeline for news classification model."""

from typing import Dict, Any, Optional, List
import torch
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from pathlib import Path
import numpy as np
from tqdm.auto import tqdm
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from ..models.classifier import NewsClassifier

class NewsDataset(Dataset):
    """PyTorch dataset for news classification."""
    
    def __init__(
        self,
        texts: List[str],
        labels: Optional[List[int]] = None,
        model: Optional[NewsClassifier] = None
    ):
        """Initialize dataset.
        
        Args:
            texts: List of input texts
            labels: Optional list of labels
            model: Model instance for tokenization
        """
        self.texts = texts
        self.labels = labels
        self.model = model
    
    def __len__(self) -> int:
        return len(self.texts)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        text = self.texts[idx]
        
        # Prepare single example
        inputs = self.model.prepare_input([text])
        item = {
            "input_ids": inputs["input_ids"][0],
            "attention_mask": inputs["attention_mask"][0]
        }
        
        # Add label if available
        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[idx])
        
        return item

class NewsTrainer:
    """Trainer for news classification model."""
    
    def __init__(
        self,
        model: NewsClassifier,
        train_texts: List[str],
        train_labels: List[int],
        val_texts: List[str],
        val_labels: List[int],
        batch_size: int = 16,
        learning_rate: float = 2e-5,
        num_epochs: int = 3,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """Initialize trainer.
        
        Args:
            model: Model instance to train
            train_texts: Training text samples
            train_labels: Training labels
            val_texts: Validation text samples
            val_labels: Validation labels
            batch_size: Training batch size
            learning_rate: Learning rate for optimization
            num_epochs: Number of training epochs
            device: Device to train on
        """
        self.model = model
        self.device = device
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        
        # Move model to device
        self.model.to(device)
        
        # Create datasets
        self.train_dataset = NewsDataset(train_texts, train_labels, model)
        self.val_dataset = NewsDataset(val_texts, val_labels, model)
        
        # Create dataloaders
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=True
        )
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=batch_size
        )
        
        # Setup optimizer and scheduler
        self.optimizer = AdamW(
            model.parameters(),
            lr=learning_rate
        )
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=num_epochs
        )
        
        # Training history
        self.history = {
            "train_loss": [],
            "val_loss": [],
            "val_accuracy": [],
            "val_f1": []
        }
    
    def train_epoch(self) -> float:
        """Train for one epoch.
        
        Returns:
            Average training loss
        """
        self.model.train()
        total_loss = 0
        
        for batch in tqdm(self.train_loader, desc="Training"):
            # Move batch to device
            batch = {k: v.to(self.device) for k, v in batch.items()}
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(**batch)
            loss = outputs["loss"]
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(self.train_loader)
    
    def evaluate(self) -> Dict[str, float]:
        """Evaluate model on validation set.
        
        Returns:
            Dictionary of evaluation metrics
        """
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Evaluating"):
                # Move batch to device
                batch = {k: v.to(self.device) for k, v in batch.items()}
                labels = batch.pop("labels")
                
                # Forward pass
                outputs = self.model(**batch, labels=labels)
                loss = outputs["loss"]
                logits = outputs["logits"]
                
                # Get predictions
                preds = torch.argmax(logits, dim=1)
                
                # Collect results
                total_loss += loss.item()
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Calculate metrics
        metrics = {
            "loss": total_loss / len(self.val_loader),
            "accuracy": accuracy_score(all_labels, all_preds),
            "f1": f1_score(all_labels, all_preds, average="weighted"),
            "precision": precision_score(all_labels, all_preds, average="weighted"),
            "recall": recall_score(all_labels, all_preds, average="weighted")
        }
        
        return metrics
    
    def train(self) -> Dict[str, List[float]]:
        """Train the model.
        
        Returns:
            Training history
        """
        best_f1 = 0
        
        for epoch in range(self.num_epochs):
            print(f"\nEpoch {epoch + 1}/{self.num_epochs}")
            
            # Training
            train_loss = self.train_epoch()
            
            # Validation
            val_metrics = self.evaluate()
            
            # Update learning rate
            self.scheduler.step()
            
            # Save best model
            if val_metrics["f1"] > best_f1:
                best_f1 = val_metrics["f1"]
                self.save_checkpoint("best_model.pt")
            
            # Update history
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_metrics["loss"])
            self.history["val_accuracy"].append(val_metrics["accuracy"])
            self.history["val_f1"].append(val_metrics["f1"])
            
            # Print metrics
            print(f"Train Loss: {train_loss:.4f}")
            print(f"Val Loss: {val_metrics['loss']:.4f}")
            print(f"Val Accuracy: {val_metrics['accuracy']:.4f}")
            print(f"Val F1: {val_metrics['f1']:.4f}")
        
        return self.history
    
    def save_checkpoint(self, filename: str) -> None:
        """Save training checkpoint.
        
        Args:
            filename: Name of checkpoint file
        """
        checkpoint = {
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": self.scheduler.state_dict(),
            "history": self.history
        }
        torch.save(checkpoint, filename)
    
    def load_checkpoint(self, filename: str) -> None:
        """Load training checkpoint.
        
        Args:
            filename: Name of checkpoint file
        """
        checkpoint = torch.load(filename)
        self.model.load_state_dict(checkpoint["model_state"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state"])
        self.history = checkpoint["history"] 