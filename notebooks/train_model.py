# %% [markdown]
# # ClaimCracker v2 - Model Training
# 
# This notebook implements the training pipeline for our fake news detection model using Google Colab's GPU.
# 
# ## Setup
# 1. Upload dataset to Google Drive
# 2. Clone repository
# 3. Install dependencies
# 4. Train model
# 5. Save and evaluate results

# %% [code]
# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# %% [code]
# Clone repository and install dependencies
!git clone https://github.com/yourusername/ClaimCracker-v2.git
%cd ClaimCracker-v2
!pip install torch transformers pandas scikit-learn tqdm

# %% [code]
# Import dependencies
import sys
sys.path.append('src')

import torch
import pandas as pd
from pathlib import Path
from ml.models.classifier import NewsClassifier
from ml.training.trainer import NewsTrainer
from ml.config.training_config import TrainingConfig

# Verify GPU
print(f"Using device: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")

# %% [code]
# Load and prepare data
df = pd.read_csv('Dataset/Dataset.csv')

# Convert labels to integers
label_map = {'Real': 0, 'Fake': 1}
df['label_idx'] = df['Label'].map(label_map)

# Create train/val split
from sklearn.model_selection import train_test_split

train_df, val_df = train_test_split(
    df,
    test_size=0.1,
    random_state=42,
    stratify=df['Label']
)

print(f"Training samples: {len(train_df)}")
print(f"Validation samples: {len(val_df)}")

# Show class distribution
print("\nClass distribution:")
print(train_df['Label'].value_counts(normalize=True))

# %% [code]
# Load configuration
config = TrainingConfig(
    model_name="distilbert-base-uncased",
    batch_size=16,
    learning_rate=2e-5,
    num_epochs=3
)

# Create model
model = NewsClassifier(
    model_name=config.model_name,
    num_classes=config.num_classes,
    dropout=config.dropout,
    max_length=config.max_length
)

# Prepare texts and labels
train_texts = train_df['News_Text'].tolist()
train_labels = train_df['label_idx'].tolist()
val_texts = val_df['News_Text'].tolist()
val_labels = val_df['label_idx'].tolist()

# Create trainer
trainer = NewsTrainer(
    model=model,
    train_texts=train_texts,
    train_labels=train_labels,
    val_texts=val_texts,
    val_labels=val_labels,
    batch_size=config.batch_size,
    learning_rate=config.learning_rate,
    num_epochs=config.num_epochs
)

# %% [code]
# Train model
history = trainer.train()

# Plot training history
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 4))

# Loss plot
plt.subplot(1, 2, 1)
plt.plot(history['train_loss'], label='Train')
plt.plot(history['val_loss'], label='Val')
plt.title('Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# Metrics plot
plt.subplot(1, 2, 2)
plt.plot(history['val_accuracy'], label='Accuracy')
plt.plot(history['val_f1'], label='F1')
plt.title('Validation Metrics')
plt.xlabel('Epoch')
plt.ylabel('Score')
plt.legend()

plt.tight_layout()
plt.show()

# Print final metrics
print("\nFinal Validation Metrics:")
print(f"Accuracy: {history['val_accuracy'][-1]:.4f}")
print(f"F1 Score: {history['val_f1'][-1]:.4f}")

# %% [code]
# Save model to Drive
save_path = Path('/content/drive/MyDrive/ClaimCracker/models/best_model')
model.save_pretrained(save_path)

# Save training config
config.save(save_path / 'training_config.json')

print(f"Model saved to {save_path}")

# %% [markdown]
# ## Next Steps
# 
# 1. The trained model is saved and can be loaded using:
# ```python
# model = NewsClassifier.from_pretrained('path/to/best_model')
# ```
# 
# 2. For inference, use:
# ```python
# texts = ["Your news article text here"]
# inputs = model.prepare_input(texts)
# with torch.no_grad():
#     outputs = model(**inputs)
# predictions = torch.argmax(outputs['logits'], dim=1)
# ``` 
