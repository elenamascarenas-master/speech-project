import torch
import pytorch_lightning as pl
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import os
import numpy as np
from pytorch_lightning.callbacks import TQDMProgressBar, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
import wandb


folder_path = './mfccs_files_clean'

mfcc_data = []
labels = []
for file_name in os.listdir(folder_path):
    # Construct the full path to the numpy file
    file_path = os.path.join(folder_path, file_name)
    # fake and real encoded as 0,1
    if "bona-fide" in file_name:
        labels.append(1)
    if "spoof" in file_name:
        labels.append(0)
    mfcc_array = np.load(file_path)
    mfcc_data.append(mfcc_array)

# Convert the list of MFCC arrays to a single 3D numpy array
mfcc_data = np.array(mfcc_data)

X = mfcc_data

X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.1, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.23, random_state=32)


class AudioDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        audio = self.X[idx]
        label = self.y[idx]
        return audio, label


class CNNModel(pl.LightningModule):
    def __init__(self, learning_rate=1e-3, input_shape=(13,1077), num_classes=2):
        super(CNNModel, self).__init__()
        # Define your CNN architecture
        self.save_hyperparameters()
        self.test_step_outputs = []

        self.model = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Linear(32 * (input_shape[0] // 4) * (input_shape[1] // 4), 128),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        # Forward pass through the network
        batch_size = x.shape[0]
        x = x.view(batch_size, 1, x.shape[1], -1)
        return self.model(x)

    def training_step(self, batch, batch_idx):
        # Training step
        x, y = batch
        y_hat = self(x)
        preds = torch.argmax(torch.sigmoid(y_hat), dim=1)
        acc = (preds == y).float().mean()
        loss = nn.CrossEntropyLoss()(y_hat, y)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_acc', acc, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        # Validation step
        x, y = batch
        y_hat = self(x)
        preds = torch.argmax(torch.sigmoid(y_hat), dim=1)
        acc = (preds == y).float().mean()
        loss = nn.CrossEntropyLoss()(y_hat, y)
        self.log('val_loss', loss, on_epoch=True, prog_bar=True)
        self.log('val_acc', acc, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        # Define optimizer and learning rate scheduler
        optimizer = optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        print("Learning Rate:", self.hparams.learning_rate)
        return optimizer

    def test_step(self, batch, batch_idx):
        # Test step
        x, y = batch
        y_hat = self(x)
        preds = torch.argmax(y_hat, dim=1)
        acc = (preds == y).float().mean()
        self.test_step_outputs.append(acc)
        self.log('test_acc', acc)

    def on_test_epoch_end(self):
        # Log average test accuracy
        avg_test_acc = torch.stack([x for x in self.test_step_outputs]).mean()
        self.log('avg_test_acc', avg_test_acc, prog_bar=True)


# Create datasets and dataloaders
train_dataset = AudioDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
val_dataset = AudioDataset(X_val, y_val)
val_loader = DataLoader(val_dataset, batch_size=4)
test_dataset = AudioDataset(X_test, y_test)
test_loader = DataLoader(test_dataset, batch_size=8)

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
wandb.finish()
wandb.init(project="Fake_Audio_Detection")
wandb_logger = WandbLogger(project="Fake_Audio_Detection", log_model=True)
# Initialize model and trainer
model = CNNModel(learning_rate=1e-3)
experiment_num = 6
checkpoint_callback = ModelCheckpoint(
    monitor='val_loss',
    dirpath='checkpoints',
    filename=f'cnn_model-{experiment_num}'+'-{epoch:02d}-{val_loss:.2f}',
    save_top_k=3,  # Save the top 3 models based on validation loss
    mode='min',  # Save the models with minimum validation loss
)

trainer = pl.Trainer(max_epochs=10, callbacks=[TQDMProgressBar(), checkpoint_callback], logger=wandb_logger)

# summary(model, (4, 13, 1077))

# Train the model
trainer.fit(model, train_loader, val_loader)
trainer.test(dataloaders=test_loader)
