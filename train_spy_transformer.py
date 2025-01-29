import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from spy_meta_transformer import MetaLearningTransformer, SPYDataset
import numpy as np
from datetime import datetime
import os

def train_model(resume_from=None):
    print("Loading dataset with improved normalization...")
    dataset = SPYDataset('data/spy_5min_data_2021_2024.csv')
    
    # Split dataset into train and validation
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)
    
    # Initialize model and training components
    model = MetaLearningTransformer(d_model=256, nhead=8, num_layers=4, dropout=0.2)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5)
    
    # Initialize training state
    start_epoch = 0
    best_val_loss = float('inf')
    num_epochs = 100
    
    # Load checkpoint if resuming
    if resume_from:
        checkpoint = torch.load(resume_from)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint['val_loss']
        print(f'Resuming from epoch {start_epoch}')
    
    # Create checkpoint directory
    checkpoint_dir = 'models/checkpoints'
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    for epoch in range(start_epoch, num_epochs):
        model.train()
        train_loss = 0
        for batch in train_loader:
            if batch is None:
                continue
                
            optimizer.zero_grad()
            output = model(batch)
            loss = criterion(output, batch['target'])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0
        predictions = []
        actuals = []
        
        with torch.no_grad():
            for batch in val_loader:
                if batch is None:
                    continue
                    
                output = model(batch)
                val_loss += criterion(output, batch['target']).item()
                predictions.extend(output.numpy())
                actuals.extend(batch['target'].numpy())
        
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        
        # Calculate prediction accuracy metrics
        predictions = np.array(predictions)
        actuals = np.array(actuals)
        direction_accuracy = np.mean(np.sign(predictions) == np.sign(actuals))
        
        print(f'Epoch {epoch+1}:')
        print(f'Train Loss: {train_loss:.6f}')
        print(f'Val Loss: {val_loss:.6f}')
        print(f'Direction Accuracy: {direction_accuracy:.2%}')
        
        scheduler.step(val_loss)
        
        # Save periodic checkpoint every 5 epochs
        if (epoch + 1) % 5 == 0:
            checkpoint_path = f'{checkpoint_dir}/spy_transformer_epoch_{epoch+1}.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss
            }, checkpoint_path)
            print(f'Periodic checkpoint saved: {checkpoint_path}')
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'models/best_spy_transformer.pth')
            print(f'New best model saved with validation loss: {val_loss:.6f}')
        
        print('-' * 50)

if __name__ == "__main__":
    # To start fresh training:
    train_model()
    
    # To resume from a checkpoint, uncomment and specify the checkpoint path:
    # train_model('models/checkpoints/spy_transformer_epoch_25.pth')
