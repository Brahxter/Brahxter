import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from spy_meta_transformer import SPYDataset, MetaLearningTransformer
from tqdm import tqdm


def train_model():
    print("Loading dataset with improved normalization...")
    dataset = SPYDataset('data/spy_5min_data_2021_2024.csv')

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)

    print("Initializing enhanced model...")
    model = MetaLearningTransformer(
        d_model=256, nhead=8, num_layers=4, dropout=0.2)

    # Improved optimizer settings
    optimizer = optim.AdamW(model.parameters(), lr=0.0001, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2)
    criterion = nn.HuberLoss(delta=1.0)  # More robust loss function

    num_epochs = 50
    best_val_loss = float('inf')
    patience = 7
    no_improve = 0

    print("Starting training with enhanced architecture...")
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0

        for batch in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}'):
            optimizer.zero_grad()
            output = model(batch)
            loss = criterion(output, batch['target'])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()

        scheduler.step()

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                output = model(batch)
                val_loss += criterion(output, batch['target']).item()

        train_loss /= len(train_loader)
        val_loss /= len(val_loader)

        print(f'\nEpoch {epoch+1}/{num_epochs}')
        print(f'Training Loss: {train_loss:.6f}')
        print(f'Validation Loss: {val_loss:.6f}')

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'models/best_spy_transformer.pth')
            print(f"New best model saved! Loss: {val_loss:.6f}")
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print("Early stopping triggered!")
                break


if __name__ == "__main__":
    train_model()
