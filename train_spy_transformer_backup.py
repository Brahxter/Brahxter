import math
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os
from spy_meta_transformer import MetaLearningTransformer, SPYDataset


class CosineWarmupScheduler(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup_epochs, max_epochs):
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        super().__init__(optimizer)

    def get_lr(self):
        epoch = self.last_epoch
        if epoch < self.warmup_epochs:
            return [base_lr * (epoch + 1) / self.warmup_epochs for base_lr in self.base_lrs]
        else:
            progress = (epoch - self.warmup_epochs) / \
                (self.max_epochs - self.warmup_epochs)
            return [base_lr * 0.5 * (1 + math.cos(math.pi * progress)) for base_lr in self.base_lrs]


def train_model(checkpoint_path=None):
    model = MetaLearningTransformer(
        d_model=256, nhead=8, num_layers=4, dropout=0.2)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=1e-4, weight_decay=0.01)
    scheduler = CosineWarmupScheduler(
        optimizer, warmup_epochs=5, max_epochs=100)
    criterion = nn.MSELoss()
    num_epochs = 100

    checkpoint_dir = 'models/checkpoints'
    os.makedirs(checkpoint_dir, exist_ok=True)

    train_dataset = SPYDataset('data/spy_train.csv')
    val_dataset = SPYDataset('data/spy_val.csv')

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    best_val_loss = float('inf')
    best_accuracy = 0
    best_epoch = 0

    if checkpoint_path and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f'Resuming from epoch {start_epoch}')
    else:
        start_epoch = 0

    for epoch in range(start_epoch, num_epochs):
        model.train()
        train_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            output = model(batch)
            loss = criterion(output, batch['target'])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)
        scheduler.step()

        model.eval()
        val_loss = 0
        correct_predictions = 0
        total_predictions = 0

        with torch.no_grad():
            for batch in val_loader:
                output = model(batch)
                val_loss += criterion(output, batch['target']).item()

                pred_direction = (output > 0).float()
                true_direction = (batch['target'] > 0).float()
                correct_predictions += (pred_direction ==
                                        true_direction).sum().item()
                total_predictions += len(output)

        val_loss /= len(val_loader)
        direction_accuracy = (correct_predictions / total_predictions) * 100

        print(f'Epoch {epoch + 1}:')
        print(f'Train Loss: {train_loss:.6f}')
        print(f'Val Loss: {val_loss:.6f}')
        print(f'Direction Accuracy: {direction_accuracy:.2f}%')

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch + 1
            torch.save(model.state_dict(), 'models/best_spy_transformer.pth')
            print(f'New best model saved with validation loss: {val_loss:.6f}')

        if direction_accuracy > best_accuracy:
            best_accuracy = direction_accuracy

        if (epoch + 1) % 5 == 0:
            checkpoint_path = f'{
                checkpoint_dir}/spy_transformer_epoch_{epoch+1}.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss
            }, checkpoint_path)
            print(f'Periodic checkpoint saved: {checkpoint_path}')

        print('-' * 50)

    print_final_stats(best_val_loss, best_accuracy, best_epoch)


def print_final_stats(best_val_loss, best_accuracy, best_epoch):
    print("\n=== Final Model Statistics ===")
    print(f"Best Validation Loss: {best_val_loss:.6f}")
    print(f"Best Direction Accuracy: {best_accuracy:.2f}%")
    print(f"Best Model Epoch: {best_epoch}")
    print("============================")


if __name__ == "__main__":
    train_model()
