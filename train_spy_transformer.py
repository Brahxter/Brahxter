import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os
from spy_meta_transformer import MetaLearningTransformer, SPYDataset


def train_model(checkpoint_path=None):
    # Initialize model and training parameters
    model = MetaLearningTransformer(
        d_model=256, nhead=8, num_layers=4, dropout=0.2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    criterion = nn.MSELoss()
    num_epochs = 100

    # Create checkpoint directory if it doesn't exist
    checkpoint_dir = 'models/checkpoints'
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Load datasets
    train_dataset = SPYDataset('data/spy_train.csv')
    val_dataset = SPYDataset('data/spy_val.csv')

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # Initialize best metrics tracking
    best_val_loss = float('inf')
    best_accuracy = 0
    best_epoch = 0

    # Resume from checkpoint if provided
    start_epoch = 0
    if checkpoint_path and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f'Resuming from epoch {start_epoch}')

    for epoch in range(start_epoch, num_epochs):
        # Training phase
        model.train()
        train_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            output = model(batch)
            loss = criterion(output, batch['target'])
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        # Validation phase
        model.eval()
        val_loss = 0
        correct_predictions = 0
        total_predictions = 0

        with torch.no_grad():
            for batch in val_loader:
                output = model(batch)
                val_loss += criterion(output, batch['target']).item()

                # Calculate direction accuracy
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

        # Update best metrics
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch + 1
            torch.save(model.state_dict(), 'models/best_spy_transformer.pth')
            print(f'New best model saved with validation loss: {val_loss:.6f}')

        if direction_accuracy > best_accuracy:
            best_accuracy = direction_accuracy

        # Save periodic checkpoint
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

    # Print final statistics
    print_final_stats(best_val_loss, best_accuracy, best_epoch)


def print_final_stats(best_val_loss, best_accuracy, best_epoch):
    print("\n=== Final Model Statistics ===")
    print(f"Best Validation Loss: {best_val_loss:.6f}")
    print(f"Best Direction Accuracy: {best_accuracy:.2f}%")
    print(f"Best Model Epoch: {best_epoch}")
    print("============================")


if __name__ == "__main__":
    train_model()
    # To resume from checkpoint:
    # train_model('models/checkpoints/spy_transformer_epoch_25.pth')
