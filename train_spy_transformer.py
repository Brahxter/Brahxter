import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from spy_meta_transformer import MetaLearningTransformer, SPYDataset
from meta_trainer import MetaTrainer
from market_tasks import MarketTask
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TransformerTrainer:
    def __init__(self, model, train_loader, val_loader, learning_rate=0.001):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        # Create meta-learning tasks from training batches.
        # For each batch in the train_loader, create three tasks
        # corresponding to different trading patterns.
        self.market_tasks = self.create_market_tasks()
        self.meta_trainer = MetaTrainer(model=self.model, tasks=self.market_tasks,
                                        inner_lr=0.001, adaptation_steps=1)

    def create_market_tasks(self):
        tasks = []
        for batch in self.train_loader:
            if batch is None:
                continue
            trend_task = MarketTask(batch, 'trend_following')
            reversion_task = MarketTask(batch, 'mean_reversion')
            breakout_task = MarketTask(batch, 'breakout')
            tasks.extend([trend_task, reversion_task, breakout_task])
        return tasks

    def train_epoch(self):
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        for batch_idx, batch in enumerate(self.train_loader):
            if batch is None:
                continue

            # Extract regular training data:
            prices = batch['prices']    # shape: (batch, seq, 4)
            targets = batch['target']    # shape: (batch, 1)

            self.optimizer.zero_grad()
            # Forward pass (model internally projects input features to d_model)
            # expected output shape: (batch, d_model) or (batch, 1) if further projected
            predictions = self.model(prices)
            loss = self.criterion(predictions, targets)

            # Compute meta-learning loss via inner-loop adaptation:
            meta_loss = self.meta_trainer.train_step(task_batch_size=4)
            total_batch_loss = loss + meta_loss

            total_batch_loss.backward()
            self.optimizer.step()

            total_loss += total_batch_loss.item()
            num_batches += 1
            if batch_idx % 100 == 0:
                logger.info(f"Batch {batch_idx}, Loss: {
                            total_batch_loss.item():.6f}")

        return total_loss / num_batches if num_batches > 0 else 0.0

    def validate(self):
        self.model.eval()
        total_loss = 0.0
        predictions_list = []
        targets_list = []
        with torch.no_grad():
            for batch in self.val_loader:
                if batch is None:
                    continue
                prices = batch['prices']
                targets = batch['target']
                outputs = self.model(prices)
                loss = self.criterion(outputs, targets)
                total_loss += loss.item()
                predictions_list.extend(outputs.cpu().numpy())
                targets_list.extend(targets.cpu().numpy())
        avg_loss = total_loss / \
            len(self.val_loader) if len(self.val_loader) > 0 else 0.0
        accuracy = self.calculate_accuracy(predictions_list, targets_list)
        return avg_loss, accuracy

    def calculate_accuracy(self, predictions, actuals):
        predictions = np.array(predictions)
        actuals = np.array(actuals)
        correct = np.sum(np.sign(predictions) == np.sign(actuals))
        return correct / len(predictions) if len(predictions) > 0 else 0.0


def main():
    checkpoint_dir = 'models'
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Data loading from CSV files using your SPYDataset implementation.
    train_dataset = SPYDataset('data/spy_train.csv')
    val_dataset = SPYDataset('data/spy_val.csv')

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # Initialize the transformer model. The model's forward() uses an input projection to map
    # raw 4-feature OHLC input into the d_model dimension.
    model = MetaLearningTransformer(
        d_model=256, nhead=8, num_layers=4, dropout=0.2)

    trainer = TransformerTrainer(
        model, train_loader, val_loader, learning_rate=0.001)

    num_epochs = 100
    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        logger.info(f"Epoch {epoch+1}/{num_epochs}")

        train_loss = trainer.train_epoch()
        logger.info(f"Training Loss: {train_loss:.6f}")

        val_loss, accuracy = trainer.validate()
        logger.info(f"Validation Loss: {
                    val_loss:.6f}, Accuracy: {accuracy:.2%}")

        # Save the best performing model:
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(
                checkpoint_dir, 'best_spy_transformer.pth'))
            logger.info("New best model saved!")

        # Periodic checkpoint saving
        if (epoch + 1) % 10 == 0:
            checkpoint_path = os.path.join(
                checkpoint_dir, f'spy_transformer_epoch_{epoch+1}.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': trainer.optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
            }, checkpoint_path)
            logger.info(f"Checkpoint saved: {checkpoint_path}")


if __name__ == "__main__":
    main()
