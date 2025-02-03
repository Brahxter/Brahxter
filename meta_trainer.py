import torch
import torch.nn as nn
import random
from torch.func import functional_call


class MetaTrainer:
    def __init__(self, model, tasks, inner_lr=0.001, adaptation_steps=1):
        """
        model: The meta-learning model (an instance of MetaLearningTransformer)
        tasks: A list of MarketTask instances
        inner_lr: Learning rate for inner-loop adaptation updates
        adaptation_steps: Number of gradient-update steps on the support set
        """
        self.model = model
        self.tasks = tasks
        self.inner_lr = inner_lr
        self.adaptation_steps = adaptation_steps
        self.criterion = nn.MSELoss()

    def train_step(self, task_batch_size=4):
        """
        For a batch of tasks, run inner-loop adaptation on each task's support set
        and compute the loss on the query set using the adapted parameters.
        Returns the average meta-loss over the selected tasks.
        """
        total_meta_loss = 0.0

        if len(self.tasks) < task_batch_size:
            task_batch = self.tasks
        else:
            task_batch = random.sample(self.tasks, task_batch_size)

        for task in task_batch:
            support_set = task.get_support_set()
            query_set = task.get_query_set()

            # Get the initial parameters of the model
            params = dict(self.model.named_parameters())

            # Perform inner-loop adaptation steps using the support set
            for _ in range(self.adaptation_steps):
                # Forward pass on the support set using the current parameters.
                support_preds = torch.func.functional_call(
                    self.model, params, support_set['prices'])
                support_loss = self.criterion(
                    support_preds, support_set['target'])
                # Compute gradients with respect to the model parameters.
                grads = torch.autograd.grad(
                    support_loss, params.values(), create_graph=True)
                # Update parameters with a gradient descent step.
                params = {name: param - self.inner_lr * grad
                          for (name, param), grad in zip(params.items(), grads)}

            # Evaluate on the query set using the adapted parameters.
            query_preds = torch.func.functional_call(
                self.model, params, query_set['prices'])
            query_loss = self.criterion(query_preds, query_set['target'])
            total_meta_loss += query_loss

        avg_meta_loss = total_meta_loss / task_batch_size
        return avg_meta_loss
