from typing import List, Iterator, Tuple
import logging
import random
from copy import deepcopy
from tqdm import tqdm
import torch
from torch import Tensor
from torch.nn import MSELoss
from torch.optim import AdamW
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils import tensorboard
from src.models.transformer import TimeFormer 
from src.config.training import TrainConfig


logger = logging.getLogger('timeformer')


class Example:
    """Holds inputs & target examples for training/testing the model"""
    def __init__(self, seq: List[float]) -> None:
        self.input_seq = Tensor(seq[:-1])
        self.target_seq = Tensor(seq[1:])


class ExampleGenerator:
    """Extract 'n_examples' seqs, of length 'seq_len', 
    at random from the initial sequence of data"""
    @staticmethod
    def sample_seqs(seq: List[float], n_examples: int, seq_len: int) -> Iterator[Example]:
        N = len(seq)
        n_examples = min(N - seq_len, n_examples)
        return [
            Example(seq[start: start + seq_len])
            for start in random.sample(range(N - seq_len), n_examples)
        ]


class Batcher:
    @staticmethod
    def create_batches(
            data: List[Example],
            size: int,
            shuffle: bool = True) -> List[Tuple[Tensor, Tensor]]:
        n = len(data)
        if shuffle:
            random.shuffle(data)
        n_batches = n // size + int(n % size != 0)
        for i in range(n_batches):
            batch = data[i * size: (i + 1) * size]
            inps = torch.stack([x.input_seq for x in batch]).unsqueeze(-1)
            targs = torch.stack([x.target_seq for x in batch]).unsqueeze(-1)
            yield inps, targs


class SupervisedTrainer:
    def __init__(self, patience: int, delta: float, device: str) -> None:
        super(SupervisedTrainer, self).__init__()
        self.patience = patience
        self.delta = delta
        self.patience_count = 0
        self.best_metric = float("inf")
        self.best_model = None
        self.train_msg = "Epoch: %d | Train L: %.3f | Valid  L: %.3f"
        self.device = device

    def train(self,
              model: TimeFormer,
              data_train: List[Example],
              data_valid: List[Example],
              epochs: int,
              batch_size: int,
              lr: float,
              tb_logger: tensorboard.writer.SummaryWriter = None) -> None:

        """Standard supervised training with BCE loss.
        Stop after validation loss hasn't improved for 'patience' epochs."""
        self.best_model = model

        optimizer = AdamW(model.parameters(), lr=lr)
        scheduler = ExponentialLR(optimizer, gamma=0.9)
        self.criterion = MSELoss()

        self.best_model = self.best_model.to(self.device)
        # Actual training.
        for epoch in range(1, epochs + 1):
            model.train()

            n, rmse_train = 0, 0
            for inp_seq, targ_seq in Batcher.create_batches(data_train, batch_size):
                inp_seq, targ_seq = inp_seq.to(self.device), targ_seq.to(self.device)

                optimizer.zero_grad()
                preds = model(inp_seq)
                loss = torch.sqrt(self.criterion(targ_seq, preds))
                rmse_train += loss.item()
                if epoch > 0:  # First epoch keep untrained model for baseline performance.
                    loss.backward()
                    optimizer.step()
                n += 1
            
            rmse_train /= n

            tb_logger.add_scalar('Loss/train', rmse_train, epoch)
            rmse_eval = self.evaluation(model, data_valid)

            # Log into tensorboard the evaluation results and print.
            tb_logger.add_scalar('Loss/eval', rmse_eval, epoch)
            logger.info(self.train_msg % (epoch, rmse_train, rmse_eval))

            if not self.has_improved(rmse_eval, model):
                break

        # logger.debug(f'LogReg training finished! Best F1: {self.best_metric}')
        return self.best_model

    def evaluation(self, model: TimeFormer, data_eval: List[Example]) -> float:
        model.eval()
        n, rmse = 0, 0
        with torch.no_grad():
            for inp_seq, targ_seq in Batcher.create_batches(data_eval, size=100):
                inp_seq, targ_seq = inp_seq.to(self.device), targ_seq.to(self.device)
                preds = model(inp_seq)
                rmse += torch.sqrt(self.criterion(preds, targ_seq)).item()
                n += 1
        return rmse / n

    def has_improved(self, metric: float, model: TimeFormer):
        """Check if the 'Metric Of Reference' has improved in the last 'patience' steps,
        and save best model."""
        if metric == 0:
            return True
        if self.best_metric - metric > self.delta:  # Metric improved (more than delta).
            self.best_metric = metric
            self.best_model = deepcopy(model)
            self.patience_count = 0
            return True
        elif self.patience_count < self.patience:  # Metric didn't improve but still patience.
            self.patience_count += 1
            return True
        return False
