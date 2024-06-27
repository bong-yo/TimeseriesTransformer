from typing import List, Iterator
import logging
import random
from copy import deepcopy
import torch
from torch import Tensor
from torch.nn import MSELoss
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils import tensorboard
from src.models.time_trans.transformer import TimeFormer
from src.utils import create_time_features, differentiate, standardize


logger = logging.getLogger('timeformer')


class Example:
    """Holds inputs & target examples for training/testing the model"""
    def __init__(self, dates: list[str], values: list[float]) -> None:
        self.dates = dates[:-1]
        self.input_seq = Tensor(values[:-1])
        self.target_seq = Tensor(values[1:])


class ExampleGenerator:
    """Extract 'n_examples' seqs, of length 'seq_len',
    at random from the initial sequence of data"""
    @staticmethod
    def sample_seqs(seq: List[float], seq_len: int, n_examples: int = None,
                    rand_shuffle: bool = True) -> Iterator[Example]:
        dates, values = seq['dates'], seq['values']
        N = len(dates)
        if n_examples is None:
            n_examples = N - seq_len
        else:
            n_examples = min(N - seq_len, n_examples)
        if rand_shuffle:
            seq_starts = random.sample(range(N - seq_len), n_examples)
        else:
            seq_starts = range(n_examples)
        return [Example(dates[start: start + seq_len],
                        values[start: start + seq_len])
                for start in seq_starts]


class Batcher:
    @staticmethod
    def create_batches(
            data: List[Example],
            size: int,
            shuffle: bool = True) -> Iterator[tuple[Tensor, Tensor]]:
        n = len(data)
        if shuffle:
            random.shuffle(data)
        n_batches = n // size + int(n % size != 0)
        for i in range(n_batches):
            batch = data[i * size: (i + 1) * size]
            dates = [x.dates for x in batch]
            inps = torch.stack([x.input_seq for x in batch])
            targs = torch.stack([x.target_seq for x in batch])
            yield dates, inps, targs


class SupervisedTrainer:
    def __init__(self, standardize: bool = False, differentiate: bool = False,
                 patience: int = 5, delta: float = 0,
                 device: str = 'cpu') -> None:
        super(SupervisedTrainer, self).__init__()
        self.standardize = standardize
        self.differentiate = differentiate
        self.patience = patience
        self.delta = delta
        self.patience_count = 0
        self.best_metric = float("inf")
        self.best_model = None
        self.train_msg = "Epoch: %d | Train L: %.3f | Valid  L: %.3f"
        self.device = device
        self.criterion = MSELoss()

    def pred_step(self, model: TimeFormer, dates: Tensor, inp_seq: Tensor) -> Tensor:
        # Preprocess time feats, differential and standardise.
        temp_feats = create_time_features(dates)
        if self.differentiate:
            prev_day = inp_seq.clone()
            inp_seq = differentiate(inp_seq)
        if self.standardize:
            inp_seq, means, stds = standardize(inp_seq)
        # Forward.
        preds, attn_weights, attn1, attn2 = \
            model(inp_seq.to(self.device), temp_feats.to(self.device))
        # Revert pre-processing.
        if self.standardize:
            preds = preds * stds.unsqueeze(1).to(self.device) + \
                means.unsqueeze(1).to(self.device)
        if self.differentiate:
            preds = prev_day + preds
        return preds, attn_weights, attn1, attn2

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
        scheduler = ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

        self.best_model = self.best_model.to(self.device)
        # Actual training.
        for epoch in range(0, epochs + 1):
            model.train()

            n, rmse_train = 0, 0
            for dates, inp_seq, targs in Batcher.create_batches(data_train,
                                                                batch_size):
                optimizer.zero_grad()
                preds = self.pred_step(model, dates, inp_seq)[0]
                loss = torch.sqrt(self.criterion(preds, targs.to(self.device)))
                if epoch > 0:  # First epoch keep untrained model for baseline performance.
                    loss.backward()
                    optimizer.step()
                rmse_train += loss.item()
                n += 1
            rmse_train /= n
            tb_logger.add_scalar('Loss/train', rmse_train, epoch)
            rmse_eval = self.evaluation(model, data_valid)
            if epoch > 0:
                scheduler.step(rmse_eval)
            # Log into tensorboard the evaluation results and print.
            tb_logger.add_scalar('Loss/eval', rmse_eval, epoch)

            keep_waiting, best = self.has_improved(rmse_eval, model)
            logger.info(self.train_msg % (epoch, rmse_train, rmse_eval
                                          ) + (" - Best!" if best else ""))
            if not keep_waiting:
                break

        # logger.debug(f'LogReg training finished! Best F1: {self.best_metric}')
        return self.best_model

    def evaluation(self, model: TimeFormer, data_eval: List[Example]) -> float:
        model.eval()
        n, rmse = 0, 0
        with torch.no_grad():
            for dates, inp_seq, targs in Batcher.create_batches(data_eval,
                                                                size=100):
                preds = self.pred_step(model, dates, inp_seq)[0]
                rmse += torch.sqrt(self.criterion(preds, targs.to(self.device))).item()
                n += 1
        return rmse / n

    def has_improved(self, metric: float, model: TimeFormer) -> tuple[bool, bool]:
        """Check if the 'Metric Of Reference' has improved in the last 'patience' steps,
        and save best model."""
        if metric == 0:
            return True, True
        if self.best_metric - metric > self.delta:  # Metric improved (more than delta).
            self.best_metric = metric
            self.best_model = deepcopy(model)
            self.patience_count = 0
            return True, True
        elif self.patience_count < self.patience:  # Metric didn't improve but still patience.
            self.patience_count += 1
            return True, False
        return False, False


class GPTrainer:
    def __init__(self, standardize: bool = False, differentiate: bool = False,
                 patience: int = 5, delta: float = 0,
                 device: str = 'cpu') -> None:
        super(SupervisedTrainer, self).__init__()
        self.standardize = standardize
        self.differentiate = differentiate
        self.patience = patience
        self.delta = delta
        self.patience_count = 0
        self.best_metric = float("inf")
        self.best_model = None
        self.train_msg = "Epoch: %d | Train L: %.3f | Valid  L: %.3f"
        self.device = device
        self.criterion = MSELoss()

    def pred_step(self, model: TimeFormer, dates: Tensor, inp_seq: Tensor) -> Tensor:
        # Preprocess time feats, differential and standardise.
        temp_feats = create_time_features(dates)
        if self.differentiate:
            prev_day = inp_seq.clone()
            inp_seq = differentiate(inp_seq)
        if self.standardize:
            inp_seq, means, stds = standardize(inp_seq)
        # Forward.
        preds, attn_weights, attn1, attn2 = \
            model(inp_seq.to(self.device), temp_feats.to(self.device))
        # Revert pre-processing.
        if self.standardize:
            preds = preds * stds.unsqueeze(1).to(self.device) + \
                means.unsqueeze(1).to(self.device)
        if self.differentiate:
            preds = prev_day + preds
        return preds, attn_weights, attn1, attn2

