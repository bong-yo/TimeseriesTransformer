import logging
from clidantic import Parser
from torch.utils.tensorboard import SummaryWriter
from src.utils import seed_everything
from src.data import ElectricProdDataset
from src.modelling import ExampleGenerator, SupervisedTrainer
from src.models.transformer import TimeFormer
from src.config.base import Config


logger = logging.getLogger('timeformer')
cli = Parser()

@cli.command()
def run(conf: Config):
    seed_everything(conf.seed)
    tb_logger = SummaryWriter(log_dir=conf.outputs_folder, comment="000")

    train, dev, test = ElectricProdDataset.load(
        conf.datasets.electricity_production, train_perc=0.7, dev_perc=0.1
    )

    model = TimeFormer(
        conf.model.input_size,
        conf.model.emb_size,
        conf.model.n_att_heads,
        conf.model.depth,
        conf.training.p_dropout,
        conf.model.max_seq_len,
        conf.model.out_size
    )

    # Sample example sequences.
    train_examples = ExampleGenerator.sample_seqs(train, 300, conf.model.max_seq_len)
    dev_examples = ExampleGenerator.sample_seqs(dev, 50, conf.model.max_seq_len)
    test_examples = ExampleGenerator.sample_seqs(test, 100, conf.model.max_seq_len)

    # Train.
    trainer = SupervisedTrainer(conf.training.patience, conf.training.delta, conf.device)
    model = trainer.train(
        model,
        train_examples,
        dev_examples,
        conf.training.epochs,
        conf.training.batch_size,
        conf.training.lr,
        tb_logger
    )

    # Test.
    loss_test = trainer.evaluation(model, test_examples)
    logger.info(f'Loss test: {loss_test}')


if __name__ == "__main__":
    cli()