import logging
from clidantic import Parser
from torch.utils.tensorboard import SummaryWriter
import torch
from src.utils import seed_everything, prepare_savedir, FileIO
from src.data import ElectricProdDataset
from src.modelling import ExampleGenerator, SupervisedTrainer
from src.models.timetrans.transformer import TimeFormer
from src.visuals import DisplayPreds
from src.config.mainconf import Config


logger = logging.getLogger('timeformer')
cli = Parser()

@cli.command()
def train(conf: Config):
    # Init stuff.
    seed_everything(conf.seed)
    savedir = prepare_savedir(conf)
    tb_logger = SummaryWriter(log_dir=f'{savedir}/tb-logs', comment="000")

    # Laod data.
    train, dev, _ = ElectricProdDataset.load(
        conf.datasets.electricity_production, train_perc=0.7, dev_perc=0.1
    )
    train_examples = ExampleGenerator.sample_seqs(train, 300, conf.model.max_seq_len)
    dev_examples = ExampleGenerator.sample_seqs(dev, 50, conf.model.max_seq_len)

    # Model.
    model = TimeFormer(
        input_size=conf.model.input_size,
        emb_size=conf.model.emb_size,
        n_att_heads=conf.model.n_att_heads,
        dim_feedforward=conf.model.dim_feedforward,
        depth=conf.model.depth,
        dropout=conf.training.p_dropout,
        max_seq_len=conf.model.max_seq_len,
        out_size=conf.model.out_size
    )

    # Train.
    trainer = SupervisedTrainer(conf.training.patience, conf.training.delta, conf.device)
    model = trainer.train(
        model=model,
        data_train=train_examples,
        data_valid=dev_examples,
        epochs=conf.training.epochs,
        batch_size=conf.training.batch_size,
        lr=conf.training.lr,
        tb_logger=tb_logger
    )

    # Save.
    FileIO.write_json(conf.dict(), f'{savedir}/config.json')
    model.save(savedir)


@cli.command()
def test(conf: Config):
    # Load data.
    _, _, test = ElectricProdDataset.load(
        conf.datasets.electricity_production, train_perc=0.7, dev_perc=0.1
    )
    test_examples = ExampleGenerator.sample_seqs(test, 100, conf.model.max_seq_len)

    # Load model.
    savedir = prepare_savedir(conf)
    model = TimeFormer.load(savedir)

    # Test.
    trainer = SupervisedTrainer()
    loss_test = trainer.evaluation(model, test_examples)
    logger.info(f'Loss test: {loss_test}')

    # Plot predicitons for one example.
    ex = test_examples[0]
    with torch.no_grad():
        inps = ex.input_seq.unsqueeze(-1).unsqueeze(0)
        preds = model(inps.to(conf.device))
    DisplayPreds.plot(preds.squeeze().tolist(), ex.target_seq.tolist())

if __name__ == "__main__":
    cli()