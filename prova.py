from os.path import abspath, dirname
from torch.utils.tensorboard import SummaryWriter
import torch
from src.utils import seed_everything, prepare_savedir, FileIO
from src.data import ElectricProdDataset, DailyTemp
from src.modelling import ExampleGenerator, SupervisedTrainer
from src.models.transformer import TimeFormer
from src.visuals import DisplayPreds
from src.config.mainconf import Config


conf = Config()
savedir = prepare_savedir(conf)
 # Laod data.
train, dev, test = DailyTemp.load(
    conf.datasets.daily_temperatures, train_perc=0.7, dev_perc=0.1
)
train_examples = ExampleGenerator.sample_seqs(train, 1000, conf.model.max_seq_len)
dev_examples = ExampleGenerator.sample_seqs(dev, 200, conf.model.max_seq_len)
test_examples = ExampleGenerator.sample_seqs(test, 400, conf.model.max_seq_len)


TRAIN = True
if TRAIN:
    seed_everything(conf.seed)
    tb_logger = SummaryWriter(log_dir=f'{savedir}/tb-logs', comment="000")

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

    # # Save.
    # FileIO.write_json(conf.dict(), f'{savedir}/config.json')
    # model.save(savedir)

TEST = True
if TEST:
    # # Load model.
    # model1 = TimeFormer.load(savedir)
    # configs = FileIO.read_json(f'{savedir}/config.json')
    # Test.
    trainer = SupervisedTrainer()

    # Plot predicitons for one example.
    ex = test_examples[0]
    with torch.no_grad():
        inps = ex.input_seq.unsqueeze(-1).unsqueeze(0)
        preds = model(inps.to(conf.device))
    DisplayPreds.plot(
        inps.squeeze().tolist(), preds.squeeze().tolist(), ex.target_seq.tolist()
    )