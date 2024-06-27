from torch.utils.tensorboard import SummaryWriter
import torch
import numpy as np
from src.utils import seed_everything, prepare_savedir
from src.data import DailyDehliClimate
from src.modelling import ExampleGenerator, SupervisedTrainer
from src.models.time_trans.full_ts_trans import TimeFormer
from src.visuals import DisplayPreds
from src.config.mainconf import Config
from src.utils import FileIO


conf = Config()
savedir = prepare_savedir(conf, 'DailyDelhiClimate')
# Laod data.
train, dev, test = DailyDehliClimate.load(conf.datasets.dailydelhi_climate,
                                          train_perc=0.8)
train_examples = ExampleGenerator.sample_seqs(train, conf.model.max_seq_len)
dev_examples = ExampleGenerator.sample_seqs(dev, conf.model.max_seq_len,
                                            rand_shuffle=False)
test_examples = ExampleGenerator.sample_seqs(test, conf.model.max_seq_len,
                                             rand_shuffle=False)


TRAIN = False
if TRAIN:
    seed_everything(conf.seed)
    tb_logger = SummaryWriter(log_dir=f'{savedir}/tb-logs', comment="000")

    # Model.
    model = TimeFormer(
        past_values_size=conf.model.values_size,
        time_feats_size=conf.model.time_feats_size,
        emb_size=conf.model.emb_size,
        n_att_heads=conf.model.n_att_heads,
        dim_feedforward=conf.model.dim_feedforward,
        depth=conf.model.depth,
        dropout=conf.training.p_dropout,
        out_size=conf.model.out_size,
        attention_type=conf.model.attention_type
    )

    # Train.
    trainer = SupervisedTrainer(conf.training.standardize,
                                conf.training.differentiate,
                                conf.training.patience, conf.training.delta,
                                conf.device)
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
    FileIO.write_json(conf.model_dump(), f'{savedir}/config.json')
    model.save(savedir)

TEST = True
if TEST:
    # Load model.
    model = TimeFormer.load(savedir)
    configs = FileIO.read_json(f'{savedir}/config.json')
    trainer = SupervisedTrainer(standardize=configs['training']['standardize'],
                                differentiate=configs['training']['differentiate'],
                                device=conf.device)

    # Plot predicitons for one example.
    dates = [x.dates for x in test_examples]
    inps = torch.stack([x.input_seq for x in test_examples])
    targs = torch.stack([x.target_seq for x in test_examples])
    with torch.no_grad():
        preds, attn_weights, attn1, attn2 = \
            trainer.pred_step(model, dates, inps)
    loss = torch.sqrt(trainer.criterion(preds, targs.to(conf.device)))
    print(f"Test loss: {loss}")
    FileIO.write_json({'test_loss': loss.item()}, f'{savedir}/test_loss.json')

    # Plot attention weights.
    DisplayPreds.plot_AttentionWeights(attn_weights, attn1, attn2,
                                       f'{savedir}/attention_weights_heatmap.png')

    feats = ['meantemp', 'humidity', 'wind_speed', 'meanpressure']
    for i in range(4):
        feat = feats[i]
        ps = preds[:, -1, i].cpu().numpy()
        ts = targs[:, -1, i].cpu().numpy()
        ds = np.array([str(x[-1])[:10] for x in dates])
        # DisplayPreds.plot_PredsVsTrues(ds, ps, ts, feat)

print("Done!")
