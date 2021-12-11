import torch
import src
import argparse
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from src.lightning_modules import LitModel, LitSNLI
from src.models import Atten, Encoder
from src.data_models import w2v

from pytorch_lightning.loggers import WandbLogger


parser = argparse.ArgumentParser()
parser.add_argument('--train_file', type=str, default='data/entail-train.hdf5', help='training data file (hdf5)')
parser.add_argument('--val_file', type=str, default='data/entail-val.hdf5', help='validation data file (hdf5)')
parser.add_argument('--test_file', type=str, default='data/entail-test.hdf5', help='test data file (hdf5)')
parser.add_argument('--max_length', type=int, default=10, help='maximum length of training sentences. -1 means no max length')
parser.add_argument('--w2v_file', type=str, default='data/w2v.hdf5', help='pretrained word vectors file (hdf5)')

parser.add_argument('--embedding_size', type=int, default=300, help='word embedding size')
parser.add_argument('--hidden_size', type=int, default=300, help='hidden layer size')

parser.add_argument('--epoch', type=int, default=250, help='number of training epochs')
parser.add_argument('--gpus', type=int, default=0, help='number of gpus to train on. -1 for all gpus')
parser.add_argument('--val_interval', type=int, default=500, help='interval for checking the validation dataset')

parser.add_argument('--optimizer', type=str, default='adagrad', choices=['adam', 'adagrad'])
parser.add_argument('--lr', type=float, default=0.05, help='learning rate')
parser.add_argument('--max_grad_norm', type=float, default=5, help='if the norm of the gradient vector exceeds this renormalize it to have the norm equal to the max_grad_norm')
parser.add_argument('--param_init', type=float, default=0.01, help='parameter initialization gaussian')
parser.add_argument('--weight_decay', type=float, default=5e-5, help='l2 regularization')

parser.add_argument('--log_dir', type=str, default='logs/', help='log file directory')
parser.add_argument('--model_path', type=str, default='model', help='path of the model (w/o suffix)')
parser.add_argument('--use_wandb', type=bool, default=False, help='whether to use wandb for logging')

args = parser.parse_args()
TRAIN_LBL_SIZE = 3

datamodule = LitSNLI(
    train_fname=args.train_file,
    val_fname=args.val_file,
    test_fname=args.test_file,
    max_length=args.max_length
)

word_vecs = w2v(args.w2v_file).word_vecs

encoder = Encoder(
    num_embeddings=word_vecs.size(0),
    embedding_size=args.embedding_size,
    hidden_size=args.hidden_size,
    param_init=args.param_init,
    intra_sent_atten=True,
)

encoder.embedding.weight.data.copy_(word_vecs)
encoder.embedding.weight.requires_grad = False

atten = Atten(
    hidden_size=args.hidden_size,
    label_size=TRAIN_LBL_SIZE,
    param_init=args.param_init
)

model = LitModel(
    encoder=encoder,
    atten=atten,
    max_grad_norm=args.max_grad_norm,
    lr=args.lr,
    optim=args.optimizer,
    weight_decay=args.weight_decay
)

if args.use_wandb:
    wandb_logger = WandbLogger(project="ANLP-Project")

checkpoint_callback = ModelCheckpoint( 
  monitor='Validation Loss', 
  dirpath='./saved_models/', 
  filename='anlp-model-{epoch:02d}',
  save_top_k = 3,
  mode='min'
)

trainer = pl.Trainer(
    max_epochs=args.epoch,
    gpus=args.gpus,
    accelerator='cpu' if args.gpus == 0 else 'ddp',
    progress_bar_refresh_rate=10,
    val_check_interval=args.val_interval,
    logger=wandb_logger if args.use_wandb else None,
    callbacks=[checkpoint_callback]
)

trainer.fit(
    model=model,
    datamodule=datamodule
)

trainer.save_checkpoint(args.model_path + ".ckpt")
