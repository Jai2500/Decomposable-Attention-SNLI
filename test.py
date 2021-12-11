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
parser.add_argument('--max_length', type=int, default=42, help='maximum length of training sentences. -1 means no max length')
parser.add_argument('--w2v_file', type=str, default='data/w2v.hdf5', help='pretrained word vectors file (hdf5)')

parser.add_argument('--embedding_size', type=int, default=300, help='word embedding size')
parser.add_argument('--hidden_size', type=int, default=300, help='hidden layer size')
parser.add_argument(
    "--intra_sent_atten",
    action="store_true",
    help="whether to use intra sentence attention",
)

parser.add_argument('--epoch', type=int, default=250, help='number of training epochs')
parser.add_argument('--gpus', type=int, default=0, help='number of gpus to train on. -1 for all gpus')
parser.add_argument('--val_interval', type=int, default=500, help='interval for checking the validation dataset')
parser.add_argument('--num_workers', type=int, default=0, help='number of workers in the dataloader')

parser.add_argument('--optimizer', type=str, default='adagrad', choices=['adam', 'adagrad'])
parser.add_argument('--lr', type=float, default=0.05, help='learning rate')
parser.add_argument('--max_grad_norm', type=float, default=5, help='if the norm of the gradient vector exceeds this renormalize it to have the norm equal to the max_grad_norm')
parser.add_argument('--param_init', type=float, default=0.01, help='parameter initialization gaussian')
parser.add_argument('--weight_decay', type=float, default=5e-5, help='l2 regularization')

parser.add_argument('--log_dir', type=str, default='logs/', help='log file directory')
parser.add_argument('--model_path', type=str, default='model', help='path of the model (w/o suffix)')
parser.add_argument(
    "--use_wandb", action="store_true", help="whether to use wandb for logging"
)

parser.add_argument('--test_dataset', type=str, default="test", help='whether to use wandb for logging')

args = parser.parse_args()
TRAIN_LBL_SIZE = 3

print(args)
datamodule = LitSNLI(
    train_file=args.train_file,
    val_file=args.val_file,
    test_file=args.test_file,
    max_length=args.max_length,
    n_workers=args.num_workers
)

word_vecs = w2v(args.w2v_file).word_vecs

encoder = Encoder(
    num_embeddings=word_vecs.size(0),
    embedding_size=args.embedding_size,
    hidden_size=args.hidden_size,
    param_init=args.param_init,
    intra_sent_atten=args.intra_sent_atten,
)

encoder.embedding.weight.data.copy_(word_vecs)
encoder.embedding.weight.requires_grad = False

atten = Atten(
    hidden_size=args.hidden_size,
    label_size=TRAIN_LBL_SIZE,
    param_init=args.param_init
)

model = LitModel.load_from_checkpoint(
    "saved_models_test/anlp-model-epoch=66-v1.ckpt",
    encoder=encoder,
    atten=atten,
    max_grad_norm=args.max_grad_norm,
    lr=args.lr,
    optim=args.optimizer,
    weight_decay=args.weight_decay,
)

trainer = pl.Trainer(
    max_epochs=args.epoch,
    gpus=args.gpus,
    accelerator='cpu' if args.gpus == 0 else 'ddp',
    progress_bar_refresh_rate=10,
)

trainer.test(
    model=model,
    dataloaders=datamodule,
    ckpt_path="anlp-model-epoch=87.ckpt",
    verbose=True,
)
