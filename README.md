# ANLP Project

This is the repo for ANLP Project where we aim to replicate and extend Decomposable Attention Model.


### Arguments

```
usage: train.py [-h] [--train_file TRAIN_FILE] [--val_file VAL_FILE]
                [--test_file TEST_FILE] [--max_length MAX_LENGTH]
                [--w2v_file W2V_FILE] [--embedding_size EMBEDDING_SIZE]
                [--hidden_size HIDDEN_SIZE] [--intra_sent_atten]
                [--epoch EPOCH] [--gpus GPUS] [--val_interval VAL_INTERVAL]
                [--num_workers NUM_WORKERS] [--optimizer {adam,adagrad}]
                [--lr LR] [--max_grad_norm MAX_GRAD_NORM]
                [--param_init PARAM_INIT] [--weight_decay WEIGHT_DECAY]
                [--log_dir LOG_DIR] [--model_path MODEL_PATH] [--use_wandb]

optional arguments:
  -h, --help            show this help message and exit
  --train_file TRAIN_FILE
                        training data file (hdf5)
  --val_file VAL_FILE   validation data file (hdf5)
  --test_file TEST_FILE
                        test data file (hdf5)
  --max_length MAX_LENGTH
                        maximum length of training sentences. -1 means no max
                        length
  --w2v_file W2V_FILE   pretrained word vectors file (hdf5)
  --embedding_size EMBEDDING_SIZE
                        word embedding size
  --hidden_size HIDDEN_SIZE
                        hidden layer size
  --intra_sent_atten    whether to use intra sentence attention
  --epoch EPOCH         number of training epochs
  --gpus GPUS           number of gpus to train on. -1 for all gpus
  --val_interval VAL_INTERVAL
                        interval for checking the validation dataset
  --num_workers NUM_WORKERS
                        number of workers in the dataloader
  --optimizer {adam,adagrad}
  --lr LR               learning rate
  --max_grad_norm MAX_GRAD_NORM
                        if the norm of the gradient vector exceeds this
                        renormalize it to have the norm equal to the
                        max_grad_norm
  --param_init PARAM_INIT
                        parameter initialization gaussian
  --weight_decay WEIGHT_DECAY
                        l2 regularization
  --log_dir LOG_DIR     log file directory
  --model_path MODEL_PATH
                        path of the model (w/o suffix)
  --use_wandb           whether to use wandb for logging
```