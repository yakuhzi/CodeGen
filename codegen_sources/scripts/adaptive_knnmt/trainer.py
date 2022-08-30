import os
import torch

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.utilities.seed import seed_everything

from .arguments import parse_arguments
from .data_module import DataModule
from .meta_k import MetaK

arguments = parse_arguments()

seed_everything(2022, workers=True)
torch.use_deterministic_algorithms(True)

gpus = 1 # torch.cuda.device_count()
learning_rate = arguments.learning_rate
strategy = None

if torch.cuda.is_available():
    learning_rate /= gpus
    # strategy = 'ddp'

data_module = DataModule(
    batch_size=arguments.batch_size, 
    samples=arguments.samples, 
    dataset_dir=arguments.dataset_dir, 
    model_dir=arguments.model_dir, 
    cache_dir=arguments.cache_dir,
    bpe_path=arguments.bpe_path, 
    language_pair=arguments.language_pair
)

model = MetaK(
    batch_size=arguments.batch_size,
    learning_rate=learning_rate,
    max_k=arguments.max_k, 
    hidden_size=arguments.hidden_size,
    knn_temperature=arguments.knn_temperature,
    tc_temperature=arguments.tc_temperature,
    vocab_size=arguments.vocab_size,
    language_pair=arguments.language_pair,
    adam_betas=arguments.adam_betas,
    knnmt_dir=arguments.knnmt_dir,
)

knnmt_mode = arguments.knnmt_dir.split("/")[-1]
betas = arguments.adam_betas.replace(", ", "-")
configuration = f"S{arguments.samples}_KT{arguments.knn_temperature}_TT{arguments.tc_temperature}_K{arguments.max_k}_H{arguments.hidden_size}_L{arguments.learning_rate}_B{betas}"

print("Configuration: ", configuration)

log_dir = os.path.join(arguments.log_dir, knnmt_mode, arguments.language_pair)
logger = TensorBoardLogger(save_dir=log_dir, name="", version=configuration)

checkpoint_dir = os.path.join(arguments.checkpoint_dir, knnmt_mode, arguments.language_pair, configuration)
latest_checkpoint_callback = ModelCheckpoint(dirpath=checkpoint_dir, every_n_epochs=5, save_top_k=-1)
best_checkpoint_callback = ModelCheckpoint(dirpath=checkpoint_dir, filename="best-{epoch}", monitor="val_loss", mode="min")

progress_bar_callback = TQDMProgressBar(refresh_rate=10)

trainer = Trainer(
    gpus=gpus,
    strategy=strategy,
    min_epochs=arguments.epochs,
    max_epochs=arguments.epochs,
    logger=logger,
    callbacks=[latest_checkpoint_callback, best_checkpoint_callback, progress_bar_callback]
)

trainer.fit(model, data_module, ckpt_path=arguments.checkpoint_path)
trainer.test(model, data_module)
