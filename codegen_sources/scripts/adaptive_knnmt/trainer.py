import os
import torch
import random

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.utilities.seed import seed_everything

from .arguments import parse_arguments
from .data_module import DataModule
from .meta_k import MetaK

version = str(random.randint(100000, 1000000))
arguments = parse_arguments()

seed_everything(2022, workers=True)
torch.use_deterministic_algorithms(True)

betas = arguments.adam_betas.replace(", ", "-")
configuration = f"BS{arguments.batch_size}_KT{arguments.knn_temperature}_TT{arguments.tc_temperature}_MK{arguments.max_k}_TK{arguments.tc_k}_HS{arguments.hidden_size}_LR{arguments.learning_rate}_B{betas}"

print("Language pair:", arguments.language_pair)
print("Configuration:", configuration)

log_dir = os.path.join(arguments.log_dir, arguments.language_pair, configuration)
logger = TensorBoardLogger(save_dir=log_dir, name="", version=version)

checkpoint_dir = os.path.join(arguments.checkpoint_dir, arguments.language_pair, configuration, version)
latest_checkpoint_callback = ModelCheckpoint(dirpath=checkpoint_dir, every_n_epochs=10, save_top_k=-1)
best_checkpoint_callback = ModelCheckpoint(dirpath=checkpoint_dir, filename="best-{epoch}", monitor="val_loss", mode="min")

progress_bar_callback = TQDMProgressBar(refresh_rate=10)

data_module = DataModule(
    batch_size=arguments.batch_size, 
    dataset_dir=arguments.dataset_dir, 
    model_dir=arguments.model_dir, 
    cache_dir=arguments.cache_dir,
    bpe_path=arguments.bpe_path, 
    language_pair=arguments.language_pair
)

model = MetaK(
    batch_size=arguments.batch_size,
    learning_rate=arguments.learning_rate,
    max_k=arguments.max_k, 
    tc_k=arguments.tc_k, 
    hidden_size=arguments.hidden_size,
    knn_temperature=arguments.knn_temperature,
    tc_temperature=arguments.tc_temperature,
    vocab_size=arguments.vocab_size,
    language_pair=arguments.language_pair,
    adam_betas=arguments.adam_betas,
    knnmt_dir=arguments.knnmt_dir,
)

trainer = Trainer(
    gpus=1,
    strategy=None,
    min_epochs=arguments.epochs,
    max_epochs=arguments.epochs,
    logger=logger,
    callbacks=[latest_checkpoint_callback, best_checkpoint_callback, progress_bar_callback]
)

trainer.fit(model, data_module, ckpt_path=arguments.checkpoint_path)
trainer.test(model, data_module)
