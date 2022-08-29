import os
import torch

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
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
    learning_rate=learning_rate,
    max_k=arguments.max_k, 
    hidden_size=arguments.hidden_size,
    temperature=arguments.temperature,
    vocab_size=arguments.vocab_size,
    language_pair=arguments.language_pair,
    adam_betas=arguments.adam_betas,
    knnmt_dir=arguments.knnmt_dir,
)

log_dir = os.path.join(arguments.log_dir, arguments.language_pair)
logger = TensorBoardLogger(save_dir=log_dir, name="logs")

checkpoint_dir = os.path.join(arguments.checkpoint_dir, arguments.language_pair)
checkpoint_callback = ModelCheckpoint(dirpath=checkpoint_dir, every_n_epochs=10, save_top_k=-1)

trainer = Trainer(
    gpus=gpus,
    strategy=strategy,
    min_epochs=arguments.epochs,
    max_epochs=arguments.epochs,
    progress_bar_refresh_rate=10,
    logger=logger,
    callbacks=[checkpoint_callback]
)

trainer.fit(model, data_module, ckpt_path=arguments.checkpoint_path)
trainer.test(model, data_module)
