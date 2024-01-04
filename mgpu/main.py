import time
import path
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
import lightning as L

L.seed_everything(42, workers=True)

# add parent dir to path for below imports to work
directory = path.Path(__file__).abspath()
sys.path.append(directory.parent.parent)

import config
import tokenizer_prep
import utilities


data = utilities.read_in_rawdata_and_cleanup()
tokenizer = tokenizer_prep.train_tokenizer(data)
vocab = tokenizer.get_vocab()
training_dl, train_dl, test_dl = utilities.make_train_test_datasets(tokenizer, data)
model = utilities.load_model(tokenizer=None)

pad_token_id = tokenizer.token_to_id(config.SPECIAL_TOKENS["PAD_TOKEN"])
# loss function
loss_fn = nn.CrossEntropyLoss(
    ignore_index=pad_token_id, label_smoothing=0.1
)  # don't compute loss for pad tokens

# optimizer
OPTIMIZER = AdamW(
    model.parameters(),
    lr=config.INITIAL_LR,
    betas=(0.9, 0.999),
    weight_decay=1e-2,
)

# learning schedule
steps_per_epoch = int(training_dl.dataset.__len__() / training_dl.batch_size)
SCHEDULER = OneCycleLR(
    OPTIMIZER,
    max_lr=config.MAX_LR,
    epochs=config.EPOCHS,
    steps_per_epoch=steps_per_epoch,
    pct_start=config.WARMUP_STEPS / (config.EPOCHS * steps_per_epoch),
    anneal_strategy="cos",
    cycle_momentum=True,
    base_momentum=0.7,
    max_momentum=0.99,  # results in ZeroDivisionError if 1.0
    div_factor=config.MAX_LR / config.INITIAL_LR,
    final_div_factor=config.INITIAL_LR / config.FINAL_LR,
)


class LightningWrapper(L.LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def training_step(self, batch, batch_idx):
        x1, x2, y, x1pad = batch[0]  # lightning wraps batch in a list
        logits = self.model(x1, x2, x1pad)
        (
            b,
            t,
            v,
        ) = logits.shape  # needed for last batch in epoch that may not be of BATCH_SIZE
        logits = logits.view(b * t, v)  # reshape for loss calc
        y = y.view(b * t)  # reshape for loss calc
        loss = loss_fn(logits, y)
        self.log(
            "train_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=False,
            sync_dist=True,  # must be True for multi-GPU training, and docs recommend logger=False
        )
        return loss

    def validation_step(self, batch, batch_idx):
        x1, x2, y, x1pad = batch
        logits = self.model(x1, x2, x1pad)
        (
            b,
            t,
            v,
        ) = logits.shape  # needed for last batch in epoch that may not be of BATCH_SIZE
        logits = logits.view(b * t, v)  # reshape for loss calc
        y = y.view(b * t)  # reshape for loss calc
        loss = loss_fn(logits, y)
        self.log(
            "val_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=False,
            sync_dist=True,  # must be True for multi-GPU training, and docs recommend logger=False
        )
        return loss

    def configure_optimizers(self):
        optimizer = OPTIMIZER
        scheduler = SCHEDULER
        scheduler_dict = {
            "scheduler": scheduler,
            "interval": "step",  # step the scheduler after each batch
        }

        return [optimizer], [scheduler_dict]


def main():
    Lmodel = LightningWrapper(model)
    trainer = L.Trainer(
        deterministic="warn",  # random seed
        accelerator="auto",  # use gpu if available, else cpu
        num_nodes=1,  # count of servers (not the same as count of GPUs ON each server)
        devices=1,  # use all GPUs; equivalent to list(range(torch.cuda.device_count()))
        strategy="ddp",  # "Distributed Data Parallel" (lots of other options exist, see URL above)
        max_epochs=config.EPOCHS,
        val_check_interval=1.0 / config.PRINT_TIMES_PER_EPOCH,
        log_every_n_steps=steps_per_epoch,
    )
    trainer.fit(
        model=Lmodel,
        train_dataloaders=[training_dl],
        val_dataloaders=[test_dl],  # train_dl
    )

if __name__ == "__main__":
    main()
