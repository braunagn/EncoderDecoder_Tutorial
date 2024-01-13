import sys
import path
import os

import torch
import torch.multiprocessing as mp
from torch.distributed import init_process_group, destroy_process_group
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR

# add parent dir to path for below imports to work
directory = path.Path(__file__).abspath()
sys.path.append(directory.parent.parent)

import config
import model as MODEL
import utilities
import tokenizer_prep
from train import Trainer


def load_train_objs(rank=None):
    data = utilities.read_in_rawdata_and_cleanup()
    tokenizer = tokenizer_prep.train_tokenizer(data)
    vocab = tokenizer.get_vocab()
    pad_token_id = tokenizer.token_to_id(config.SPECIAL_TOKENS["PAD_TOKEN"])
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=pad_token_id, label_smoothing=0.1)
    model = MODEL.LanguageModel(vocab_len=len(vocab))
    optimizer = AdamW(model.parameters(), lr=config.INITIAL_LR)
    train_data, test_data = utilities.make_train_test_datasets(tokenizer, data)
    print(f"train length: {train_data.__len__()}")
    print(f"test length: {test_data.__len__()}")
    train_dl, val_dl, test_dl = prep_dataloaders(train_data, test_data)
    steps_per_epoch = len(train_dl)
    scheduler = OneCycleLR(
        optimizer,
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
    print(f"GPU{rank}: all training objects initialized")
    return (
        model,
        optimizer,
        scheduler,
        tokenizer,
        loss_fn,
        train_data,
        test_data,
        train_dl,
        val_dl,
        test_dl,
    )


def prep_dataloaders(train_data, test_data):
    print("preparing dataloaders...")
    gpu = torch.cuda.is_available()
    train_dl = DataLoader(
        train_data,
        batch_size=config.BATCH_SIZE,
        shuffle=False,  # keep sequences of the same length together
        drop_last=False,
        pin_memory=False,
        sampler=DistributedSampler(train_data, shuffle=False) if gpu else None,
    )
    val_dl = DataLoader(
        train_data,
        batch_size=config.BATCH_SIZE_EVAL,
        shuffle=False,
        drop_last=False,
        pin_memory=False,
        sampler=DistributedSampler(train_data, shuffle=True) if gpu else None,
    )
    test_dl = DataLoader(
        test_data,
        batch_size=config.BATCH_SIZE_EVAL,
        shuffle=False,
        drop_last=False,
        pin_memory=False,
        sampler=DistributedSampler(test_data, shuffle=True) if gpu else None,
    )
    print("dataloaders setup")
    return train_dl, val_dl, test_dl


# add parent dir to path for below imports to work
directory = path.Path(__file__).abspath()
sys.path.append(directory.parent.parent)


def ddp_setup(rank, world_size):
    os.environ["MASTER_ADDR"] = config.MASTER_ADDR
    os.environ["MASTER_PORT"] = config.MASTER_PORT
    init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


# def main(rank: int, world_size: int, save_every: int, total_epochs: int, batch_size: int):
def main(rank: int, world_size: int):
    if torch.cuda.is_available():
        print(f"GPU{rank}: setting up GPU process group...")
        ddp_setup(rank, world_size)
    else:
        print("GPU not available, executing on CPU...")
    (
        model,
        optimizer,
        scheduler,
        _,
        loss_fn,
        train_data,
        test_data,
        train_dl,
        val_dl,
        test_dl,
    ) = load_train_objs(rank)
    print("initializing Trainer...")
    trainer = Trainer(
        model, optimizer, scheduler, loss_fn, train_dl, val_dl, test_dl, rank
    )
    print("commencing with training...")
    trainer.train(config.EPOCHS)
    destroy_process_group()


if __name__ == "__main__":
    import argparse

    # parser = argparse.ArgumentParser()
    # parser.add_argument("--master_port", type=str)
    # args = parser.parse_args()

    world_size = torch.cuda.device_count()
    print(f"world_size: {world_size}")
    if world_size == 0:
        main(rank=0, world_size=world_size)
    else:
        mp.spawn(main, args=(world_size,), nprocs=world_size)
    # import argparse
    # parser = argparse.ArgumentParser(description='simple distributed training job')
    # parser.add_argument('total_epochs', type=int, help='Total epochs to train the model')
    # parser.add_argument('save_every', type=int, help='How often to save a snapshot')
    # parser.add_argument('--batch_size', default=32, type=int, help='Input batch size on each device (default: 32)')
    # args = parser.parse_args()

    # world_size = torch.cuda.device_count()
    # mp.spawn(main, args=(world_size, args.save_every, args.total_epochs, args.batch_size), nprocs=world_size)
