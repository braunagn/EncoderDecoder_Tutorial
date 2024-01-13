import os
import time
from collections import OrderedDict

import config

import torch
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group


class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.OneCycleLR,
        loss_fn: torch.nn.CrossEntropyLoss,
        train_dl: torch.utils.data.DataLoader,
        val_dl: torch.utils.data.DataLoader,
        test_dl: torch.utils.data.DataLoader,
        gpu_id: int,
    ) -> None:
        print(f"GPU{gpu_id}: got here 1")
        self.gpu = torch.cuda.is_available()
        print(f"GPU{gpu_id}: got here 2")
        self.model = model
        print(f"GPU{gpu_id}: got here 3")
        if self.gpu:
            self.model = model.to(gpu_id)
            print(f"GPU{gpu_id}: got here 3a")
        print(f"GPU{gpu_id}: got here 4")
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_fn = loss_fn
        self.train_dl = train_dl
        self.val_dl = val_dl
        print(f"GPU{gpu_id}: got here 5")
        # self.val_iter = iter(self.val_dl)
        self.test_dl = test_dl
        # self.test_iter = iter(self.test_dl)
        print(f"GPU{gpu_id}: got here 6")
        self.gpu_id = gpu_id
        if config.LOADEXISTING_PATH is not None:
            self._load_checkpoint()
            if self.gpu:
                self.model = DDP(self.model, device_ids=[gpu_id])
            print(f"GPU{gpu_id}: got here 7")

    def _run_batch(
        self,
        x1,
        x2,
        y,
        x1padmask,
        train=False,
    ):
        if self.gpu:
            x1 = x1.to(self.gpu_id)
            x2 = x2.to(self.gpu_id)
            y = y.to(self.gpu_id)
            x1padmask = x1padmask.to(self.gpu_id)

        def _get_loss(x1, x2, y, x1padmask):
            logits = self.model(x1, x2, x1padmask)
            b, t, v = logits.shape
            logits = logits.view(b * t, v)  # reshape for loss calc
            y = y.view(b * t)  # reshape for loss calc
            return self.loss_fn(logits, y)

        if train:
            self.model.train()
            self.optimizer.zero_grad()
            loss = _get_loss(x1, x2, y, x1padmask)
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()
        else:
            self.model.eval()
            with torch.no_grad():
                loss = _get_loss(x1, x2, y, x1padmask)
            return loss

    def _run_epoch(self, epoch):
        if self.gpu:
            # print(f"GPU{self.gpu_id}: setting epoch...")
            self.train_dl.sampler.set_epoch(epoch)
            self.val_dl.sampler.set_epoch(epoch)
            self.test_dl.sampler.set_epoch(epoch)
        # train
        for batch_id, (x1, x2, y, x1padmask) in enumerate(self.train_dl):
            # print(f"GPU{self.gpu_id}: train batch size: {x1.shape}")
            _ = self._run_batch(x1, x2, y, x1padmask, train=True)

            if batch_id % (self.bpe // config.PRINT_TIMES_PER_EPOCH) == 0:
                # val
                for _, (x1, x2, y, x1padmask) in enumerate(self.val_dl):
                    val_loss = self._run_batch(x1, x2, y, x1padmask, train=False)
                    self.val_losses.append(val_loss.item())
                    break
                # test
                for _, (x1, x2, y, x1padmask) in enumerate(self.test_dl):
                    test_loss = self._run_batch(x1, x2, y, x1padmask, train=False)
                    self.test_losses.append(test_loss.item())
                    break

                # checkpoint save
                if self.gpu_id == 0:
                    self._save_checkpoint(epoch)

                # logging
                lr = self.scheduler.get_last_lr()[0]
                t = 0 # time.time() - self.times[-1]
                # self.times.append(time.time())

                print(
                    f"GPU{self.gpu_id} | epoch: {epoch+1}  |  batch_id: {batch_id}  |  val loss: {val_loss:.5f}  |  test loss: {test_loss:.5f}  |  lr: {lr:.3e} | runtime: {t//60//60 % 60:.0f}h {t//60 % 60:.0f}m {t % 60:.0f}s"
                )

    def _save_checkpoint(self, epoch):
        if self.gpu:
            ckp = self.model.module.state_dict()
        else:
            ckp = self.model.state_dict()
        checkpoint = {
            "MODEL": ckp,
            "EPOCH": epoch,
            "VAL_LOSS": self.val_losses,
            "TEST_LOSS": self.test_losses,
        }
        PATH = f"{config.SAVE_PATH_MODEL_OBJ}/{config.MODEL_OBJ_NAME}"
        torch.save(checkpoint, PATH)
        print(f"GPU{self.gpu_id} | epoch: {epoch+1}  |  Training checkpoint saved at: {PATH}")

    def _load_checkpoint(self):
        PATH = config.LOADEXISTING_PATH
        device = f"cuda:{self.gpu_id}" if self.gpu else "cpu"
        print(f"attempting to load checkpoint from: {PATH}")
        cp = torch.load(PATH, map_location=device)

        model_on_ddp = any("module." in k for k in cp["MODEL"].keys())
        if model_on_ddp:
            print("loading, got here 1")
            new_state_dict = OrderedDict()
            for k, v in cp["MODEL"].items():
                name = k[7:] # remove `module.`
                new_state_dict[name] = v
            print("loading, got here 1a")    
            self.model.load_state_dict(new_state_dict)
        else:
            print("loading, got here 1b")
            self.model.load_state_dict(cp["MODEL"])

        # if self.gpu:
        #     print("loading, got here 2")
        #     self.model = DDP(self.model, device_ids=[self.gpu_id])
        #     print("loading, got here 2a")
        # else:
        #     self.model.load_state_dict(cp["MODEL"])
        print(f"GPU{self.gpu_id}: model object loaded")

    def train(self, max_epochs: int):
        self.val_losses, self.test_losses = [], []
        self.bpe = len(self.train_dl)
        gpu_name = torch.cuda.get_device_name() if self.gpu else "CPU"
        dcnt = torch.cuda.device_count()

        print(f"GPU{self.gpu_id}: training with: {gpu_name} | device count: {dcnt} | epochs: {config.EPOCHS} | batch size: {config.BATCH_SIZE} | batches-per-epoch: {self.bpe}")

        for epoch in range(max_epochs):
            # self.times = [time.time()]
            self._run_epoch(epoch)