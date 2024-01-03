import time
import config
import tokenizer_prep
# import model
import utilities

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR

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
optimizer = AdamW(
    model.parameters(),
    lr=config.INITIAL_LR,
    betas=(0.9, 0.999),
    weight_decay=1e-2,
)

# learning schedule
steps_per_epoch = int(training_dl.dataset.__len__() / training_dl.batch_size)
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

# Training
val_loss = []
bpe = training_dl.dataset.__len__() // config.BATCH_SIZE
train_iter = iter(train_dl)  # for eval of loss over training set
test_iter = iter(test_dl)  # for eval of loss over testing set
gpu = torch.cuda.is_available()
gpu_name = torch.cuda.get_device_name() if gpu else "CPU"

print(
    f"Training with: GPU: {gpu}({gpu_name}) | epochs: {config.EPOCHS} | batch size: {config.BATCH_SIZE} | batches-per-epoch: {bpe}"
)

for e in range(config.EPOCHS):
    times = [time.time()]
    for batch_id, (
        x1_train_batch,
        x2_train_batch,
        y_train_batch,
        x1padmask_batch,
    ) in enumerate(training_dl):
        model.train()
        logits = model(x1_train_batch, x2_train_batch, x1padmask_batch)
        (
            b,
            t,
            v,
        ) = logits.shape  # needed for last batch in epoch that may not be of BATCH_SIZE
        logits = logits.view(b * t, v)  # reshape for loss calc
        y_train_batch = y_train_batch.view(b * t)  # reshape for loss calc
        training_loss = loss_fn(logits, y_train_batch)

        optimizer.zero_grad()
        training_loss.backward()
        optimizer.step()
        scheduler.step()

        # print training updates as desired frequency
        if batch_id % (bpe // config.PRINT_TIMES_PER_EPOCH) == 0:
            train_iter = iter(train_dl)
            x1_train, x2_train, y_train, x1padmask_train = next(train_iter)
            logits = model(x1_train, x2_train, x1padmask_train)
            logits = logits.view(config.BATCH_SIZE_EVAL * config.T, len(vocab))
            y_train = y_train.view(config.BATCH_SIZE_EVAL * config.T)
            train_loss = loss_fn(logits, y_train)

            test_iter = iter(test_dl)
            x1_test, x2_test, y_test, x1padmask_test = next(test_iter)
            logits = model(x1_test, x2_test, x1padmask_test)
            logits = logits.view(config.BATCH_SIZE_EVAL * config.T, len(vocab))
            y_test = y_test.view(config.BATCH_SIZE_EVAL * config.T)
            test_loss = loss_fn(logits, y_test)

            if config.SAVE_PATH_MODEL_OBJ is not None:
                # given a small model size, it's a good idea to save frequently in the event training timesouts (which
                # is often the case on free services like google colab).  If this occurs, simply reload the model
                # and start the training again (see config.LOAD_PATH_TRAINED_MODEL_OBJ).
                torch.save(model, config.SAVE_PATH_MODEL_OBJ)

            lr = scheduler.get_last_lr()[0]
            t = time.time() - times[-1]
            times.append(time.time())

            print(
                f"epoch: {e+1}  |  batch_id: {batch_id}  |  train loss: {train_loss:.7f}  |  test loss: {test_loss:.7f}  |  lr: {lr:.3e}  |  runtime: {t//60//60 % 60:.0f}h {t//60 % 60:.0f}m {t % 60:.0f}s"
            )
