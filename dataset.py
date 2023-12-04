from . import config
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader


class LanguageDataset(Dataset):
    def __init__(self, X1, X2, y, pad_token_id=None):
        super(LanguageDataset).__init__()
        self.X1 = torch.tensor(X1, dtype=torch.int32, device=config.DEVICE)
        self.X2 = torch.tensor(X2, dtype=torch.int32, device=config.DEVICE)
        self.y = torch.tensor(y, dtype=torch.float32, device=config.DEVICE)  # float to compare w/model output
        self.pad_token_id = pad_token_id

    def __getitem__(self, index):
        if self.pad_token_id is not None:
            x1pad = self.X1[index]==pad_token_id
            return self.X1[index], self.X2[index], self.y[index], x1pad[None,:]
        return self.X1[index], self.X2[index], self.y[index]

    def __len__(self):
        return len(self.X1)


X1_train, X1_test, X2_train, X2_test, y_train, y_test = train_test_split(
    X1,
    X2,
    y,
    test_size=0.15,
    shuffle=False, # already shuffled and grouped
)
train_data = LanguageDataset(X1_train, X2_train, y_train, pad_token_id=pad_token_id)
test_data = LanguageDataset(X1_test, X2_test, y_test, pad_token_id=pad_token_id)

training_dl = DataLoader(
    train_data,
    batch_size=config.BATCH_SIZE,
    shuffle=False,  # keep sequences of the same length together
    drop_last=False,
)

# for loss performance over train/test datasets (vs. batch being trained on)
train_dl = DataLoader(
    train_data,
    batch_size=config.BATCH_SIZE_VAL,
    shuffle=True,   # sample across the dataset, regardless of sequence len
    drop_last=False,
)

test_dl = DataLoader(
    test_data,
    batch_size=config.BATCH_SIZE_VAL,
    shuffle=True,
    drop_last=False,
)