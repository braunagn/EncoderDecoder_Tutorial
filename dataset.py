import config
import torch
from torch.utils.data import Dataset


class LanguageDataset(Dataset):
    def __init__(self, X1, X2, y, pad_token_id=None):
        super(LanguageDataset).__init__()
        self.X1 = torch.tensor(X1, device=config.DEVICE)  # NL sequences
        self.X2 = torch.tensor(X2, device=config.DEVICE)  # EN sequences
        self.y = torch.tensor(y, dtype=torch.long, device=config.DEVICE)  # EN shifted
        self.pad_token_id = pad_token_id

    def __getitem__(self, index):
        # each sample returns: NL seq (X1), EN seq (X2), EN+1 seq (y) and pad masking for NL seq (x1pad)
        if self.pad_token_id is not None:
            x1pad = self.X1[index]==self.pad_token_id
            return self.X1[index], self.X2[index], self.y[index], x1pad[None,:]
        return self.X1[index], self.X2[index], self.y[index]

    def __len__(self):
        return len(self.X1)