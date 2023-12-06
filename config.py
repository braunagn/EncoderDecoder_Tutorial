# tokenizer and model settings

import torch

####################
### model params ###
####################

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
T = 30  # max context length; informed via quick analysis
BATCH_SIZE = 8
BATCH_SIZE_VAL = 50


###########################
### vocab and tokenizer ###
###########################

SPECIAL_TOKENS = {
    "BOS_TOKEN": "<s>",
    "EOS_TOKEN": "</s>",
    "PAD_TOKEN": "[PAD]",
    "UNK_TOKEN": "[UNK]",
}
VOCAB_SIZE = 30000
IGNORE = [
    # chars that appear very infrequently (1-5 times) in the dataset.  Ignoring these sentence
    # all together given negligible impact on training and project focus is educational
    '°', '²', '½', 'Á', 'Ç', 'É', '×', 'ß', 'à', 'á', 'â', 'ã', 'ä', 'å', 'ç', 'è', 'é', 'ê', 'ë', 'ì', 'í', 'î', 'ï',
    'ð', 'ñ', 'ó', 'ô', 'ö', 'ú', 'û', 'ü', 'ā', 'ă', 'Ĉ', 'ĉ', 'Č', 'ĝ', 'ĥ', 'ī', 'ı', 'ĵ', 'ł', 'ō', 'ŝ', 'ş', 'š',
    'ŭ', 'ș',  'ə', 'ʻ', 'π', 'ḥ', 'ṛ',  '/', '…', '√', '🌡', '😷', '🤒', '🤧', '🤮', '🦠',  '🧼', '«', '»', 'Í',
    'Ö', 'ć', 'ń', 'ŏ', 'ū', 'М', 'Ч', 'а', 'з', 'и', 'к', 'л', 'о', 'р', 'с', 'т', 'ы', 'э', 'ׁ', '‐', '–', '—', '‘',
    '’', '“', '”', '₂', '€', '→', 'あ', '@', '^', '+', '"', '&', '_', '{', '}', '(', ')', '[', ']', '#', "...",
  ]