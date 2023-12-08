# tokenizer and model settings

import torch

####################
### model params ###
####################

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# architecture
C = 512
T = 30  # max context length; informed via quick analysis
N_LAYERS = 6
NUM_HEADS = 8
HEAD_SIZE = 64  # C // NUM_HEADS = 512 // 8

# training
DROPOUT = 0.1
BATCH_SIZE = 8
BATCH_SIZE_VAL = 50
EPOCS = 15
INITIAL_LR = 1e-7
MAX_LR = 1e-5
FINAL_LR = 1e-6
WARM_UP_STEPS = 1000


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
    # chars that appear very infrequently (~1-5 times) in the dataset.  Ignoring these sentence
    # all together given negligible impact on training and project focus is educational
    "°",
    "²",
    "½",
    "Á",
    "Ç",
    "É",
    "×",
    "ß",
    "à",
    "á",
    "â",
    "ã",
    "ä",
    "å",
    "ç",
    "è",
    "é",
    "ê",
    "ë",
    "ì",
    "í",
    "î",
    "ï",
    "ð",
    "ñ",
    "ó",
    "ô",
    "ö",
    "ú",
    "û",
    "ü",
    "ā",
    "ă",
    "Ĉ",
    "ĉ",
    "Č",
    "ĝ",
    "ĥ",
    "ī",
    "ı",
    "ĵ",
    "ł",
    "ō",
    "ŝ",
    "ş",
    "š",
    "ŭ",
    "ș",
    "ə",
    "ʻ",
    "π",
    "ḥ",
    "ṛ",
    "/",
    "…",
    "√",
    "🌡",
    "😷",
    "🤒",
    "🤧",
    "🤮",
    "🦠",
    "🧼",
    "«",
    "»",
    "Í",
    "Ö",
    "ć",
    "ń",
    "ŏ",
    "ū",
    "М",
    "Ч",
    "а",
    "з",
    "и",
    "к",
    "л",
    "о",
    "р",
    "с",
    "т",
    "ы",
    "э",
    "ׁ",
    "‐",
    "–",
    "—",
    "‘",
    "’",
    "“",
    "”",
    "₂",
    "€",
    "→",
    "あ",
    "@",
    "^",
    "+",
    '"',
    "&",
    "_",
    "{",
    "}",
    "(",
    ")",
    "[",
    "]",
    "#",
    "...",
]
