# tokenizer and model settings

import torch

####################
### directories ####
####################
REPO_DIR = "C:/Users/braun/OneDrive/Desktop/NL2EN"
SAVE_PATH_MODEL_OBJ = f"{REPO_DIR}/model_object"  # set to None for no checkpoints
# to continue training a model that has already been initialized:
LOAD_PATH_TRAINED_MODEL_OBJ = "C:/Users/braun/OneDrive/Desktop/semi_trained_model_object" # set to None if not loading from storage


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
TEST_SPLIT = 0.15  # % of data that is test
DROPOUT = 0.1
BATCH_SIZE = 8
BATCH_SIZE_EVAL = 50 # num batches for eval of train and test loss
EPOCHS = 20
INITIAL_LR = 1e-7
MAX_LR = 1e-5
FINAL_LR = 1e-6
WARMUP_STEPS = 5000
PRINT_TIMES_PER_EPOCH = 50


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
