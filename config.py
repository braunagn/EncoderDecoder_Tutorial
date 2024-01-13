# tokenizer and model settings
import os
import torch

####################
### directories ####
####################
REPO_DIR = "C:/Users/braun/OneDrive/Desktop/NL2EN"
if not os.path.exists(REPO_DIR):
    REPO_DIR = "~/NL2EN"

SAVE_PATH_MODEL_OBJ = "C:/Users/braun/OneDrive/Desktop"
if not os.path.exists(SAVE_PATH_MODEL_OBJ):
    SAVE_PATH_MODEL_OBJ = "."
MODEL_OBJ_NAME = "model_object.cp"
# to continue training an existing model checkpoint:
LOADEXISTING_NAME = "model_object_semi_trained.cp"
# LOADEXISTING_NAME = "model_object.cp"
# LOADEXISTING_PATH = None
LOADEXISTING_PATH = f"{SAVE_PATH_MODEL_OBJ}/{LOADEXISTING_NAME}"

####################
### model params ###
####################

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
MASTER_ADDR = "localhost"
MASTER_PORT = "12355"

# architecture
C = 512
T = 30  # max context length; informed via quick analysis
N_LAYERS = 6
NUM_HEADS = 8
HEAD_SIZE = 64  # C // NUM_HEADS = 512 // 8

# training
TEST_SPLIT = 0.15  # % of data that is test
DROPOUT = 0.1
BATCH_SIZE = 1
BATCH_SIZE_EVAL = 48 # num batches for eval of train and test loss
EPOCHS = 100
INITIAL_LR = 1e-7
MAX_LR = 1e-5
FINAL_LR = 1e-6
WARMUP_STEPS = 2500
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
