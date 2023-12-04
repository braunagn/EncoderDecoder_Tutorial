from . import config
from collections import Counter

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import BertPreTokenizer, ByteLevel


def train_tokenizer(data):
    # data: pd.Dataframe with sentence pairs (2x columns max)
    model = BPE(
        unk_token=config.SPECIAL_TOKENS["UNK_TOKEN"],
        fuse_unk=False,
    )
    tokenizer = Tokenizer(model=model)
    tokenizer.pre_tokenizer = ByteLevel()
    tokenizer.enable_padding(
        direction="right",
        length=config.T,  # length of entire sequence, NOT amount of padding
        pad_token=config.SPECIAL_TOKENS["PAD_TOKEN"],
        pad_id=list(config.SPECIAL_TOKENS.keys()).index("PAD_TOKEN"),
    )
    tokenizer.enable_truncation(
        max_length=T,
        direction="right",
    )
    trainer = BpeTrainer(
        vocab_size=config.VOCAB_SIZE + len(config.SPECIAL_TOKENS.items()),
        min_frequency=2,
        special_tokens=config.SPECIAL_TOKENS.items(),
        continuing_subword_prefix="##",
        end_of_word_suffix="ġ",
        max_token_length=None,
    )
    tokenizer.train_from_iterator(data.values, trainer=trainer)
    print("tokenizer training complete")
    return tokenizer


def encode_sentences(data):
    # data: Pd.Series with one language set of sentences 
    return tokenizer.encode_batch(data.values)


def sequence_length(s):
    # determine sequence length (only counts ids not in `ignore`)
    ignore = [
        config.SPECIAL_TOKENS["BOS_TOKEN"],
        config.SPECIAL_TOKENS["EOS_TOKEN"],
        config.SPECIAL_TOKENS["PAD_TOKEN"],
    ]
    return sum([v for k,v in Counter(s).items() if k not in ignore])


def group_sentences(x1, x2):
    """ to assist with training, group sentences based upon length """
    # x1: language 1 from tokenizer.encode_batch(); to be translated from
    # x2: language 2 from tokenizer.encode_batch(); to be translated into

    # shuffle data and group sequences together based upon length
    x1_len = [sequence_length(x1.ids) for x in x1]
    groups = reversed([tmp for _, tmp in pd.DataFrame(x1_len, columns=["x1_len"]).sample(frac=1.).groupby("x1_len")])
    grouped_index = pd.concat(groups).index
    sorted_data = [(x1_encoded[i].ids, x2_encoded[i].ids) for i in grouped_index]
    return sorted_data