import config
import pandas as pd

""" simple data cleanup prior to tokenization """


def ignore(sentence):
	""" ignore sentences with very infrequent character occurence """
	for c in sentence:
		if c in config.IGNORE:
			return True
	return False

def replace_chars(sentence):
	char_replace = {
		"\xa0": "",
		"\u200b": "",
	}
	for k, v in char_replace.items():
		sentence = sentence.replace(k,v)
	return sentence

def initial_cleanup(df):
	df = df[df.isnull().any(axis=1)==False]  # remove 1x null entry
	data = df[~(df.applymap(ignore).any(axis=1))]
	data = data[~(data.applymap(lambda x: "..." in x).any(axis=1))]
	data = data.applymap(replace_chars)
	data = data.applymap(lambda x: x.strip())
	# add begin/end of sentence characters for tokenization
	data = data.apply(lambda x: config.SPECIAL_TOKENS["BOS_TOKEN"] + x + config.SPECIAL_TOKENS["EOS_TOKEN"])
	return data.reset_index(drop=True)