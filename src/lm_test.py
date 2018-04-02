import nltk
from languagemodel import NgramLM
from dataset import LMDataset


def text_parse(file):
	with open(file) as f:
		data = f.read()
		return nltk.sent_tokenize(data)

def text_test(file, n):
	print('***','Data:', file, 'n:',n,'***')
	data = text_parse(file)
	lm = NgramLM(data, n)
	lm.train()
	print(lm.generate())

def tunes_parse(file):
	lmd = LMDataset(file)
	return lmd.get_data()

def tunes_test(file, n):
	data = tunes_parse(file)
	for key, tunes in data.items():
		lm = NgramLM(tunes, n)
		lm.train()
		print(key, lm.generate())

# text_test('data/test/temp.txt', 6)
# text_test('data/test/tinyshakespeare.txt', 6)
tunes_test('abc_all.txt', 10)
