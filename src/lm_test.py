import nltk
from languagemodel import NgramLM

def text_parseData(file):
	with open(file) as f:
		data = f.read()
		return nltk.sent_tokenize(data)

def text_test(file, n):
	print('***','Data:', file, 'n:',n,'***')
	data = text_parseData(file)
	lm = NgramLM(data, n)
	lm.train()
	print(lm.generate())

text_test('data/test/temp.txt', 6)
text_test('data/test/tinyshakespeare.txt', 6)
