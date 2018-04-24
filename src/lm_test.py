import nltk, re
from languagemodel import NgramLM
from dataset import LMDataset
import numpy as np
import matplotlib.pyplot as plt


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
	lmd = LMDataset(file, byKey=True)
	for key, tunes in lmd.key_date.items():
		lm = NgramLM(tunes, n)
		lm.train()
		print(key, lm.generate())

def k_experiment(file, n):
	output = ''

	lmd = LMDataset(file)
	split = int(len(lmd.raw_data)*0.8)
	train, test = lmd.raw_data[:split], lmd.raw_data[split:]

	X = []
	Y = []

	for k in np.arange(0.1, 1.1, step=0.1):
		lm = NgramLM(data=train, n=n)
		lm.train()
		tune = lm.generate()
		acc, pp = lm.test(data=test, s='add-k', k=k)
		X.append(k)
		Y.append(pp)
		op = 'n: {}\nk: {}\n{}\nAccuracy: {}\nPerplexity: {}\n\n'.format(
			str(n), str(k), tune, str(acc), str(pp))
		print(op)
		output += op

	with open('data/lm_k_exp', 'w') as f:
		f.write(output)

	plt.plot(X, Y, '-')
	plt.xlabel('k of add-k smoothing')
	plt.ylabel('Perplexity')
	plt.show()

def n_experiment(file, smoothing, k):
	output = ''

	lmd = LMDataset(file)
	split = int(len(lmd.raw_data)*0.8)
	train, test = lmd.raw_data[:split], lmd.raw_data[split:]

	X = []
	Y = []
	Y_acc = []

	for n in range(2, 21):
		lm = NgramLM(data=train, n=n)
		lm.train()
		tune = lm.generate()
		acc, pp = lm.test(data=test, s=smoothing, k=k)
		X.append(n)
		Y.append(pp)
		Y_acc.append(acc)
		op = 'n: {}\nk: {}\n{}\nAccuracy: {}\nPerplexity: {}\n\n'.format(
			str(n), str(k), tune, str(acc), str(pp))
		print(op)
		output += op

	with open('data/lm_n_exp_t', 'w') as f:
		f.write(output)

	plt.plot(X, Y, '-')
	plt.xlabel('n of n-gram model')
	plt.ylabel('Perplexity')
	plt.savefig('data/lm_n_exp_pp_t')
	plt.clf()
	plt.plot(X, Y_acc, '-')
	plt.xlabel('n of n-gram model')
	plt.ylabel('Accuracy')
	plt.savefig('data/lm_n_exp_acc_t')


def plott_k():
	with open('data/lm_k_exp', 'r') as f:
		raw_data = f.read()
		k = re.findall(r'\nk: (.*)\n', raw_data)
		pp = re.findall(r'\nPerplexity: (.*)\n\n', raw_data)
		plt.plot(k, pp, 'o-')
		plt.xlabel('k of add-k smoothing')
		plt.ylabel('Perplexity')
		plt.savefig('data/lm_n_exp_k')
		plt.show()


def plott():
	with open('data/lm_n_exp_t', 'r') as f:
		raw_data = f.read()
		n = re.findall(r'n: (.*)\n', raw_data)
		k = re.findall(r'\nk: (.*)\n', raw_data)
		acc = re.findall(r'\nAccuracy: (.*)\n', raw_data)
		pp = re.findall(r'\nPerplexity: (.*)\n\n', raw_data)
		plt.plot(n, pp, 'o-')
		plt.xlabel('n of n-gram model')
		plt.ylabel('Perplexity')
		plt.show()
		plt.savefig('data/lm_n_exp_pp_t')
		plt.clf()
		plt.plot(n, acc, 'o-')
		plt.xlabel('n of n-gram model')
		plt.ylabel('Accuracy')
		plt.show()
		plt.savefig('data/lm_n_exp_acc_t')


# text_test('data/test/temp.txt', 6)
# text_test('data/test/tinyshakespeare.txt', 6)
# k_experiment('data/abc_all.txt', 10)
# n_experiment('data/abc_all.txt', 'add-k', 0.1)
# plott()