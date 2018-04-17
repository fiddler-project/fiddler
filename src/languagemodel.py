from collections import Counter
import math, re
import nltk
import numpy as np
import math
import operator


class NgramLM(object):

	def __init__(self, data, n):
		"""
		`data` is a list of tunes
		`n` is the order of n-gram
		"""
		self.n = n
		self.tokens = Counter()
		self.ng_counts = Counter()
		self.ctxt_counts = Counter()
		self.prob = {}
		for tune in data:
			self._addTune(tune, self.tokens, self.ng_counts, self.ctxt_counts)

	def _addTune(self, tune, tokens, ng_counts, ctxt_counts):
		tune = ['$',] * (self.n-1) + list(tune) + ['#']
		ngrams = zip(*[tune[i:] for i in range(self.n)])
		for grams in ngrams:
			ctxt = grams[:-1]
			token = grams[-1]
			tokens[token] += 1
			ng_counts[(ctxt, token)] += 1
			ctxt_counts[ctxt] += 1

	def train(self):
		for (ctxt, token), ng_count in self.ng_counts.items():
			ctxt_str = ''.join(ctxt)
			p = float(ng_count) / self.ctxt_counts[ctxt]
			if ctxt_str in self.prob:
				self.prob[ctxt_str].update({token: p})
			else:
				self.prob[ctxt_str] = {token: p}

	def _pick_next(self, seed):
		tokens = list(self.prob[seed].keys())
		probs = list(self.prob[seed].values())
		return np.random.choice(tokens, 1, p=probs)[0]

	def _pick_next_best(self, seed):
		return max(self.prob[seed].iteritems(), key=operator.itemgetter(1))[0]

	def _base_tune(self, ctxt):
		if ctxt is None:
			seed = ''.join('$',) * (self.n-1)
			tune = seed + self._pick_next(seed)
		else:
			tune = ''.join(('$',) * (self.n-1) + tuple(ctxt))
		return tune

	def _clean(self, tune):
		return re.sub(r'[$#]', '', tune)

	def generate(self, ctxt=None):
		""" Generates a tune based on trained n-gram LM """
		tune = self._base_tune(ctxt)
		seed = tune[-(self.n - 1):]
		c = ''
		while(c != '#'):
			c = self._pick_next(seed)
			tune += c
			seed = tune[-(self.n - 1):]
		return self._clean(tune)

	def _accuracy(self, ng_counts):
		true = 0.0
		for (ctxt, token), ng_count in ng_counts.items():
			ctxt_str = ''.join(ctxt)
			if((ctxt_str in self.prob) and 
				(self._pick_next_best(ctxt_str) == token)):
				true += 1
		return true/len(ng_counts)

	def _add_k(self, ng_count, ctxt_count, k):
		N = len(self.tokens)
		return (float(ng_count) + k) / (ctxt_count + (k*N))

	def _perplexity(self, ctxt_counts, ng_counts, s, k):
		entropy = 0.0
		N = len(ng_counts)
		for (ctxt, token), ng_count in ng_counts.items():
			ctxt_str = ''.join(ctxt)
			p = 0.0
			if s == 'add-k':
				count = self.ng_counts.get((ctxt, token), 0)
				p = self._add_k(count, self.ctxt_counts[ctxt], k)
			else:
				p = self.prob[ctxt_str].get(token, 0.0)
			entropy += math.log(p, 2)
		entropy /= -N
		return math.pow(2, entropy)

	def test(self, data, s, k):
		tokens = Counter()
		ng_counts = Counter()
		ctxt_counts = Counter()
		
		for tune in data:
			self._addTune(tune, tokens, ng_counts, ctxt_counts)
		
		accuracy = self._accuracy(ng_counts)
		perplexity = 0.0

		if s is not None:
			perplexity = self._perplexity(ctxt_counts, ng_counts, s, k)

		return accuracy, perplexity
