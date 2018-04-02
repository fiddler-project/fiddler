from collections import Counter
import math, re
import nltk
import numpy as np


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
			self._addTune(tune)

	def _addTune(self, tune):
		tune = ['$',] * (self.n-1) + list(tune) + ['#']
		ngrams = zip(*[tune[i:] for i in range(self.n)])
		for grams in ngrams:
			ctxt = grams[:-1]
			token = grams[-1]
			self.tokens[token] += 1
			self.ng_counts[(ctxt, token)] += 1
			self.ctxt_counts[ctxt] += 1

	def train(self):
		for (ctxt, token), ng_count in self.ng_counts.items():
			ctxt_str = ''.join(ctxt)
			p = ng_count / self.ctxt_counts[ctxt]
			if ctxt_str in self.prob:
				self.prob[ctxt_str].update({token: p})
			else:
				self.prob[ctxt_str] = {token: p}

	def _pick_next(self, seed):
		tokens = list(self.prob[seed].keys())
		probs = list(self.prob[seed].values())
		return np.random.choice(tokens, 1, p=probs)[0]

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
