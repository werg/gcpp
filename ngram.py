from nltk.corpus import brown
import random
from copy import copy, deepcopy
import numpy as np

balph = [' ', 'a', 'c', 'b', 'e','d', 'g', 'f', 'i', 'h', 'k', 'j', 'm', 'l', 'o', 'n', 'q', 'p','s', 'r', 'u', 't', 'w', 'v', 'y', 'x', 'z']	

#SMOOTH = 0.3

l = len(balph)
charvectors = np.mat([[1.0 if i==j else -1.0 for i in xrange(l)] for j in xrange(l)])


def ngram_bstream(n):
	gram = " " * (n-1)
	for w in brown.words():
		for c in w:
			if c in balph:
				yield (gram, c)
				gram = gram[1:] + c
		yield (gram,' ')
		gram = gram[1:] + ' '
				
def build(stream, n, smooth):
	grams = copy(balph)
	for i in xrange(n-2):
		newgrams = []
		for g in grams:
			for b in balph:
				newgrams.append(g+b)
		grams = newgrams
	
	ngrams = {}
	total = dict([(g,len(balph) * smooth) for g in grams])
	ba1 = dict([ (b,smooth) for b in balph])
	
	for g in grams:
		ngrams[g] = copy(ba1)
	
	for (gram, c) in stream:
		ngrams[gram][c] += 1
		total[gram] += 1
		
	for g in ngrams:
		for b in balph:
			ngrams[g][b] /= total[g]
			
	return ngrams


def ngram_generator(n, smooth):
	ngrams = build(ngram_bstream(n), n, smooth)
	gram = " " * (n-1)
	while True:
		r = random.random()
		p = 0.0
		for c in ngrams[gram]:
			p += ngrams[gram][c]
			if p >= r:
				break
		gram = gram[1:] + c
		yield c


#print ngram_generator(4).next()

class ngram_source:
	def __init__(self,n, smooth):
		self.ngrams = build(ngram_bstream(n), n, smooth)
		self.n = n
		self.reset()
		
	def reset(self):
		self.gram = " " * (self.n-1)
		
	def update_step(self):
		r = random.random()
		p = 0.0
		for c in self.ngrams[self.gram]:
			p += self.ngrams[self.gram][c]
			if p >= r:
				break
		self.gram = self.gram[1:] + c
		return charvectors[:,balph.index(c)]
		
	def run(self, input_gen):
		"""Run an ngram-model as a generator."""
		self.reset()
		for i in input_gen:
			yield self.update_step()

	def run_times(self, input_gen, times):
		"""Run an ngram-model as a generator, times times."""
		self.reset()
		for n in xrange(times):
			yield self.update_step()
	
	def run_clean(self, times = None):
		if times is None:
			times = cf['eval_len']
		for i in xrange(times):
			yield self.update_step()
			
	def size(self):
		return 1
			
			
def make_ngram_pop(size, max_size):
	return [ngram_source(random.randint(3,max_size), random.random()) for i in  xrange(size)] 

