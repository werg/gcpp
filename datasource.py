import numpy as np
import random

def sentstream(sent, alphabet):
	for w in sent:
		for c in w:
			c = c.lower()
#                if not c in cf['alphabet']:
#                    c = '*'
			if c in alphabet:
				yield c
		yield " "

def corpus_stream(nltk_corpus, alphabet):
	while True:
		for sent in nltk_corpus.sents():
			yield sentstream(sent, alphabet)
		
balph = [' ', 'a', 'c', 'b', 'e','d', 'g', 'f', 'i', 'h', 'k', 'j', 'm', 'l', 'o', 'n', 'q', 'p','s', 'r', 'u', 't', 'w', 'v', 'y', 'x', 'z']	
def brownsource():
	from nltk.corpus import brown
	return corpus_stream(brown, balph)

def abba():
	def abbasent():
		for i in xrange(25):
			yield 'a'
			yield 'b'

	def abbanest():
		while True:
			yield abbasent()
	
	return abbanest()
		
abalph = ['a','b']

abbatable = [[(1, 'a')], [(0, 'b'), (2, 'b')]]		
		
rebertable = [[(1,'b')],
	[(2,'t'), (3,'p')],
	[(2,'s'), (4,'x')],
	[(3,'t'), (5,'v')],
	[(3,'x'), (6,'s')],
	[(4,'p'), (6,'v')],
	[(7,'e')]]
		

def genfrom_fsa(table):
	def generator():
		for i in xrange(random.randint(1,8)):
			state = 0
			while state != len(table):
				(state, letter) = random.choice(table[state])
				yield letter
			
	def sentwrapper():
		while True:
			yield generator()
	 
	return sentwrapper

def process_fsa(table):
    transmap = {}
    possible_entries = [[] for t in table]
    for (i,t) in enumerate(table):
        for (n, c) in t:
            possible_entries[n].append((i, c))
            
    for (en, ex) in zip(possible_entries, table):
        for (nn, nc) in en:
            for (xn, xc) in ex:
                trans = nc + xc
                if not trans in transmap:
                    transmap[trans] = []
                    
                transmap[trans].append((nn, xn))
                
    return transmap

def transfsa_fitness(string, fsa_tbl, transmap):
    s0 = string[0]
    for s in string[1:]:
        pass
        
	
reberalph = ['b','t','p','x','v','s','e']	

sources = {
	'brown': (brownsource, balph), 
	'abba': (abba, abalph),
	'reber': (genfrom_fsa(rebertable), reberalph),
}

def setup_source(source_name):
	(src, alph) = sources[source_name]
	l = len(alph)
	charvectors = np.mat([[1.0 if i==j else -1.0 for i in xrange(l)] for j in xrange(l)])

	def cvstream(sent):
		"""Yields a column-vector representing the character at hand."""
		for c in sent:
			yield charvectors[:,alph.index(c)]

	def sentcvstream():
		for sent in src():
			yield cvstream(sent)
		
	return (sentcvstream(), alph)
