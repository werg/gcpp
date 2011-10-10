import resnet
import random
import numpy as np
import helpers

from nltk.corpus import brown
balph = [' ', 'a', 'c', 'b', 'e','d', 'g', 'f', 'i', 'h', 'k', 'j', 'm', 'l', 'o', 'n', 'q', 'p','s', 'r', 'u', 't', 'w', 'v', 'y', 'x', 'z']	

def charstream():
	while True:
		for w in brown.words():
			for c in w:
				c = c.lower()
				if c in balph:
					yield c
			yield " "

def vcs():
	for c in charstream():
		yield np.mat([[1.0 if d == c else -1.0] for d in balph])

#balph = ['1', '0']	
#def vcs():
#	while True:
#		yield np.mat([[1.0],[-1.0]])

TIMES = 9000
RESSIZE = 600

# ressize, insize, sparseness, scale
classifier = resnet.ResClass(RESSIZE, len(balph), 0.3, 0.5)


		   
classifier.pretrain(vcs(), resnet.extremeRI(len(balph)), TIMES, 80)
	
teststring = [v for (v,_) in zip(vcs(), xrange(100))]
testout = ''
for out in classifier.run_signal(teststring):
	testout += '+' if out[0][0,0] > 0 else '-'
	
print ''.join([helpers.pick_letter(balph, t) for t in teststring])
print testout

	 
teststring = [v for (v,_) in zip(resnet.extremeRI(len(balph)), xrange(100))]
testout = ''
for out in classifier.run_signal(teststring):
	testout += '+' if out[0][0,0] > 0 else '-'
	
print ''.join([helpers.pick_letter(balph, t) for t in teststring])
print testout

