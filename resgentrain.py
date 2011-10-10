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
		

teacher = [v for (v,_) in zip(vcs(), xrange(40000))]


RESSIZE = 400
INSIZE = 40
# __init__(self, ressize, insize, outsize, sparseness, scale):
gen = resnet.ResGen(RESSIZE,   INSIZE,  len(balph),  0.3,       0.5)

#print gen.outnet

gen.teacher_forced(teacher, 80)

print ".. after training:"
#print gen.outnet

testout = ""
for i in xrange(100):
	testout += helpers.pick_letter(balph, gen.update_step()[0])
	
print testout
