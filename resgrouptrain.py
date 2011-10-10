# example usage:



import resnet
import random
import numpy as np
import helpers
import resgroup
from optparse import OptionParser
import cPickle
from optparse import OptionParser
import copy
import os

#from nltk.corpus import brown
balph = [' ', 'a', 'c', 'b', 'e','d', 'g', 'f', 'i', 'h', 'k', 'j', 'm', 'l', 'o', 'n', 'q', 'p','s', 'r', 'u', 't', 'w', 'v', 'y', 'x', 'z']	

def charstream(source, delimeter = " "):
	while True:
		for w in source:
			for c in w:
				c = c.lower()
				if c in balph:
					yield c
			yield delimeter

def vs(source):
	for c in source:
		yield np.mat([[1.0 if d == c else -1.0] for d in balph])

def vcs(source):
	return vs(charstream(source))


def filesource(filename):
	file = open(filename)
	lines = file.readlines()
	file.close()
	while True:
		for l in lines:
			yield l.strip()
	

def bigeval(startgroup, endgroup, refdata, times):
	base = 0.0
	perf = 0.0
	cont = 0.0
	for (cs,ce) in zip(startgroup.classifiers, endgroup.classifiers):
		for g in startgroup.generators:
			base += resgroup.evalstep(ce,[v[0] for v in g.run_times(times)], times, balph)[0]
		for g in startgroup.generators:
			cont += resgroup.evalstep(cs,[v[0] for v in g.run_times(times)], times, balph)[0]
		for g in endgroup.generators:
			perf += resgroup.evalstep(ce,[v[0] for v in g.run_times(times)], times, balph)[0]
			
	base /= len(endgroup.classifiers) * len(startgroup.generators)
	perf /= len(endgroup.classifiers) * len(endgroup.generators)
	cont /= len(startgroup.classifiers) * len(startgroup.generators)
	
	return (base, perf, cont)

# base: startgroup gens against endgroup clas
# cont: startgroup gens against startgroup clas (control)
# perf: endgroup gens against endgroup clas (performance)
# of interest:
# perf - base, base - cont (to see whether 
			

INSIZE = 15 # 
SIZE = 10 #
NETSIZE = 400 #
PRETIMES = 600 #
#STEPS = 10
TIMESPERSTEP = 25
LEVENMUTATE = .25


if __name__ == '__main__':
# python resgrouptrain -
	parser = OptionParser()
	parser.add_option("-t", "--test", dest="test", default='-1',
                  help="Test every N generations and send output to standard out, default is 0 = not at all.", metavar ="N")
	parser.add_option("-o", "--outfile", dest="outfile",
                  help="Send performance data to OUTFILE", metavar="OUTFILE")
	parser.add_option("-s", "--store", dest="store",
                  help="Store networks in STORE", metavar="STORE")
	parser.add_option("-l", "--load", dest="load",
                  help="Load networks from LOAD", metavar="LOAD")
	parser.add_option("-r", "--refdata", dest="refdata",
                  help="Load reference data from REFDATA", metavar="REFDATA")
	parser.add_option("-p", "--sparse", dest="sparse", default = '0.3',
                  help="Set sparseness value to SPARSE", metavar="SPARSE")
	parser.add_option("-c", "--scale", dest="scale", default = '0.5',
                  help="Set scale value to SCALE", metavar="SCALE")
	parser.add_option("-e", "--learn", dest="learn", default = '0.08',
                  help="Set learnrate to LEARN", metavar="LEARN")
                  
                  
	(options, args) = parser.parse_args()
	SPARSE = float(options.sparse)
	SCALE = float(options.scale)
	LEARNRATE = float(options.learn)
	if options.refdata:
		refsource = filesource(options.refdata)
	else:
		from nltk.corpus import brown
		def brownsource():
			while True:
				for w in brown.words():
					yield w
		refsource = brownsource()
#		def vcsb():
#			from nltk.corpus import brown
#			for v in vcs(brown.words()):
#				yield v

	refdata = vcs(refsource)
		

	
	if options.outfile:
		if not os.path.exists(options.outfile):
			os.makedirs(options.outfile)
			
		correctfile = open(options.outfile + "/correct", 'w')
		baselinefile = open(options.outfile + "/baseline", 'w')
		accuracyfile = open(options.outfile + "/accuracy", 'w')
		levenfile = open(options.outfile + "/levenacc", 'w')
	

	
	if True:
		def csn():
			for c in charstream(refsource):
				if random.random() < LEVENMUTATE:
					yield random.choice(balph)
				else:
					yield c
					
		vcsn = vs(csn())
	else:
		vcsn = resnet.extremeRI(len(balph))
	
	if options.load:
		loadfile = open(options.load)
		(startgroup, group) = cPickle.load(loadfile)
		loadfile.close()
		if not options.store:
			options.store = options.load
		group._copyrepair()
		startgroup._copyrepair()

	else:
		group = resgroup.ResGroup(SIZE, NETSIZE, refdata, vcsn, INSIZE, len(balph), SPARSE, SCALE, PRETIMES)
			
		group._copyscrub()
		startgroup = copy.deepcopy(group)
		group._copyrepair()
		startgroup._copyrepair()
	
	thresh = int(options.test)
	t = 0
	for _ in xrange(int(args[0])):
		group.trainstep(LEARNRATE, refdata, TIMESPERSTEP)
		#(refin, refclass, randin, randclass, genin, genclass)
		(correct, baseline, levenacc, accuracy, evalouts) = group.crossevaluate(refdata, resnet.extremeRI(len(balph)), vcsn, balph, 30)
		t += min(thresh, 1)
		if t >= thresh:
			for out in evalouts:
				print out
			t = 0
		
		performance = str(correct) + " " + str(baseline) + " " + str(accuracy)
		if options.outfile:
			correctfile.write(str(correct) + "\n")
			baselinefile.write(str(baseline) + "\n")
			accuracyfile.write(str(accuracy) + "\n")
			levenfile.write(str(levenacc) + "\n")
		else:
			print str(correct) + " " + str(baseline) + " " + str(accuracy)
			
	if options.outfile:
		correctfile.close()
		baselinefile.close()
		accuracyfile.close()
		levenfile.close()
		
	(base, perf, cont) = bigeval(startgroup, group, refdata, TIMESPERSTEP)
	print str(base) + ' ' + str(perf) + ' ' + str(cont)

	if options.store:
		storage = open(options.store, 'w')
		group._copyscrub()
		startgroup._copyscrub()
		cPickle.dump((startgroup, group), storage, protocol = cPickle.HIGHEST_PROTOCOL)
		storage.close()
