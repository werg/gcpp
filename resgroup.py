import resnet
import numpy as np
import random
import helpers
# cinint_false = randomInput(len(balph))
		
	
def evalstep(classifier, teststring, times, balph):
	correct = 0.0
	testout = ''
	for out in classifier.run_signal(teststring):
		ac = out[0][0,0] > 0
		correct += 1.0/times if ac else 0.0
		testout += '+' if ac else '-'
		
	refin = ''.join([helpers.pick_letter(balph, t) for t in teststring])
	return (correct, testout, refin)
	
PRERUN = 80
class ResGroup:
	def __init__(self, size, netsize, refdata, cinit_false, genin_size, alphsize, sparseness, scale, pretimes):
		self.generators = [resnet.ResGen(netsize, genin_size, alphsize, sparseness, scale) for i in xrange(size)]
		self.classifiers = [resnet.ResClass(netsize, alphsize, sparseness, scale) for i in xrange(size)]
		
		for c in self.classifiers:
			c.pretrain(refdata, cinit_false, pretimes, PRERUN)
			
		for g in self.generators:
			g.teacher_forced([v for (v,_) in zip(refdata, xrange(pretimes))], PRERUN)
				
		
	def trainstep(self, learnrate, refdata, times):
		for c in self.classifiers:
			c.changes = np.zeros_like(c.outnet)
		for g in self.generators:
			g.changes = np.zeros_like(g.outnet)
			for c in self.classifiers:
				(gc, cc) = resnet.crosstrain(g, c, learnrate, refdata, times)
				g.changes += gc/len(self.classifiers)
				c.changes += cc/len(self.generators)
				
			g.outnet += g.changes
			
		for c in self.classifiers:
			c.outnet += c.changes
		

	def crossevaluate(self, refdata, randdata, levendata, balph, times):
		generator = random.choice(self.generators)
		classifier = random.choice(self.classifiers)


		(correct, refclass, refin) = evalstep(classifier, [v for (v,_) in zip(refdata, xrange(times))],times, balph)
		(baseline, randclass, randin) = evalstep(classifier, [v for (v,_) in zip(randdata, xrange(times))], times,balph)
		(levenacc, levenclass, levenin) = evalstep(classifier, [v for (v,_) in zip(levendata, xrange(times))], times,balph)
		(accuracy, genclass, genin) = evalstep(classifier, [v[0] for v in generator.run_times(times)],times, balph)
	
		
		return (correct,baseline, levenacc, accuracy,(refin, refclass, randin, randclass, genin, genclass))
		
		
	def _copyscrub(self):
		for g in self.generators:
			del g.ingen
			
	def _copyrepair(self):
		for g in self.generators:
			g.reset()
