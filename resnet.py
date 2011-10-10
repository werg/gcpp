import numpy as np
import numpy.linalg as linalg
import helpers
import random
import math

cf = {'init_sdv': 0.24, 'init_sdv0': 0.05}

WINDOW = 4

def fromout_dtanh(out):
	"""Hyperbolic tangent derivative based on outnet state which has already gone through tanh."""
	return np.ones_like(out) - np.multiply(out,out)
	
def deltas(out, target):
	return np.multiply(fromout_dtanh(out), target - out)

def weightchange(out, target, state):
	delt = deltas(out, target)
	return (np.hstack([state for i in xrange(delt.shape[0])]) * delt).transpose()

def smoothify_success(success):
	result = [0 for s in success]
	quot = WINDOW/2 * (WINDOW + 1)
	for i in xrange(len(success)):
		for w in xrange(min(i,WINDOW)+1):
			result[i-w] = (WINDOW - w) * success[i]/quot
	
	return result

# TODO:
# find out whether success commutes down to target thingy
# make sure learnrate is included correctly

def crosstrain(generator, classifier, learnrate, refdata, times):
	success = [None for i in xrange(times)]
	outs = [None for i in xrange(times)]
	states = [None for i in xrange(times)]

	classchanges = np.zeros_like(classifier.outnet)
	
	for i in xrange(times):
		(outs[i],states[i]) = generator.update_step()
		(success[i], classchange) = classifier.trainstep(helpers.extremify(outs[i]), -1)
		classchanges += learnrate/times * classchange
		
	smoothify_success(success)
	
	genchanges = np.zeros_like(generator.outnet)
	for (out, state, suc) in zip(outs, states, success):
		genchanges += learnrate/times * weightchange(out, helpers.extremify(suc * out), state)

	
	for (i,ref) in zip(xrange(times), refdata):
		(suc, classchange) = classifier.trainstep(ref, 1)
		classchanges += learnrate/times * classchange
		
	return (genchanges, classchanges)
		

def tridistro(ratio, offset, mu0, sigma0, sigma1):
    """Return random values in a combined distribution with one normal
       distribution around mu0, flanked by two distributions at offset
       in the negative and positive."""
    seed = random.random()
    if seed < ratio:
        return random.gauss(mu0, sigma0)
    elif seed < ratio + (1.0-ratio)/2:
        return random.gauss(offset, sigma1)
    else:
        return random.gauss(-offset, sigma1)

ttanh = lambda x: math.tanh(x)
vtanh = np.vectorize(ttanh)

# TODO: classifier learning
# TODO: just run generator learning without classifier & all the GCPP stuff

@helpers.arraygen
def random_gen():
    return random.uniform(-1,1)
    
@np.vectorize
def random_change(x):
	x1 = random.uniform(-1,1)
	if random.random() > 0.5:
		return (x1 + x)/2
	else:
		return x1 


def randomInput(input_size):
    """Generator function for random input."""
    value = np.mat(random_gen((input_size,1)))
    while True:
        yield value
        yield value
        value = random_change(value)
        
def extremeRI(input_size):
	for i in randomInput(input_size):
		yield helpers.extremify(i)




class ResNet:
	def __init__(self, ressize, insize, outsize, sparseness, scale):
		tres = [[tridistro(sparseness,scale,0,cf['init_sdv0'],cf['init_sdv']) for i in xrange(ressize+insize+outsize)] for j in xrange(ressize)]
		self.reservoir = np.mat(tres)
		tout = [[random.gauss(0.0, cf['init_sdv']) for i in xrange(ressize)] for j in xrange(outsize)]
		self.outnet = np.mat(tout)
		self.state = np.mat([[0.0] for i in xrange(ressize)])
		self.out = np.mat([[0.0] for i in xrange(outsize)])
		
	def reset(self):
		self.state = np.mat(np.zeros_like(self.state))
		self.out = np.mat(np.zeros_like(self.out))
		
	def update_step(self, inp, teacher_signal = None):
		# TODO consider biases
		if not teacher_signal is None:
			self.out = teacher_signal
			
		inactiv = np.vstack((self.state, inp, self.out))
		self.state = vtanh(self.reservoir * inactiv)
		self.out = vtanh(self.outnet * self.state)
		
		return (self.out, self.state)
		

class ResGen(ResNet):
	def __init__(self, ressize, insize, outsize, sparseness, scale):
		ResNet.__init__(self, ressize, insize, outsize, sparseness, scale)
		self.insize = insize
		
	def reset(self):
		ResNet.reset(self)
		self.ingen = randomInput(self.insize)
		
	
	def update_step(self, teacher_signal = None):
		return ResNet.update_step(self, self.ingen.next(), teacher_signal)
		
	def run_signal(self, teacher_signal, prerun = 0):
		self.reset()
		if prerun == 0:
			yield self.update_step()
		else:
			self.update_step()
		for (i,t) in enumerate(teacher_signal):
			if i < prerun:
				self.update_step(t)
			else:
				yield self.update_step(t)
				
				
	def run_times(self,times):
		self.reset()
		for i in xrange(times):
			yield self.update_step()
					
	def teacher_forced(self, refdata, prerun):
		"""Do linear regression training as described in http://www.faculty.iu-bremen.de/hjaeger/pubs/ESNTutorial.pdf
		   Refdata must be an iterable of column matrices.
		   
		   Difference to regress_learn in ResClass: this allows for a teacher-signal."""
#		import pdb
#		pdb.set_trace()
		refstates = [r.transpose() for r in refdata[prerun:]]
		states = [s[1].transpose() for s in self.run_signal(refdata, prerun)]
		M = np.vstack(states)
#		pdb.set_trace()
		
		self.outnet = (linalg.pinv(M) * np.vstack(refstates)).transpose()

class ResClass(ResNet):
	def __init__(self, ressize, insize,  sparseness, scale):
		ResNet.__init__(self, ressize, insize, 1, sparseness, scale)
	
	def run_signal(self, insignal, prerun = 0):
		self.reset()
		for (i,insig) in enumerate(insignal):
			if i < prerun:
				self.update_step(insig)
			else:
				yield self.update_step(insig)[1]
		
	def trainstep(self, invector, target):
		(out, state) = self.update_step(invector)
		return (out[0,0], weightchange(out, np.mat(target), state))
		
	def pretrain(self, refdata, negdata, times, prerun):
		"""Do linear regression training as described in http://www.faculty.iu-bremen.de/hjaeger/pubs/ESNTutorial.pdf
		   Refdata and falsein must be an iterable of column matrices.
		   
		   Difference to teacher_forced in ResGen: this does not allow for a teacher-signal (since that would make the task trivial)."""
		M = np.vstack([s.transpose() for (_,s) in zip(xrange(times),self.run_signal(refdata,prerun))] \
					+ [s.transpose() for (_,s) in zip(xrange(times),self.run_signal(negdata,prerun))] )
		
		negtarget = np.mat(-1)
		postarget = np.mat(1)
		T = np.vstack([postarget for _ in xrange(times)] + [negtarget for _ in xrange(times)])
		
		self.outnet = (linalg.pinv(M) * T).transpose()
		

	#def regress_learn_changes(self, refdata, negdata, prerun):
		#"""Do linear regression training as described in http://www.faculty.iu-bremen.de/hjaeger/pubs/ESNTutorial.pdf
		   #Refdata must be an iterable of column matrices.
		   
		   #Difference to teacher_forced in ResGen: this does not allow for a teacher-signal (since that would make the task trivial)."""
		#M = np.vstack([s.transpose() for s in self.run_signal(refdata)])
##		print M.shape
		
		#T = np.vstack(refout)
		#self.outnet = (linalg.pinv(M) * T).transpose()
