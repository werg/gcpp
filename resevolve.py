import numpy as np

ttanh = lambda x: math.tanh(x)
vtanh = np.vectorize(ttanh)


def evolve(gen_pop, class_pop, gen_adapt_pop):
	

#we evolve
#	generator
#			randin_size
#			size
#			layers
#			sparseness
#			scale
#			pretrain_len
#	
#	classifier
#			size
#			layers
#			sparseness
#			scale
#			pretrain_len
#	
#	gen_adapter window
#			ignore (one or zero)
#			kill
#			neutralize
#			amount
			

def eval_run(inrun, classifier):
	while classifer.eval(genrun.next()) >= 0:
		yield 1
		
def evaluate(generator, classifier, refdata, maxrun):
#	run single evaluation only until generator is caught
#	a classifier only makes as much dent in generator's fitness as it gets reference data right
		for (genlen,_) in zip(xrange(maxrun), eval_run(generator.run(), classifier)):
			pass
		
		for (reflen,_) in zip(xrange(maxrun), eval_run(refdata, classifier)):
			pass
		
		return (genlen, reflen)

		
class ResClass(resnet):
		
		
learning
	generator
		initialization
			teacher-forced corpus training, supervised
				slightly weaken strongest (or all by strength) if wrong
				strengthen correct response
				
			reinforcement 
				every good hit until a catch-response from classifier gets rewared
				several ranges of history that get afflicted
					ignore first one or zero (factor 0)
					kill preciding k (factor -1)
					neutralize preceding n (factor 0)
					and just keep reinforcement on everyone else
					apply weighted adaption amounts
					


