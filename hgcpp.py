import matnn	
import numpy as np
import operator
import math

def evo_step(breed_pop, eval_hof):
		chosen_one = None
		eval_pop = []
		while not chosen_one:
			(breed_pop, eval_pop) = refresh_pop(breed_pop, eval_pop)
			chosen_one = None
			for e in eval_pop:
				if e.evaluate_against(eval_hof):
					chosen_one = e
					break
			
		return ([chosen_one] + breed_pop, chosen_one)
		
def refresh_pop(breed_pop, eval_pop):
	breed_pop += eval_pop
	breed_pop.sort(key=operator.attrgetter('perf'), reverse=True)
	breed_pop = breed_pop[:cf['pop_size']]
	
	make_random = breed_pop[0].__class__
	eval_pop = [b.makeChild() for b in breed_pop] + [make_random() for i in xrange(cf['rand_probes'])]
	return (breed_pop, eval_pop)
	
def gen_process_params(output_size = None, weights = None, biases = None, gen_input = None):
	if gen_input is None:
		gen_input = matnn.randomInput
	if output_size is None:
		output_size = len(cf['alphabet'])
	hidden_size = cf['init_hidden']
	size = hidden_size + output_size
	if weights is None:
		weights = np.mat(matnn.genentry((size,size+cf['gen_insize'])))
	if biases is None:
		biases = np.mat(matnn.genbias((size, 1)))
		
	return (output_size, weights, biases, gen_input)
		
	

class Generator(matnn.GenMatRNN):
	def __init__(self, *params):
		matnn.GenMatRNN.__init__(self,*gen_process_params(*params))
		self.perf = 0.0
		
	def evaluate_against(self,c_hof):
		successful = True
		self.perf = 0.0
		for c in c_hof:
			(score, ratio) = self.run_eval(c.run(self.run_clean()))
			self.perf += ratio
			if ratio < cf['perf_thresh']:
				successful = False
				break
		return successful
		
	def run_eval(self, c):
		totalweights = 0
		score = 0.0
		count = 0.0
		while totalweights < 1:
			for (i,out) in enumerate(c):
				w = min(i, cf['fitweight_thresh'])
				totalweights += w
				score += w * out[0,0]
				count += math.copysign(1.0, out[0,0])
				
		return (score / totalweights, count/(i+1))

def class_process_params(output_size = 1, weights = None, biases = None):
	input_size = len(cf['alphabet'])
	hidden_size = cf['init_hidden']
	size = hidden_size + output_size
	if weights is None:
		weights = np.mat(matnn.genentry((size,size+input_size)))
	if biases is None:
		biases = np.mat(matnn.genbias((size, 1)))

	return (output_size, weights, biases)

class Classifier(matnn.MatRNN):
	def __init__(self, *params):
		matnn.MatRNN.__init__(self, *class_process_params(*params))
		self.perf = 0.0

		
	def evaluate_against(self,g_hof):
		successful = True
		(score, self.perf) = self.run_eval(self.run(ref_data.next()))
		self.perf *= -1
		if self.perf < cf['perf_thresh']:
			successful = False
		else:
			for g in g_hof:
				(score, ratio) = self.run_eval(self.run(g.run_clean()))
				self.perf += ratio
				if ratio < cf['perf_thresh']:
					successful = False
					break

		return successful
		
	def run_eval(self, g):
		totalweights = 0
		score = 0.0
		count = 0.0
		while totalweights < 1:
			for (i,out) in enumerate(g):
				w = min(i, cf['fitweight_thresh'])
				totalweights += w
				score -= w * out[0,0]
				count -= math.copysign(1.0, out[0,0])
				
		return (score / totalweights, count/(i+1))
