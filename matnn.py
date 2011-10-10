import numpy as np
import math
import random
import numpy.random as npr
#from control import cf
import helpers
import copy

ttanh = lambda x: math.tanh(x)
vtanh = np.vectorize(ttanh)

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

@helpers.arraygen
def genentry():
    """Generates an entry using tridistro with config parameters."""
    return tridistro(cf['init_ratio'],cf['init_offs'],0,cf['init_sdv0'],cf['init_sdv'])

@helpers.arraygen
def genbias():
    return random.gauss(0, cf['init_biasdev'])


@np.vectorize
def mutate(x):
    """Mutates values of a numpy array such that near-zero values stay near zero."""

    r = random.random()
    if r < cf['mut_toggle']:
        # toggle between introduce/delete neuron
        height = min(1.0,1/(cf['mut_stretch']*x+1-cf['mut_horiz']))
        return height * random.choice([-1.0,1.0]) * random.gauss(cf['init_offs'], cf['init_sdv'])

    elif r < cf['mut_toggle'] + cf['mut_modify']:
        height = max(0.0,1-1/(cf['mut_stretch']*x+1-cf['mut_horiz']))
        return height * random.gauss(x, cf['mut_sdv'] * height)
    else:
        return x

class MatRNN(object):
    def __init__(self, output_size, weights, biases):
        # biases has to be a one-column matrix.
        # TODO: ensure shapes?
        self.weights = weights
        self.biases = biases
        self.reset()
        self.output_size = output_size
        self.input_size = weights.shape[1] - weights.shape[0]

    def reset(self):
        self.activation = np.zeros_like(self.biases)

    def update_step(self, step_input):
        """Central step function, takes step_input, propagates for one step. Extracts output.
            step_input has to be a one-column matrix."""
        inactiv = np.vstack((step_input, self.activation))
        self.activation = vtanh(self.weights * inactiv + self.biases)
        return self.activation[:self.output_size]        

    def _prepareChild(self):
        weights = self.weights
        biases = self.biases

        # delete
        if self.size() - self.output_size:
            del_n = npr.binomial(min(cf['max_ndel'], self.size() - self.output_size), cf['p_ndel'])
        else:
            del_n = 0

        if del_n:
            del_indices = [random.randrange(self.output_size, self.size()) for d in xrange(del_n)]
            del_indices = sorted(set(del_indices))
            keep_indices_r = []
            keep_indices_c = range(self.input_size)
            last_d = -1

            for d in del_indices:
                keep_indices_r.extend(range(last_d+1, d))
                keep_indices_c.extend(range(last_d+1+self.input_size, d+ self.input_size))
                last_d = d

            keep_indices_r.extend(range(last_d+1, weights.shape[0]))
            keep_indices_c.extend(range(last_d+1+self.input_size, weights.shape[1]))

            weights = weights[np.ix_(keep_indices_r, keep_indices_c)]
            biases = biases[np.ix_(keep_indices_r,[0])]

        weights = mutate(weights)
        # initialize
        intro_n = npr.binomial(cf['max_nintro'], cf['p_nintro'])
        if intro_n:
            weights = np.hstack((weights,genentry((weights.shape[0], intro_n))))
            weights = np.vstack((weights,genentry((intro_n, weights.shape[1]))))
            biases = np.vstack((mutate(biases),genbias((intro_n, 1))))
            
        return (weights, biases)

    def makeChild(self):
        (weights,biases) = self._prepareChild()
        return self.__class__(self.output_size, weights, biases)

    def size(self):
        return self.biases.shape[0]
            
    def run(self, input_gen):
        """Run an RNN as a generator."""
        self.reset()
        for i in input_gen:
            out = self.update_step(i)
            yield out

    def run_times(self, input_gen, times):
        """Run an RNN as a generator, times times."""
        self.reset()
        igit = input_gen.__iter__()
        for n in xrange(times):
            nextin = igit.next()
            yield self.update_step(nextin)


class GenMatRNN(MatRNN):
    def __init__(self, output_size, weights, biases, gen_input):
        MatRNN.__init__(self, output_size, weights, biases)
        self.gen_input = gen_input
    
    def run_clean(self, times = None):
        """Runs a generator so that output is clean, as if it came from corpus."""
        if times is None:
            times = cf['eval_len']
        for out in self.run_times(self.gen_input(self.input_size), times):
#            import pdb; pdb.set_trace()
            yield helpers.extremify(out)
            
    def makeChild(self):
        (weights,biases) = self._prepareChild()
        return self.__class__(self.output_size, weights, biases, self.gen_input)

    
class HistMatRNN(MatRNN):
    def __init__(self, output_size, weights, biases, histentry_size):
        self.histentry_size = histentry_size
        MatRNN.__init__(self, output_size, weights, biases)
        
    def reset(self):
        MatRNN.reset(self)
        self.hist = np.mat(np.zeros((cf['hist_size']*self.histentry_size,1)))

    def update_step(self, step_input):
        
        hist_input =  np.vstack((step_input, self.hist))
        accu = 0.0
        valquot = 0.0
        val = np.zeros_like(self.activation[:self.output_size-1])
        
        while accu < cf['step_thresh']:
            out = MatRNN.update_step(self, hist_input)
            val += out[1:] * accu
            accu += out[0] * 0.5 + 0.5 + cf['step_inc']
            valquot += accu
        
        val /= valquot
        return val
    
            
    def makeChild(self):
        (weights,biases) = self._prepareChild()
        return self.__class__(self.output_size, weights, biases, self.histentry_size)

  
        
class GenHMatRNN(HistMatRNN):
    def __init__(self, output_size, weights, biases, histentry_size, gen_input):
        assert histentry_size == output_size - 1
        HistMatRNN.__init__(self, output_size, weights, biases, histentry_size)

    def update_step(self, step_input):
        out = HistMatRNN.update_step(self, step_input)
        self.hist = np.vstack((self.hist[self.histentry_size:],helpers.extremify(out)))
        return out
    
    def run_clean(self, times = None):
        """Runs a generator so that output is clean, as if it came from corpus."""
        if times is None:
            times = cf['eval_len']
        for out in run_times(self, self.gen_input, times):
            yield helpers.extremify(out)


class ClassHMatRNN(HistMatRNN):
    def __init__(self, output_size, weights, biases, histentry_size):
        HistMatRNN.__init__(self, output_size, weights, biases, histentry_size)

    def update_step(self, step_input):
        out = HistMatRNN.update_step(self, step_input)
        self.hist = np.vstack((self.hist[self.histentry_size:],step_input))
        return out


def makeMatRNN(input_size, output_size, hidden_size):
    size = hidden_size + output_size
    weights = np.mat(genentry((size,size+input_size)))
    biases = np.mat(genbias((size, 1)))

    return MatRNN(output_size, weights, biases)


@np.vectorize
def random_change(x):
    return random.gauss(x, cf['ringen_chngsdv'])

@helpers.arraygen
def random_gen():
    return random.gauss(cf['ringen_mean'], cf['ringen_sdv'])

def randomInput(input_size):
    """Generator function for random input."""
    value = np.mat(random_gen((input_size,1)))
    while True:
        value = random_change(value)
        yield value
    

def makeGenMatRNN(input_size, output_size, hidden_size, gen_input = None):
    if gen_input is None:
        gen_input = randomInput
        
    size = hidden_size + output_size
    weights = np.mat(genentry((size,size+input_size)))
    biases = np.mat(genbias((size, 1)))

    return GenMatRNN(output_size, weights, biases, gen_input)
