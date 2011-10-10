import copy
from math import tanh
from random import gauss, random, randint, choice
from params import *

# TODO:
#   * think about representation (hypercube)
#   * abbruchbedingung (raw-input)
#   * interleave newly generated nets

class Neuron(object):
    """Neuron base class."""
    def __init__(self, bias, inputs = None, targets = None, transfer = tanh):
        if inputs is None:
            self.inputs = []
        else:
            self.inputs = inputs
        if targets is None:
            self.targets = []
        else:
            self.targets = targets
        self.transfer = tanh
        self.bias = bias


    def update(self):
        """Calculate activation and pass it on to target synapses."""
        activation = self.bias
        for i in self.inputs:
            activation += i()

        activation = self.transfer(activation)
        for target in self.targets:
            target.pushInput(activation)

        self.activation = activation

        return activation


class InputUnit(object):
    """InputUnits are placeholders for input-information. Can have outgoing synapses."""
    def __init__(self, targets = None):
        if targets is None:
            self.targets = []
        else:
            self.targets = targets

    def pushInput(self, inactivation):
        """Push input to all outgoing synapses."""
        for target in self.targets:
            target.pushInput(inactivation)



class Synapse(object):
    def __init__(self, source, target, weight):
        self.source = source
        self.target = target
        self.weight = weight
#        self.inactivation = gauss(ACTIV_RINIT_MEAN, ACTIV_RINIT_SDEV)

    def pushInput(self, inactivation):
        """Set input, without propagating / calculating activation.
           This is done to maintain new activation on backburner until
           it can be made sure that the target neuron has already processed
           the old input-value."""
        self.inactivation = inactivation

    def update(self):
        """Retreives inactivation from backburner and installs it as activation
           to be reitreived by __call__ in the next cycle."""
        self.activation = self.inactivation * self.weight

    def __call__(self):
        """Just returns the activation"""
        return self.activation


def connected(source, target):
    """Returns True if source and target are connected neurons. Else False."""
    result = False
    for t in source.targets:
        for s in target.inputs:
            if t is s:
                result = True

    return result



class RecurrentNet(object):
    def __init__(self, output_units = None, inputs = None, hidden = None, synapses = None):
        if output_units is None:
            self.output_units = []
        else:
            self.output_units = output_units

        if hidden is None:
            self.hidden = []
        else:
            self.hidden = neurons

        if inputs is None:
            self.inputs = []
        else:
            self.inputs = inputs
        if synapses is None:
            self.synapses = []
        else:
            self.synapses = synapses

#       self._randomInitActiv()

    def size(self):
        return len(self.hidden) + len(self.synapses) + len(self.output_units)

    def addSynapse(self, source, target, weight):
        """Add a synapse. No check is done whether source and target are already connected."""
        synapse = Synapse(source, target, weight)
        self.synapses.append(synapse)
        source.targets.append(synapse)
        target.inputs.append(synapse)
        return synapse

    def addNeuron(self, bias):
        """Appends new neuron of bias to list of neurons."""
        neuron = Neuron(bias)
        self.hidden.append(neuron)
        return neuron

    def deleteSynapse(self, synapse):
        self.synapses.remove(synapse)
        synapse.source.targets.remove(synapse)
        synapse.target.inputs.remove(synapse)
        del synapse


    def deleteNeuron(self, neuron):
        self.hidden.remove(neuron)
        for s in neuron.inputs:
            self.deleteSynapse(s)
        for s in neuron.targets:
            self.deleteSynapse(s)
        del neuron

    def update_step(self, step_input):
        """Central step function, takes step_input, propagates for one step Extracts output."""
        for (sin,nin) in zip(step_input, self.inputs):
            nin.pushInput(sin)

        for s in self.synapses:
            s.update()

        for n in self.hidden:
            n.update()

        for n in self.output_units:
            n.update()

        return [o.activation for o in self.output_units]

    def _randomInitActiv(self):
        """Random initialize all activity."""
        for synapse in self.synapses:
            synapse.pushInput(gauss(ACTIV_RINIT_MEAN,ACTIV_RINIT_SDEV))

    def reset(self):
        self._randomInitActiv()

    def rInsertNeuron(self, syn_n_insert, max_syn_n_insert):
        neuron = self.addNeuron(gauss(BIAS_INIT_MEAN,BIAS_INIT_SDEV))
        for i in xrange(max_syn_n_insert):
            if random() < syn_n_insert:
                if randint(0,1):
                    self.rInsertSynapse(target = neuron)
                else:
                    self.rInsertSynapse(source = neuron)

    def rInsertSynapse(self, source = None, target = None):
        while source is None or target is None:
            if source is None:
                source1 = choice(self.hidden + self.inputs)
            else:
                source1 = source
            if target is None:
                target1 = choice(self.hidden + self.output_units)
            else:
                target1 = target

            if not connected(source1, target1):
                source = source1
                target = target1

        return self.addSynapse(source, target, gauss(WEIGHT_INIT_MEAN, WEIGHT_INIT_SDEV))


    def mutate(self, change_weight, change_bias, change_rate, n_insert,\
     max_n_insert, syn_n_insert, max_syn_n_insert, s_insert,\
     max_s_insert, n_delete, max_n_delete, s_delete, max_s_delete):
        """Mutation operator. changes weights, biases, inserts and deletes synapses and
           neurons."""

        # delete neurons
        for i in xrange(max_n_delete):
            if random() < n_delete and len(self.hidden) > 0:
                del_n = choice(self.hidden)
                self.deleteNeuron(del_n)

        # delete synapses
        for i in xrange(max_s_delete):
            if random() < s_delete and len(self.synapses) > 0:
                del_s = choice(self.synapses)
                self.deleteSynapse(del_s)

        # change biases
        for n in self.hidden + self.output_units:
            if random() < change_bias:
                n.bias += gauss(0,change_rate)

        # change weights
        for s in self.synapses:
            if random() < change_weight:
                s.weight += gauss(0,change_rate)

        # insert synapse
        for i in xrange(max_s_insert):
            if random() < s_insert:
                self.rInsertSynapse()

        # insert neuron
        for i in xrange(max_n_insert):
            if random() < n_insert:
                self.rInsertNeuron(syn_n_insert, max_syn_n_insert)

    def makeChild(self):
        child = copy.deepcopy(self)
#       child = RecurrentNet(copyself.output_units, self.inputs, self.hidden)
        child.mutate(CHANGE_WEIGHT, CHANGE_BIAS, CHANGE_RATE, N_INSERT,\
            MAX_N_INSERT, SYN_N_INSERT, MAX_SYN_N_INSERT, S_INSERT,\
            MAX_S_INSERT, N_DELETE, MAX_N_DELETE, S_DELETE, MAX_S_DELETE)

        return child

#def setup_rnn(input_source, output_size, net_size):
#    neurons = [


