from optparse import OptionParser
from cPickle import load
import evo
import math
from helpers import pick_letter

import pdb

def test_generator(generator, times):
    result = []
#    print "gen"
    for out in generator.run_clean(times):
#        print out
#        pdb.set_trace()
        result.append(out)

    return result


def test_classifier(classifier, text):

    # TODO: this assumes tanh as activation function
    result = []
#    print "class"
#    print text
#    print classifier.biases
    for out in classifier.run(text):
#        print out
#        pdb.set_trace()
        if out[0] > 0:
            result.append("+")
        else:
            result.append("-")
    return result

        
def abbafit(string):
    s0 = string[0]
    score  = 0.0
    for s in string[1:]:
        if s0 != s:
            score += 1
        s0 = s
    
    return score / (len(string) - 1)
    
def runscore(gen_best):
    score = 0.0
    for gen in gen_best:
        genout = test_generator(gen, cf['test_times'])
        gentxt = "".join([pick_letter(cf['alphabet'], out) for out in genout])
        score += abbafit(gentxt)
    return score / len(gen_best)

def teststep(gen, cla, ref_data):
        genout = test_generator(gen, cf['test_times'])
    #    print genout
        gentxt = [pick_letter(cf['alphabet'], out) for out in genout]
        gcltxt = test_classifier(cla, genout)
        score = ""
        if cf['datasource'] == 'abba':
            score = " score: " + str(abbafit(''.join(gentxt)))
        print "".join(gentxt) + score
        print "".join(gcltxt)
        
        corin = [c for c in ref_data.next()]
        corin = corin[:min(cf['test_times'], len(corin))]
    #    corin = []
    #    for c in cortxt:
    #        corin.append(charvectors[alphabet.index(c)])
    #    corin = [charvectors[alphabet.index(c)] for c in cortxt]
    #    print corin
        cortxt = [pick_letter(cf['alphabet'], c) for c in corin]
        ccltxt = test_classifier(cla, corin)
        print "".join(cortxt)
        print "".join(ccltxt)


def runtest(gen_best, class_best, ref_data):
#    file = open(filename, 'r')
#    (gen_best, class_best) = load(file)
#    file.close()

    evo.cf = cf

    for (gen, cla) in zip(gen_best,class_best):
        teststep(gen, cla, ref_data)
