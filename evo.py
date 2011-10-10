import random
#from control import cf
import math
import numpy as np
import helpers
import copy

def evaluate(generator, classifier, ref_data):
    """Evaluate generator against classifier, classifier against generator output and reference data.
       Returns classifier and generator fitness."""
       
    totalweights = 0
    positive = 0.0
    while totalweights < 1:
        for (i, out) in enumerate(classifier.run(ref_data.next())):
            w = min(i, cf['fitweight_thresh'])
            totalweights += w
            positive += w * out[0,0]
        
    positive /= totalweights

    totalweights = 0
    negative = 0.0
    while totalweights < 1:
        for (i,out) in enumerate(classifier.run(generator.run_clean())):
            w = min(i, cf['fitweight_thresh'])
            totalweights += w
            negative += w * out[0,0]
            
    negative /= totalweights
    balance = cf['balance'] * math.fabs((positive + negative)/2.0)
    decisiveness = cf['decisiveness'] * (1.0 - (math.fabs(positive) + math.fabs(negative))/2.0)

    return (cf['refdata_pref'] * positive - negative - balance - decisiveness, negative)

def _biasedChoice(li):
    n = len(li)
    m = n * (n+1) / 2
    r = random.randint(1,m)
    i = 0
    l = n
    while r > l:
        i += 1
        l += n - i

    return li[i]

def cmp(x, y):
    diff = y.fitness - x.fitness
    if diff:
        return int(diff/math.fabs(diff))
    else:
        return 0

def cmp_size(x, y):
    diff = y.fitness/(y.size()+0.01) - x.fitness/(x.size()+0.01)
    if diff:
        return int(diff/(math.fabs(diff)+0.01))
    else:
        return 0

def _liveOn(population):
    """Pick individuals to live on via biased choice."""
    popsize1 = int(len(population)*cf['longevity_rate'])
    result = []
    for i in xrange(popsize1):
        liver = _biasedChoice(population)
        while liver in result:
            liver = _biasedChoice(population)
        result.append(liver)
    return result


def evolution_step(generator_pop, classifier_pop, ref_data, gen_input, evolve_gn = True, evolve_cl = True):

    gpopsize1 =  int(len(generator_pop) * cf['longevity_rate'])
    gpopsize2 = len(generator_pop) - gpopsize1
    cpopsize1 =  int(len(classifier_pop) * cf['longevity_rate'])
    cpopsize2 = len(classifier_pop) - cpopsize1

    if evolve_gn:
        generator_pop1 = _liveOn(generator_pop)
    else:
        gp = copy.copy(generator_pop)
        random.shuffle(gp)
        generator_pop1 = gp[:gpopsize1]
        generator_pop2 = gp[gpopsize1:]
    if evolve_cl:
        classifier_pop1 = _liveOn(classifier_pop)
    else:
        cp = random.shuffle(classifier_pop)
        classifier_pop1 = cp[:cpopsize1]
        classifier_pop2 = cp[cpopsize1:]
    
    # TODO: put MAINTAIN into cf
    MAINTAIN = 1 - cf['fit_up_rate']
    overall_cfit = 0.0
    overall_gfit = 0.0
    for (generator,classifier) in zip(generator_pop1, classifier_pop1):
        (cfit, gfit) = evaluate(generator, classifier, ref_data)
        generator.fitness *= MAINTAIN
        generator.fitness += gfit * cf['fit_up_rate']
        classifier.fitness *= MAINTAIN
        classifier.fitness += cfit * cf['fit_up_rate']
        overall_cfit += cfit
        overall_gfit += gfit
        
    avg_cfit1 = overall_cfit / len(classifier_pop1)
    avg_gfit1 = overall_gfit / len(generator_pop1)
    
    # make children
    if evolve_gn:
        s_genpop = sorted(generator_pop, cmp_size)
        generator_pop2 = [_biasedChoice(s_genpop).makeChild() for i in xrange(gpopsize2)]
    
    overall_gfit = 0.0    
    for (generator,classifier) in zip(generator_pop2, classifier_pop1):
        (cfit, gfit) = evaluate(generator, classifier, ref_data)
        generator.fitness = gfit
        overall_gfit += gfit

    overall_cfit = 0.0
    if evolve_cl:
        s_classpop = sorted(classifier_pop, cmp_size)
        classifier_pop2 = [_biasedChoice(s_classpop).makeChild() for i in xrange(cpopsize2)]
    
    for (generator,classifier) in zip(generator_pop1, classifier_pop2):
        (cfit, gfit) = evaluate(generator, classifier, ref_data)
        classifier.fitness = cfit
        overall_cfit += cfit
       
    avg_cfit2 = overall_cfit / len(classifier_pop2)
    avg_gfit2 = overall_gfit / len(generator_pop2)

    generator_pop = generator_pop1 + generator_pop2
    classifier_pop = classifier_pop1 + classifier_pop2
    generator_pop.sort(cmp)
    classifier_pop.sort(cmp)

    return (generator_pop, classifier_pop, avg_cfit1, avg_cfit2, avg_gfit1, avg_gfit2)


def evolve(generator_pop, classifier_pop, ref_data, gen_input, evolve_gn, evolve_cl):
    while True:
        (generator_pop, classifier_pop,avg_cfit1,avg_cfit2,avg_gfit1,avg_gfit2) = evolution_step(generator_pop,  classifier_pop, ref_data, gen_input, evolve_gn, evolve_cl)
        yield (generator_pop, classifier_pop, avg_cfit1, avg_cfit2, avg_gfit1, avg_gfit2)
