
# TODO:
# matnn.mutate
# close file / flush / new syntax with
# debug
cf = None

import cPickle
from random import gauss
import numpy as np
import datetime
from nltk.corpus import brown

from optparse import OptionParser
import os
from configobj import ConfigObj
from validate import Validator

import matnn as nn
import evo
import datasource
import helpers
import ngram

# more imports follow below


def process_config(dirname):
    cf = ConfigObj(dirname +"/config.cfg", unrepr=True, configspec="configspec.cfg")
    cf.validate(Validator(), copy=True)
    return cf


run_args = None
if __name__ == '__main__':

    parser = OptionParser()
    parser.add_option("-t", "--test", action="store_true", dest="test", default=False,
                  help="Test generation NUMBER from FOLDER.")
    parser.add_option("-d", "--datasrc", dest="datasrc",
                  help="DATASRC to load", metavar="DATASRC")
    parser.add_option('-n','--ngram', action="store_true", dest="ngram", default=False,
                  help="Run with ngram-model-generator.")
    parser.add_option('-l','--leven', action="store_true", dest="levenshtein", default=False,
                  help="Pre-train classifier with Levenshtein/minimum edit distance skewed negative examples.")

    (options, args) = parser.parse_args()
    if options.test:
        dirname = args[1]
    else:
        if len(args) > 1:
            dirname = args[1]
        else:
            dt = datetime.datetime.today()
            dirname = dt.strftime("%y-%m-%d_%H:%M:%S_%f")

        if not os.path.exists(dirname):
            os.makedirs(dirname)

    cf = process_config(dirname)

    evo.cf = cf
    nn.cf = cf
    ngram.cf = cf

    if options.datasrc:
        cf['datasource'] = options.datasrc
        
    (ref_data,cf['alphabet']) = datasource.setup_source(cf['datasource'])

    if options.test:
        import test
        test.cf = cf
        class_best_file = open(dirname + "/cbest" + args[0] + ".pickle")
        class_best = cPickle.load(class_best_file)
        class_best_file.close()
        if options.ngram:
            final_file = open(dirname + "/final.pickle")
            (t, gen_pop, class_pop) = cPickle.load(final_file)
            final_file.close()
            gen_best = gen_pop[:cf['best_size']]
        else:
            gen_best_file = open(dirname + "/gbest" + args[0] + ".pickle")
            gen_best = cPickle.load(gen_best_file)
            gen_best_file.close()
        test.runtest(gen_best, class_best, ref_data)

    else:
        if len(args) > 0:
            runtimes = int(args[0])
        else:
            runtimes = cf['runtimes']
        run_args = [runtimes, ref_data, dirname, cf['pop_size'], cf['gen_insize'], cf['init_hidden']]
        times = 0
        class_pop = None
        gen_pop = None
        evolve_gn = True
        if os.path.exists(dirname + "/final.pickle"):
            popsfile = open(dirname + "/final.pickle")
            (times, gen_pop, class_pop) = cPickle.load(popsfile)
            popsfile.close()
        if options.ngram:
#            if os.path.exists('ngram_pop'):
                gpfile = open('ngram_pop.pickle')
                gen_pop = cPickle.load(gpfile)
                gpfile.close()
                evolve_gn = False
 #           else:
 #               gen_pop = ngram.make_ngram_pop(cf['pop_size'], cf['max_ngram_n'])
        if options.leven:
            # do something :)
            
            evolve_gn = false
            
        run_args.extend((times, gen_pop, class_pop, evolve_gn))



## please note that there's another portion belonging to "main" below


def setup_net(insize, outsize, hidden_size):
    net = nn.makeMatRNN(insize,  outsize, hidden_size)
    net.fitness = 0

    return net
  
def setup_gen(insize, outsize, hidden_size):
    net = nn.makeGenMatRNN(insize,  outsize, hidden_size)
    net.fitness = 0

    return net
  


def setup_pops(pop_size, gen_insize, class_insize, output_size, hidden_size):
#    generator_pop = [setup_gen(gen_insize, class_insize, hidden_size) for i in xrange(pop_size)]
#    classifier_pop = [setup_class(class_insize, 1, hidden_size) for i in xrange(pop_size)]
    generator_pop = [setup_gen(gen_insize, class_insize, hidden_size) for i in xrange(pop_size)]
    classifier_pop = [setup_net(class_insize, 1, hidden_size) for i in xrange(pop_size)]
    return (generator_pop, classifier_pop)

def prepare_corpus(nltk_corpus):
    result = {}
    for w in nltk_corpus.words():
        for c in w:
            result[c] = True

    return result


def run_evo(times, ref_data, run_dir, pop_size, gen_insize, hidden_init,\
            times_init = 0, gen_pop = None, class_pop = None, evolve_gn = True, evolve_cn = True):

    if class_pop is None:
        (gen_pop1, class_pop) = setup_pops(pop_size, gen_insize, len(cf['alphabet']), 1, hidden_init)
        if gen_pop is None:
            gen_pop = gen_pop1


#    ref_data = corpus_stream(corpus)
    evo_params = (gen_pop, class_pop, ref_data, lambda: evo.randomInput(gen_insize), evolve_gn, evolve_cn)
    
    avg_cfit1_file = open(run_dir + "/avg_cfit1", 'a')
    avg_cfit2_file = open(run_dir + "/avg_cfit2", 'a')
    avg_gfit1_file = open(run_dir + "/avg_gfit1", 'a')
    avg_gfit2_file = open(run_dir + "/avg_gfit2", 'a')
    
    if cf['datasource'] == 'abba':
        import test
        test.cf = cf
        avg_score_file = open(run_dir + "/avg_score", 'a')

#    import time
#    def evolverep(*params):
#        while True:
#            time.sleep(1)
#            yield ([1,2], [0,2], 1, 0, 1, 0)


    for (i, (gen_pop, class_pop,avg_cfit1,avg_cfit2,avg_gfit1,avg_gfit2)) in zip(xrange(times_init, times_init+times),evo.evolve(*evo_params)):
#    for (i, (gen_pop, class_pop,avg_cfit1,avg_cfit2,avg_gfit1,avg_gfit2)) in zip(xrange(times_init, times_init+times),evolverep(*evo_params)):
        if evolve_cn:
#            import pdb; pdb.set_trace()
#            print "Ich bin hier"
            best_file = open(run_dir + "/cbest" + str(i) + ".pickle", 'w')
            cPickle.dump(class_pop[:cf['best_size']], best_file, protocol = cPickle.HIGHEST_PROTOCOL)
            best_file.flush()
            best_file.close()
        if evolve_gn:
            best_file = open(run_dir + "/gbest" + str(i) + ".pickle", 'w')
            cPickle.dump(gen_pop[:cf['best_size']], best_file, protocol = cPickle.HIGHEST_PROTOCOL)
            best_file.flush()
            best_file.close()
        print >> avg_cfit1_file, avg_cfit1
        print >> avg_cfit2_file, avg_cfit2
        print >> avg_gfit1_file, avg_gfit1
        print >> avg_gfit2_file, avg_gfit2
        
        if cf['datasource'] == 'abba':
            print >> avg_score_file, test.runscore(gen_pop[:cf['best_size']])

    avg_cfit1_file.close()
    avg_cfit2_file.close()
    avg_gfit1_file.close()
    avg_gfit2_file.close()
    if cf['datasource'] == 'abba':
        avg_score_file.close()
    
    final_file = open(run_dir + "/final.pickle", 'w')
    cPickle.dump((times + times_init, gen_pop, class_pop), final_file, protocol = cPickle.HIGHEST_PROTOCOL)
    final_file.close()
    cf.write()



if __name__ == "__main__" and not options.test:
    run_evo(*run_args)
