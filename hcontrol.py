import hgcpp
import control
import matnn
import datasource

from optparse import OptionParser
import os
from configobj import ConfigObj
from validate import Validator
import cPickle
import time


cf = None

def process_config(dirname):
	cf = ConfigObj(dirname +"/config.cfg", unrepr=True, configspec="h_configspec.cfg")
	cf.validate(Validator(), copy=True)
	return cf

if __name__ == '__main__':

	parser = OptionParser()
	parser.add_option("-t", "--test", dest="test", default="0",
				  help="Test generation NUMBER from FOLDER.")
	parser.add_option("-d", "--datasrc", dest="datasrc",
				  help="DATASRC to load", metavar="DATASRC")

	(options, args) = parser.parse_args()
	dirname = args[0]

	if not os.path.exists(dirname):
		os.makedirs(dirname)

	cf = process_config(dirname)
	
	matnn.cf = cf
	hgcpp.cf = cf
	
	
	if options.datasrc:
		cf['datasource'] = options.datasrc
		
	(ref_data,cf['alphabet']) = datasource.setup_source(cf['datasource'])
	hgcpp.ref_data = ref_data

	testnum = int(options.test)
	if testnum:
		import test
		# lets see what we put here
		hoffile = open(dirname + "/hof.pickle")
		(gen_hof, class_hof) = cPickle.load(hoffile)
		hoffile.close()
		test.cf = cf
		test.runtest(gen_hof[-testnum:], class_hof[-testnum:], ref_data)
	else:
		if len(args) > 0:
			runtimes = int(args[1])
		else:
			runtimes = cf['runtimes']

		times = 0
		class_pop = None
		gen_pop = None
		gen_hof = None
		class_hof = None
		if os.path.exists(dirname + "/breed.pickle"):
			popsfile = open(dirname + "/breed.pickle")
			(gen_pop, class_pop) = cPickle.load(popsfile)
			popsfile.close()
		else:
			gen_pop = [hgcpp.Generator() for i in xrange(cf['pop_size'])]
			class_pop = [hgcpp.Classifier() for i in xrange(cf['pop_size'])]
			
		if os.path.exists(dirname + "/hof.pickle"):
			hoffile = open(dirname + "/hof.pickle")
			(gen_hof, class_hof) = cPickle.load(hoffile)
			hoffile.close()
		else:
			gen_hof = [hgcpp.Generator()]
			class_hof = [hgcpp.Classifier()]
		
		print 0
		t0 = time.time()
		for i in xrange(runtimes):
			(class_pop, c) = hgcpp.evo_step(class_pop, gen_hof)
			t1 = time.time()
			print "Classifier HOF member found after " + str(t1 - t0) + " seconds"
			t0 = t1
			class_hof = [c] + class_hof
			(gen_pop, g) = hgcpp.evo_step(gen_pop, class_hof)		
			t1 = time.time()
			print "Generator HOF member found after " + str(t1 - t0) + " seconds"
			t0 = t1
			gen_hof = [g] + gen_hof
			test.teststep(g,c, ref_data)
			print i + 1
			
			
		final_file = open(dirname + "/breed.pickle", 'w')
		cPickle.dump((gen_pop, class_pop), final_file, protocol = cPickle.HIGHEST_PROTOCOL)
		final_file.close()
		final_file = open(dirname + "/hof.pickle", 'w')
		cPickle.dump((gen_hof, class_hof), final_file, protocol = cPickle.HIGHEST_PROTOCOL)
		final_file.close()
		cf.write()

			
