from optparse import OptionParser
from pickle import load

parser = OptionParser()

(options, args) = parser.parse_args()
directory = args[0]
filename = directory + "/best" + args[1] + ".pickle"
file = open(filename, 'r')
(gen_best, class_best) = load(file)
file.close()

for (g,c) in zip(gen_best, class_best):
    print str(g.size()) + " " + str(c.size())
