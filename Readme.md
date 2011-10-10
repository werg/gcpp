# Generator-Classifier Predator-Prey Model Co-Evolution Simulation Environment
(c) 2009- G.Pickard

## Prerequisites:

NumPy - http://numpy.scipy.org/
Python NLTK - http://www.nltk.org
Python ConfigObj - http://configobj.sf.net  (Is included in source distribution)

## Usage:

 python control.py [-t] NUMBER DIRECTORY_NAME

e.g.
 python control.py 100 long_run

Will run evolution for 100 generations in the sub-folder "long_run".
If "long_run" already exists, control.py looks for a config file,
takes parameters from there and evolution picks up with the final generation
any given previous runs.

If long_run does not exist, the directory is created.

One can also omit DIRECTORY_NAME in the call above, this will create
a new directory with a rather longish time-stamp. Better ideas for naming
these folders are very welcome. Possibly use seconds since the epoch.

 python control.py -t 50 long_run

Will test the 51st generation in the folder "long_run" in a rudimentary
manner of fashion.

It will output stuff like this:

yk;*;;;*;;**;**
-----+---------
r-all charge of
+++++++++++-+++
.kh*,-**k-h*klm
+++-++++++++-++
 the election ,

The gibberish is obviously from the generator, parseable output from the corpus,
the plus- and minus below each line mark the classifier's verdict (plus should
be reference data, minus should be generator output).

## Parameters:


Parameter files can be put in subfolders to run evolution in and should be
called 'config.cfg'. An commented example 'example.cfg' is provided in main folder.
'configspec.cfg' contains type information, range and default values of
parameters.
