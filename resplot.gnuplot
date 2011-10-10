set terminal png
plot 'accuracy'   with lines title "generator", 'baseline'  with lines title "random", 'levenacc'  with lines title "levenshtein", 'correct'  with lines title "refdata"
