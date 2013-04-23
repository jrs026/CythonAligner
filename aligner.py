#!/usr/bin/python

import copy
import math
import os
import re
import sys

from optparse import OptionParser

import sentence_pair_extractor

def main():
  parser = OptionParser()
  default_base_dir = os.environ.get("MT_DATA_DIR")

  parser.add_option("--prefix", dest="base_dir", default=default_base_dir,
      help="This path will be appended to the beginning of every file location")

  parser.add_option("-c", "--comparable_data", dest="comp_data",
      default="comparable/annotated/wiki_es_en_dev",
      help="Annotated comparable data, expecting \".source\", \".target\", and \".alignment\"")

  parser.add_option("-t", "--t-table", dest="m1_data",
      default="models/t_tables/europarl_es_en.lex",
      help="Word alignment parameters from some parallel data")

  # ----------------

  parser.add_option("-p", "--parallel_data", dest="training_file",
      default="data/euro_esen_10k",
      help="Parallel data, expecting \".source\" and \".target\"")

  parser.add_option("-r", "--raw-data", dest="raw_data", default="",
      help="Raw comparable data, expecting \".source\" and \".target\"")

  parser.add_option("-d", "--dictionary", dest="dictionary",
      default="", help="Location of bilingual dictionary")

  parser.add_option("-e", "--example-window", dest="example_window", type="int",
      default=3, help="Size of the example window for gathering training data")

  parser.add_option("--length-ratio", type="float", dest="max_len_ratio",
      default=3.0, 
      help="Maximum length ratio for sentences to be considered parallel")

  parser.add_option("--test-max", type="int", dest="test_max", default=100,
      help="Number of sentences from the parallel data to use as test data")

  parser.add_option("--prob-floor", type="float", dest="prob_floor",
      default=1e-4, help="Lowest probability value for LM and M1")

  parser.add_option("--max_iterations", type="int", dest="max_iterations",
      default=25, help="Maximum number of L-BFGS iterations")

  parser.add_option("--l2-norm", type="float", dest="l2_norm", default="2.0",
      help="L2 normalizing value for the Maxent model")

  (opts, args) = parser.parse_args()
  spe = sentence_pair_extractor.SentencePairExtractor(opts)

  # Read available data
  if opts.comp_data:
    if opts.base_dir is not None:
      opts.comp_data = opts.base_dir + "/" + opts.comp_data
    (source_docs, target_docs, alignments) = spe.read_comp_data(opts.comp_data)
  if opts.m1_data:
    if opts.base_dir is not None:
      opts.m1_data = opts.base_dir + "/" + opts.m1_data
    spe.read_m1_probs(opts.m1_data)

  spe.init_feature_functions()

  if opts.comp_data:
    print "Performance on comparable data:"
    annotated_comp_data = spe.create_annotated_comp_data(
        source_docs, target_docs, alignments)
    print_feature_stats(annotated_comp_data)
    folds = range(5)
    for fold in folds:
      comp_test_data = []
      comp_train_data = []
      train_true = 0
      test_true = 0
    
      print "\nFold " + str(fold+1) + ":"
      for i,event in enumerate(annotated_comp_data):
        if i % len(folds) == fold:
          comp_test_data.append(event)
          if event[1] == 'true':
            test_true += 1
        else:
          comp_train_data.append(event)
          if event[1] == 'true':
            train_true += 1
      print "Positive training examples:", train_true
      print "Positive test examples:", test_true
      spe.train_model(comp_train_data)
      print spe.me.get_features()
      for threshold in drange(0.1, 1.0, 0.1):
        print threshold, spe.test_model(comp_test_data, threshold)

def print_feature_stats(train_data):
  count_by_outcome = {}
  feature_stats = {}
  feature_names = {}
  for (context, outcome) in train_data:
    if not count_by_outcome.get(outcome):
      count_by_outcome[outcome] = 0
    count_by_outcome[outcome] += 1
    for (name, value) in context:
      if not feature_names.get(name):
        feature_names[name] = True
      if not feature_stats.get(name + ' ' + outcome):
        feature_stats[name + ' ' + outcome] = [0.0, 0.0]
      feature_stats[name + ' ' + outcome][0] += value
      feature_stats[name + ' ' + outcome][1] += value * value

  for name in feature_names.keys():
    print name + ":"
    for outcome in count_by_outcome.keys():
      total = count_by_outcome[outcome]
      if (feature_stats.get(name + ' ' + outcome)):
        mean = feature_stats[name + ' ' + outcome][0] / total
        variance = (feature_stats[name + ' ' + outcome][1] / total) - (mean * mean)
        print outcome, ':', mean, variance

def drange(start, stop, step):
  r = start
  while r < stop:
    yield r
    r += step

if __name__ == "__main__":
  main()
