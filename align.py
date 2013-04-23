#!/usr/bin/env python
"""
The interface for the sentence alignment tools
"""
# align.py
#
# 

import copy
import math
import os
import re
import sys

from optparse import OptionParser

import sentence_pair_extractor

def main():
  parser = OptionParser()

  default_source = "ko"
  default_target = "en"
  default_base_dir = "/Users/jrsmith/mt_data/ko_en"

  parser.add_option("--prefix", dest="base_dir", default=default_base_dir,
      help="This path will be appended to the beginning of every file location")
  parser.add_option("-s", "--source", dest="source", default=default_source,
      help="Source language (two letter code)")
  parser.add_option("-t", "--target", dest="target", default=default_target,
      help="Target language (two letter code)")
  parser.add_option("-l", "--lex", dest="lex_prefix", default="lex.1",
      help="Prefix for Moses' lexical probability files (\".e2f\" and \".f2e\")")
  parser.add_option("-p", "--parallel", dest="parallel", default="",
      help="Prefix for the parallel data used for training/testing")

  parser.add_option("-f", "--folds", dest="folds", default=5, type="int",
      help="The number of folds for cross-validation")

  # Options from the old code:
  parser.add_option("-e", "--example-window", dest="example_window", type="int",
      default=3, help="Size of the example window for gathering training data")
  parser.add_option("--length-ratio", type="float", dest="max_len_ratio",
      default=3.0, 
      help="Maximum length ratio for sentences to be considered parallel")
  parser.add_option("--prob-floor", type="float", dest="prob_floor",
      default=1e-4, help="Lowest probability value for t-table")
  parser.add_option("--max_iterations", type="int", dest="max_iterations",
      default=25, help="Maximum number of L-BFGS iterations")
  parser.add_option("--l2-norm", type="float", dest="l2_norm", default="2.0",
      help="L2 normalizing value for the Maxent model")

  (opts, args) = parser.parse_args()

  spe = sentence_pair_extractor.SentencePairExtractor(opts)

  # Load the lexical probabilities if available
  if opts.lex_prefix:
    pst_filename = os.path.join(opts.base_dir, opts.lex_prefix + ".e2f")
    pts_filename = os.path.join(opts.base_dir, opts.lex_prefix + ".f2e")
    spe.read_lex_probs(pst_filename, pts_filename)

  spe.init_feature_functions()

  # Train and test a model on parallel data
  if opts.parallel:
    source_filename = os.path.join(opts.base_dir,
        opts.parallel + "." + opts.source)
    target_filename = os.path.join(opts.base_dir,
        opts.parallel + "." + opts.target)
    source_sents = spe.read_sentences(source_filename)
    target_sents = spe.read_sentences(target_filename)
    annotated_data = spe.create_multiclass_parallel_data(
        source_sents, target_sents)

    for fold in xrange(opts.folds):
      train_data, test_data = [], []
      print "\n Fold %d:" % (fold+1)
      for i, event in enumerate(annotated_data):
        if i % opts.folds == fold:
          test_data.append(event)
        else:
          train_data.append(event)
      spe.train_multi(train_data)
      print spe.me.get_features()
      for threshold in drange(0.0, 1.0, 0.1):
        print threshold, spe.test_multi(test_data, threshold)

    #print_feature_stats(annotated_data)

def print_feature_stats(train_data):
  count_by_outcome = {"True" : 0, "False": 0}
  feature_stats = {}
  feature_names = {}
  for instance_set in train_data:
    count_by_outcome["True"] += 1
    count_by_outcome["False"] += len(instance_set.instances) - 1
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
