import codecs
import copy
import math
import os

import py_maxent

import feature_function
import vocab

class SentencePairExtractor:
  """Contains a classifier for determining whether or not a sentence pair is
  parallel.
  """

  def __init__(self, opts, debug=False):
    """Pass in program options for now."""
    # This class will directly read certain program options:
    # TODO: list them
    self.opts = opts
    # Vocab is a mapping from string to integer used for both the source and
    # target.
    self.vocab = vocab.Vocab()
    # p(s|t) and p(t|s): Lexical probabilities. These are stored as a dict from
    # (int, int) tuples to a float - the integers are vocab indices.
    self.pst = {}
    self.pts = {}
    self.feature_functions = []
    self.debug = debug

  def init_feature_functions(self):
    """Should be called after loading Model1, language model, or other data."""
    self.feature_functions.append(feature_function.DummyFeature(self.opts))
    self.feature_functions.append(feature_function.LengthFeatures(self.opts))
    if len(self.pst) > 0 and len(self.pts) > 0:
      self.feature_functions.append(feature_function.LexicalFeatures(
          self.opts, self.pst, self.pts))

  def get_features(self, source_sent, target_sent):
    """Return a featurized context for a sentence pair based on all feature
    functions.
    """
    if self.debug:
      source_words = []
      for s in source_sent:
        source_words.append(self.vocab.id_lookup(s))
      target_words = []
      for t in target_sent:
        target_words.append(self.vocab.id_lookup(t))
      print "Source:", ' '.join(source_words)
      print "Target:", ' '.join(target_words)
    context = []
    for ff in self.feature_functions:
      context.extend(ff.get_features(source_sent, target_sent))
    return context

  def read_comp_data(self, base_filename):
    """Read comparable document pairs (which may be annotated)."""
    source_docs = self.read_docs(base_filename + ".source")
    target_docs = self.read_docs(base_filename + ".target")
    if os.path.exists(base_filename + ".alignment"):
      alignment_docs = self.read_docs(base_filename + ".alignment")
      return (source_docs, target_docs, alignment_docs)
    else:
      return (source_docs, target_docs)

  def train_model(self, training_data, norm_data=True):
    self.me = py_maxent.PyMaxent(self.opts.l2_norm)
    pos_weight = 1
    if norm_data:
      pos_examples = 0
      neg_examples = 0
      for example in training_data:
        if example[1] == 'true':
          pos_examples += 1
        elif example[1] == 'false':
          neg_examples += 1
      #pos_weight = int(neg_examples / pos_examples)
      pos_weight = 1
      print ("Positive examples:", pos_examples, "Negative examples",
          neg_examples, "Positive example weight:", pos_weight)

    data = []
    for example in training_data:
      instance_set = self.convert_data(example[0], example[1])
      data.append(instance_set)

    self.me.set_training_data(data)
    self.me.lbfgs_train()

  def train_multi(self, training_data):
    self.me = py_maxent.PyMaxent(self.opts.l2_norm)
    self.me.set_training_data(training_data)
    self.me.lbfgs_train()

  def test_multi(self, test_data, threshold):
    total = 0.0
    true_positives = 0.0
    false_positives = 0.0
    total_positives = 0.0
    for instance_set in test_data:
      total_positives += 1 # TODO add neg examples
      probs = self.me.get_probs(instance_set)
      total += len(probs) # Total number of items that could be output
      max_p, max_i = probs[0], 0
      for i, p in enumerate(probs):
        if p > max_p:
          max_p = p
          max_i = i

      if max_i == 0 and probs[0] > threshold:
        true_positives += 1
      elif max_p > threshold:
        false_positives += 1

    correct = true_positives + total - total_positives - false_positives
    
    accuracy = (correct / total) * 100
    precision = 0.0
    if ((true_positives + false_positives) > 0.0):
      precision = (true_positives / (true_positives + false_positives)) * 100
    recall = 0.0
    if (total_positives > 0.0):
      recall = (true_positives / total_positives) * 100
    f1 = 0.0
    if (precision + recall > 0.0):
      f1 = (2 * precision * recall) / (precision + recall)

    return (accuracy, precision, recall, f1)

  def test_model(self, test_data, threshold=0.5):
    """Return accuracy, precision, recall, and f1 using the given classification
    threshold.
    """
    total = 0.0
    true_positives = 0.0
    false_positives = 0.0
    total_positives = 0.0
    for (context, output) in test_data:
      instance_set = self.convert_data(context, output)
      total += 1
      if (output == 'true'):
        total_positives += 1
        if (self.me.get_prob(instance_set, 0) > threshold):
          true_positives += 1
      elif (output == 'false'):
        if (self.me.get_prob(instance_set, 0) > threshold):
          false_positives += 1

    correct = true_positives + total - total_positives - false_positives
    
    accuracy = (correct / total) * 100
    precision = 0.0
    if ((true_positives + false_positives) > 0.0):
      precision = (true_positives / (true_positives + false_positives)) * 100
    recall = 0.0
    if (total_positives > 0.0):
      recall = (true_positives / total_positives) * 100
    f1 = 0.0
    if (precision + recall > 0.0):
      f1 = (2 * precision * recall) / (precision + recall)

    return (accuracy, precision, recall, f1)
    
  def extract_sentences(self, raw_source, raw_target, out_file, threshold=0.5):
    """Extract parallel sentences from source and target comparable
    documents.
    """
    s_out = open(out_file + '.source', 'w')
    t_out = open(out_file + '.target', 'w')
    for i in xrange(0, len(raw_source)):
      for s_sent in raw_source[i]:
        for t_sent in raw_target[i]:
          len_ratio = len(t_sent) / (1.0 * len(s_sent))
          if (len_ratio < 1.0):
            len_ratio = 1.0 / len_ratio
          if (len_ratio < self.opts.max_len_ratio):
            context = get_features(s_sent, t_sent)
            if (self.me.eval(context, 'true') > threshold):
              source_words = []
              for s in s_sent:
                source_words.append(self.vocab.id_lookup(s))
              target_words = []
              for t in t_sent:
                target_words.append(self.vocab.id_lookup(t))
              s_out.write(' '.join(source_words) + "\n")
              t_out.write(' '.join(target_words) + "\n")

    s_out.close()
    t_out.close()

  def create_annotated_parallel_data(self, source_sents, target_sents):
    """Create annotated sentence pair instances from parallel data."""
    instances = []
    for i,source in enumerate(source_sents):
      neg_examples = 0
      for j in range(i - self.opts.example_window, i + self.opts.example_window + 1):
        if j >= 0 and j < len(target_sents) and j != i:
          len_ratio = float(len(target_sents[j])) / float(len(source))
          if (len_ratio < 1.0):
            len_ratio = 1.0 / len_ratio
          if (len_ratio < self.opts.max_len_ratio):
            neg_examples += 1
            context = self.get_features(source, target_sents[j])
            instances.append((context, 'false'))

      true_context = self.get_features(source, target_sents[i])
      instances.append((true_context, 'true'))

    return instances

  def create_multiclass_parallel_data(self, source_sents, target_sents):
    """Create annotated multi-class instances."""
    instances = []
    for i,source in enumerate(source_sents):
      contexts = []
      # The true instance is always the first one
      contexts.append(self.get_features(source, target_sents[i]))
      for j in range(i - self.opts.example_window, i + self.opts.example_window + 1):
        if j >= 0 and j < len(target_sents) and j != i:
          len_ratio = float(len(target_sents[j])) / float(len(source))
          if (len_ratio < 1.0):
            len_ratio = 1.0 / len_ratio
          if (len_ratio < self.opts.max_len_ratio):
            contexts.append(self.get_features(source, target_sents[j]))

      instances.append(self.create_multiclass_instance_set(contexts, 0))

    return instances

  def create_annotated_comp_data(self, source_docs, target_docs, alignments):
    """Create annotated sentence pair instances from comparable data."""
    # Read the alignments into a dict for each document
    a_dicts = []
    for a in alignments:
      a_dict = {}
      for pair in a:
        (s, t) = pair.split()
        a_dict[(int(s), int(t))] = 1.0
      a_dicts.append(a_dict)

    annotated_data = []
    for i in xrange(0, len(source_docs)):
      source_sents = source_docs[i]
      target_sents = target_docs[i]
      a_dict = a_dicts[i]
      for s,s_sent in enumerate(source_sents):
        for t,t_sent in enumerate(target_sents):
          outcome = 'false'
          if (s, t) in a_dict:
            outcome = 'true'
          context = self.get_features(s_sent, t_sent)
          annotated_data.append((context, outcome))
      
    return annotated_data

  def read_lex_probs(self, pst_filename, pts_filename):
    """Read Lexical probabilities (t-tables) in both directions."""
    self.read_lex_file(pst_filename, self.pst)
    self.read_lex_file(pts_filename, self.pts)

  def read_lex_file(self, filename, t_table):
    """Read an individual lexical probability file and update the t-table."""
    for line in codecs.open(filename, mode="r", encoding="utf-8"):
      (s, t, cost) = line.strip().split()
      if float(cost) < self.opts.prob_floor:
        continue
      pair_index = (self.vocab.add_word(s), self.vocab.add_word(t))
      if pair_index in t_table:
        if t_table[pair_index] < float(cost):
          t_table[pair_index] = float(cost)
      else:
        t_table[pair_index] = float(cost)

  @staticmethod
  def create_multiclass_instance_set(contexts, outcome=None):
    """Create a multi-way classification instance where outcome is the index of
    the correct instance (if available)."""
    instance_set = py_maxent.PyInstanceSet()
    if outcome != None:
      instance_set.true_instance = outcome
    else:
      instance_set.true_instance = -1
    instance_set.instances = contexts
    return instance_set

  @staticmethod
  def convert_data(context, outcome=None):
    """Temporary function to convert data from the old format to the new one."""
    instance_set = py_maxent.PyInstanceSet()
    if outcome == 'true':
      instance_set.true_instance = 0
    elif outcome == 'false':
      instance_set.true_instance = 1
    else:
      instance_set.true_instance = -1
    parallel_instance = []
    nonparallel_instance = []
    for (name, weight) in context:
      parallel_instance.append((name + "_t", weight))
      nonparallel_instance.append((name + "_f", weight))
    instance_set.instances.append(parallel_instance)
    instance_set.instances.append(nonparallel_instance)
    return instance_set

  def read_docs(self, filename):
    """Read documents (or alignments) separated by blank lines"""
    docs = []
    current_doc = []
    for line in codecs.open(filename, mode="r", encoding="utf-8"):
      if len(line.strip()) == 0:
        if len(current_doc) > 0:
          docs.append(copy.deepcopy(current_doc))
          current_doc = []
      else:
        current_sent = []
        for token in line.split():
          current_sent.append(self.vocab.add_word(token))
        current_doc.append(current_sent)

    if len(current_doc) > 0:
      docs.append(current_doc)
    return docs

  def read_sentences(self, filename):
    """Read text from the file into arrays of vocabulary entries."""
    sents = []
    for line in codecs.open(filename, mode="r", encoding="utf-8"):
      current_sent = []
      for token in line.split():
        current_sent.append(self.vocab.add_word(token))
      sents.append(current_sent)
    return sents
