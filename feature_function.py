import math

# TODO: Update LexicalFeatures, change from the old feature representation to
# the new one.

class FeatureFunction(object):
  """A generic class for extracting features from a sentence pair."""

  def __init__(self, opts):
    self.opts = opts

  def get_features(self, source, target):
    return []

  @staticmethod
  def poisson_prob(mean, observed):
    p = math.exp(-mean)
    for i in xrange(observed):
      p *= mean
      p /= i+1
    return p

class DummyFeature(FeatureFunction):
  """Always extracts a dummy feature with value 1.0."""

  def __init__(self, opts):
    super(DummyFeature, self).__init__(opts)

  def get_features(self, source, target):
    return [('bias', 1.0)]

class LengthFeatures(FeatureFunction):
  """Extracts features only based on the lengths of the two sentences."""

  def __init__(self, opts):
    super(LengthFeatures, self).__init__(opts)

  def get_features(self, source, target):
    source_len = len(source)
    target_len = len(target)
    if source_len == 0 or target_len == 0:
      # TODO: have some reasonable values
      return []
    len_ratio = float(target_len) / float(source_len)
    if len_ratio < 1.0:
      len_ratio = 1.0 / len_ratio
    poisson_length = self.poisson_prob(source_len, target_len)

    context = []
    context.append(('poisson_length', math.log(poisson_length)))
    context.append(('ratio', len_ratio - 1.0))
    return context

class DictionaryFeatures(FeatureFunction):
  """Extracts bag-of-words features after a projection through a bilingual
  dictionary.
  """

  def __init__(self, opts, dictionary):
    super(DictionaryFeatures, self).__init__(opts)
    self.dictionary = dictionary

  def get_features(self, source, target):
    proj_source = self.project(source)
    context = []
    context.append(('cosine_sim', self.cosine_sim(target, proj_source)))
    return context

  def cosine_sim(self, list_vec, hash_vec):
    """Computes the cosine similarity between two vectors represented by a list
    and a hash.
    """
    # Sum of squares:
    list_sos = len(list_vec)
    hash_sos = 0.0
    for key,val in hash_vec.iteritems():
      hash_sos += val**2
    denominator = math.sqrt(list_sos) * math.sqrt(hash_sos)

    numerator = 0.0
    for x in list_vec:
      numerator += hash_vec.get(x, 0.0)

    if denominator > 0.0:
      return numerator / denominator
    else:
      return 0.0

  def project(self, source):
    """Project the source sentence through the dictionary and return the
    bag-of-words vector in the target space.
    """
    proj_source = {}
    for s in source:
      s_proj = self.dictionary.get(s, [])
      for t in s_proj:
        if not t in proj_source:
          proj_source[t] = 0.0
        proj_source[t] += 1.0

    return proj_source

class LexicalFeatures(FeatureFunction):
  """Extracts features based on lexical probabilities. Requres t-tables in the
  source to target and target to source directions."""

  def __init__(self, opts, pst, pts):
    super(LexicalFeatures, self).__init__(opts)
    self.pst = pst
    self.pts = pts

  # TODO: source / target may be mixed up - fix
  def get_features(self, source, target):
    #cov_vals = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5,
    #    0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
    cov_vals = [0.1, 0.2, 0.3]
    target_cov = {}
    source_cov = {}
    for v in cov_vals:
      target_cov[v] = 0.0
      source_cov[v] = 0.0
    ts_max_prob, ts_total_prob = 0.0, 0.0
    for t in target:
      ts_score, max_ts_score = 0.0, 0.0
      for s in source:
        prob = self.pts.get((t, s), self.opts.prob_floor)
        if s == t:
          prob = 1.0
        ts_score += prob
        if (prob > max_ts_score):
          max_ts_score = prob

      ts_max_prob += max_ts_score
      ts_total_prob += ts_score / len(source)
      for v in cov_vals:
        if (max_ts_score > v):
          target_cov[v] += 1
    st_max_prob, st_total_prob = 0.0, 0.0
    for s in source:
      st_score, max_st_score = 0.0, 0.0
      for t in target:
        prob = self.pst.get((s, t), self.opts.prob_floor)
        if s == t:
          prob = 1.0
        st_score += prob
        if (prob > max_st_score):
          max_st_score = prob
      
      st_max_prob += max_st_score
      st_total_prob += st_score / len(target)
      for v in cov_vals:
        if (max_st_score > v):
          source_cov[v] += 1

    context = []

    for v in cov_vals:
      context.append(("target_cov_" + str(v), target_cov[v] / len(target)))
      context.append(("source_cov_" + str(v), source_cov[v] / len(source)))

    context.append(("t|s_max_prob", ts_max_prob / len(target)))
    context.append(("t|s_total_prob", ts_total_prob / len(target)))
    context.append(("s|t_max_prob", st_max_prob / len(source)))
    context.append(("s|t_total_prob", st_total_prob / len(source)))

    return context
