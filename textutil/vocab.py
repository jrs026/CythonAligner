#!/usr/bin/env python
"""
Stores a mapping between words and integers for efficiently storing text.
"""

from textutil.normalizer import Normalizer

class Vocab:

  def __init__(self, normalizer=Normalizer()):
    self.normalizer = normalizer
    self.word_to_id = {}
    # id_to_word[0] is reserved as the special OOV token
    self.id_to_word = ['__OOV__']

  def word_lookup(self, word):
    """
    Return the id of the given word, or 0 if not found.
    """
    return self.word_to_id.get(self.normalizer.normalize(word), 0)

  def id_lookup(self, word_id):
    """
    Return the word for a given id, or the OOV token if the id is out of
    range.
    """
    if word_id > len(self.id_to_word):
      return self.id_to_word[0]
    else:
      return self.id_to_word[word_id]

  def add_word(self, word):
    """
    Add a word to the vocabulary (if it doesn't already exist) and return its
    id.
    """
    processed_word = self.normalizer.normalize(word)
    word_id = self.word_to_id.get(processed_word, -1)
    if word_id == -1:
      self.id_to_word.append(processed_word)
      word_id = len(self.id_to_word) - 1
      self.word_to_id[processed_word] = word_id
    return word_id
