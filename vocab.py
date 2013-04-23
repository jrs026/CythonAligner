#!/usr/bin/python
import re
# Stores maps between words and integers for efficiently storing text

class Vocab:

  def __init__(self, lowercase=False):
    self.lowercase = lowercase
    self.word_to_id = {}
    # id_to_word[0] is reserved as the special OOV token
    self.id_to_word = ['__OOV__']

  def word_lookup(self, word):
    """Return the id of the given word, or 0 if not found."""
    if self.lowercase:
      return self.word_to_id.get(word.lower(), 0)
    else:
      return self.word_to_id.get(word, 0)

  def id_lookup(self, word_id):
    """Return the word for a given id, or the OOV token if the id is out of
    range.
    """
    if word_id > len(self.id_to_word):
      return self.id_to_word[0]
    else:
      return self.id_to_word[word_id]

  def add_word(self, word):
    """Add a word to the vocabulary (if it doesn't already exist) and return its
    id.
    """
    if self.lowercase:
      word = word.lower()
    word_id = self.word_to_id.get(word, -1)
    if word_id == -1:
      self.id_to_word.append(word)
      word_id = len(self.id_to_word) - 1
      self.word_to_id[word] = word_id
    return word_id
