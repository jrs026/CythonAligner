#!/usr/bin/env python
"""
A class for normalizing text in various ways.
"""

import regex
import unicodedata

class Normalizer:

  def __init__(self):
    """
    The default behavior is to perform Unicode normalization and lowercase only.
    """
    # Options
    self.lowercase = True
    self.unicode_norm = "NFKC"
    self.norm_digits = False

    # Compiled regexes
    self.digit_re = regex.compile(r"\p{N}", regex.U)

  def normalize(self, text):
    text = unicodedata.normalize(self.unicode_norm, text)
    if self.lowercase:
      text = text.lower()
    if self.norm_digits:
      text = self.digit_re.sub("0", text)
    
    return text

  def set_lowercase(self, value):
    """
    Set lowercasing on or off (value should be true or false).
    """
    self.lowercase = value

  def set_unicode_normalization(self, value):
    """
    Set the type of Unicode normalization (value should be "NFC", "NFKC", "NFD",
    or "NFKD".
    """
    self.unicode_norm = value

  def set_digit_normalization(self, value):
    """
    Set digit normalization on or off (value should be true or false).
    This will map all numeric characters to 0.
    """
    self.norm_digits = value
