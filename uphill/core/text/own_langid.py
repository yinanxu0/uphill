#!/usr/bin/env python
"""
langid.py - 
Language Identifier by Marco Lui April 2011

Based on research by Marco Lui and Tim Baldwin.

Copyright 2011 Marco Lui <saffsd@gmail.com>. All rights reserved.

Redistribution and use in source and binary forms, with or without modification, are
permitted provided that the following conditions are met:

   1. Redistributions of source code must retain the above copyright notice, this list of
      conditions and the following disclaimer.

   2. Redistributions in binary form must reproduce the above copyright notice, this list
      of conditions and the following disclaimer in the documentation and/or other materials
      provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDER ``AS IS'' AND ANY EXPRESS OR IMPLIED
WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND
FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

The views and conclusions contained in the software and documentation are those of the
authors and should not be interpreted as representing official policies, either expressed
or implied, of the copyright holder.
"""
from __future__ import print_function


# Defaults for inbuilt server
HOST = None #leave as none for auto-detect
PORT = 9008
FORCE_WSGIREF = False
NORM_PROBS = False # Normalize output probabilities.

# NORM_PROBS defaults to False for a small speed increase. It does not
# affect the relative ordering of the predicted classes. It can be 
# re-enabled at runtime - see the readme.

import base64
import bz2
import sys
import numpy as np
from collections import defaultdict
from pathlib import Path
try:
  from cPickle import loads
except ImportError:
  from pickle import loads


from uphill import loggerx

root_dir = Path(__file__).parent
model = open(root_dir/"lang.model", "rb").readline().strip()

# Convenience methods defined below will initialize this when first called.
identifier = None

def set_languages(langs=None):
  """
  Set the language set used by the global identifier.

  @param langs a list of language codes
  """
  global identifier
  if identifier is None:
    load_model()

  return identifier.set_languages(langs)


def classify(instance):
  """
  Convenience method using a global identifier instance with the default
  model included in langid.py. Identifies the language that a string is 
  written in.

  @param instance a text string. Unicode strings will automatically be utf8-encoded
  @returns a tuple of the most likely language and the confidence score
  """
  global identifier
  if identifier is None:
    load_model()

  return identifier.classify(instance)

def rank(instance):
  """
  Convenience method using a global identifier instance with the default
  model included in langid.py. Ranks all the languages in the model according
  to the likelihood that the string is written in each language.

  @param instance a text string. Unicode strings will automatically be utf8-encoded
  @returns a list of tuples language and the confidence score, in descending order
  """
  global identifier
  if identifier is None:
    load_model()

  return identifier.rank(instance)
  
def cl_path(path):
  """
  Convenience method using a global identifier instance with the default
  model included in langid.py. Identifies the language that the file at `path` is 
  written in.

  @param path path to file
  @returns a tuple of the most likely language and the confidence score
  """
  global identifier
  if identifier is None:
    load_model()

  return identifier.cl_path(path)

def rank_path(path):
  """
  Convenience method using a global identifier instance with the default
  model included in langid.py. Ranks all the languages in the model according
  to the likelihood that the file at `path` is written in each language.

  @param path path to file
  @returns a list of tuples language and the confidence score, in descending order
  """
  global identifier
  if identifier is None:
    load_model()

  return identifier.rank_path(path)

def load_model(path = None):
  """
  Convenience method to set the global identifier using a model at a
  specified path.

  @param path to model
  """
  global identifier
  loggerx.debug('initializing identifier')
  if path is None:
    identifier = LanguageIdentifier.from_modelstring(model)
  else:
    identifier = LanguageIdentifier.from_modelpath(path)

class LanguageIdentifier(object):
  """
  This class implements the actual language identifier.
  """

  @classmethod
  def from_modelstring(cls, string, *args, **kwargs):
    b = base64.b64decode(string)
    z = bz2.decompress(b)
    model = loads(z)
    nb_ptc, nb_pc, nb_classes, tk_nextmove, tk_output = model
    nb_numfeats = int(len(nb_ptc) / len(nb_pc))

    # reconstruct pc and ptc
    nb_pc = np.array(nb_pc)
    nb_ptc = np.array(nb_ptc).reshape(nb_numfeats, len(nb_pc))
   
    return cls(nb_ptc, nb_pc, nb_numfeats, nb_classes, tk_nextmove, tk_output, *args, **kwargs)

  @classmethod
  def from_modelpath(cls, path, *args, **kwargs):
    with open(path) as f:
      return cls.from_modelstring(f.read().encode(), *args, **kwargs)

  def __init__(self, nb_ptc, nb_pc, nb_numfeats, nb_classes, tk_nextmove, tk_output,
               norm_probs = NORM_PROBS):
    self.nb_ptc = nb_ptc
    self.nb_pc = nb_pc
    self.nb_numfeats = nb_numfeats
    self.nb_classes = nb_classes
    self.tk_nextmove = tk_nextmove
    self.tk_output = tk_output

    if norm_probs:
      def norm_probs(pd):
        """
        Renormalize log-probs into a proper distribution (sum 1)
        The technique for dealing with underflow is described in
        http://jblevins.org/log/log-sum-exp
        """
        # Ignore overflow when computing the exponential. Large values
        # in the exp produce a result of inf, which does not affect
        # the correctness of the calculation (as 1/x->0 as x->inf). 
        # On Linux this does not actually trigger a warning, but on 
        # Windows this causes a RuntimeWarning, so we explicitly 
        # suppress it.
        with np.errstate(over='ignore'):
          pd = (1/np.exp(pd[None,:] - pd[:,None]).sum(1))
        return pd
    else:
      def norm_probs(pd):
        return pd

    self.norm_probs = norm_probs

    # Maintain a reference to the full model, in case we change our language set
    # multiple times.
    self.__full_model = nb_ptc, nb_pc, nb_classes

  def set_languages(self, langs=None):
    loggerx.debug("restricting languages to: %s", langs)

    # Unpack the full original model. This is needed in case the language set
    # has been previously trimmed, and the new set is not a subset of the current
    # set.
    nb_ptc, nb_pc, nb_classes = self.__full_model

    if langs is None:
      self.nb_classes = nb_classes 
      self.nb_ptc = nb_ptc
      self.nb_pc = nb_pc

    else:
      # We were passed a restricted set of languages. Trim the arrays accordingly
      # to speed up processing.
      for lang in langs:
        if lang not in nb_classes:
          raise ValueError("Unknown language code %s" % lang)

      subset_mask = np.fromiter((l in langs for l in nb_classes), dtype=bool)
      self.nb_classes = [ c for c in nb_classes if c in langs ]
      self.nb_ptc = nb_ptc[:,subset_mask]
      self.nb_pc = nb_pc[subset_mask]

  def instance2fv(self, text):
    """
    Map an instance into the feature space of the trained model.
    """
    if (sys.version_info > (3, 0)):
      # Python3
      if isinstance(text,str):
        text = text.encode('utf8')
    else:
      # Python2
      if isinstance(text,unicode):
        text = text.encode('utf8')
      # Convert the text to a sequence of ascii values
      text = map(ord, text)

    arr = np.zeros((self.nb_numfeats,), dtype='uint32')

    # Count the number of times we enter each state
    state = 0
    statecount = defaultdict(int)
    for letter in text:
      state = self.tk_nextmove[(state << 8) + letter]
      statecount[state] += 1

    # Update all the productions corresponding to the state
    for state in statecount:
      for index in self.tk_output.get(state, []):
        arr[index] += statecount[state]

    return arr

  def nb_classprobs(self, fv):
    # compute the partial log-probability of the document given each class
    pdc = np.dot(fv,self.nb_ptc)
    # compute the partial log-probability of the document in each class
    pd = pdc + self.nb_pc
    return pd

  def classify(self, text):
    """
    Classify an instance.
    """
    fv = self.instance2fv(text)
    probs = self.norm_probs(self.nb_classprobs(fv))
    cl = np.argmax(probs)
    conf = float(probs[cl])
    pred = str(self.nb_classes[cl])
    return pred, conf

  def rank(self, text):
    """
    Return a list of languages in order of likelihood.
    """
    fv = self.instance2fv(text)
    probs = self.norm_probs(self.nb_classprobs(fv))
    return [(str(k),float(v)) for (v,k) in sorted(zip(probs, self.nb_classes), reverse=True)]

  def cl_path(self, path):
    """
    Classify a file at a given path
    """
    with open(path) as f:
      retval = self.classify(f.read())
    return path, retval

  def rank_path(self, path):
    """
    Class ranking for a file at a given path
    """
    with open(path) as f:
      retval = self.rank(f.read())
    return path, retval
