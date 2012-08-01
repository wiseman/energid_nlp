# Copyright 2012 Energid Technologies

import unittest

from energid_nlp import logic


def cook_bindings(b):
  """Converts a set of bindings made of strings (e.g., {"?x":
  "Petunia"}) to one containing Expr's (e.g., {logic.expr("?x"):
  logic.expr("Petunia")}.  Saves a lot of wordiness in the tests.
  """
  if b is False:
    return b
  bindings = {}
  for var in b:
    bindings[logic.expr(var)] = logic.expr(b[var])
  return bindings


def descriptify(a):
  if isinstance(a, str):
    return logic.Description(a)
  elif isinstance(a, logic.Expr):
    return descriptify(a.op)
  elif isinstance(a, logic.Description):
    newslots = {}
    for slot in a.slots:
      newslots[slot] = descriptify(a.slots[slot])
    return logic.Description(a.base, newslots)
  else:
    return a


def parse_equal(a, b):
  return descriptify(a) == descriptify(b)


class KBTestCase(unittest.TestCase):
  def assertBindingsEqual(self, a, b):
    """Checks that two sets of bindings are equivalent."""
    bindingsa = cook_bindings(a)
    bindingsb = cook_bindings(b)
    self.assertEqual(bindingsa, bindingsb)

  def assertAllBindingsEqual(self, a, b):
    """Checks that two sets of solutions (i.e. collections of
    bindings) are equivalent.
    """
    if a is False:
      self.failUnless(b is False, '%s != %s' % (a, b))
    if b is False:
      self.failUnless(a is False, '%s != %s' % (a, b))

    bindings_a = map(cook_bindings, a)
    bindings_b = map(cook_bindings, b)
    for x in bindings_a:
      if not x in bindings_b:
        self.fail('%s != %s' % (bindings_a, bindings_b))
    for x in bindings_b:
      if not x in bindings_a:
        self.fail('%s != %s' % (bindings_a, bindings_b))

  def assertParseEqual(self, parses, results):
    self.failUnless(len(parses) == len(results),
                    '%s != %s' % (parses, results))
    for (parse, result) in zip(parses, results):
      self.failUnless(
        parse_equal(parse, result),
        '%s != %s (in particular, %s != %s)' % (
          parses, results, parse, result))

  def assertApproxEqual(self, a, b, epsilon):
    if abs(a - b) > epsilon:
      self.fail('%s is not within %s of %s' % (a, epsilon, b))
