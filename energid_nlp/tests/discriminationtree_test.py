# Copyright 2012 Energid Technologies

import unittest

from energid_nlp import discriminationtree


def is_variable(symbol):
  return symbol == '?'


class DiscriminationTreeTests(unittest.TestCase):
  def test_trivial_tree(self):
    """Trivial tests."""
    tree = discriminationtree.make_root_discrimination_tree(is_variable)
    self.assertEqual([], list(tree.retrieve('a')))
    self.assertEqual([], list(tree.retrieve('abc')))
    self.assertEqual([], list(tree.retrieve('abd')))
    self.assertEqual([], list(tree.retrieve('acz')))
    tree.erase('abc')
    tree.erase('b')
    tree.erase('a')
    self.assertRaises(
      discriminationtree.Error,
      lambda: discriminationtree.DiscriminationTree('ROOT'))

  def test_simple_retrieval(self):
    """Test simple retrieval."""
    tree = discriminationtree.make_root_discrimination_tree(is_variable)
    tree.put('abc', 1)
    self.assertEqual([1], list(tree.retrieve('abc')))
    self.assertEqual([], list(tree.retrieve('a')))
    tree.put('abd', 2)
    self.assertEqual([1], list(tree.retrieve('abc')))
    self.assertEqual([], list(tree.retrieve('a')))
    self.assertEqual([2], list(tree.retrieve('abd')))

  def test_var_retrieval(self):
    """Test variable retrieval."""
    tree = discriminationtree.make_root_discrimination_tree(is_variable)
    tree.put('abc', 1)
    tree.put('abd', 2)
    tree.put('abde', 3)
    self.assertEqual([1], list(tree.retrieve('a?c')))
    self.assertEqual([3], list(tree.retrieve('????')))
    self.assertEqual([], list(tree.retrieve('a?')))
    self.assertEqual([], list(tree.retrieve('?')))
    self.assertEqual([], list(tree.retrieve('??')))
    self.assertEqual([1, 2], list(tree.retrieve('???')))
    self.assertEqual([3], list(tree.retrieve('????')))
    tree.put('acde', 4)
    self.assertEqual([2], list(tree.retrieve('a?d')))
    self.assertEqual([3, 4], list(tree.retrieve('a?de')))
    self.assertEqual([3, 4], list(tree.retrieve('a??e')))
    self.assertEqual([3, 4], list(tree.retrieve('????')))

  def test_erase(self):
    """Test erase."""
    tree = discriminationtree.make_root_discrimination_tree(is_variable)
    tree.put('abc', 1)
    tree.put('abd', 2)
    tree.put('abde', 3)
    self.assertEqual([2], list(tree.retrieve('abd')))
    self.assertEqual([3], list(tree.retrieve('abde')))
    tree.erase('abd')
    self.assertEqual([], list(tree.retrieve('abd')))
    self.assertEqual([], list(tree.retrieve('abde')))
    self.assertEqual([1], list(tree.retrieve('abc')))


if __name__ == '__main__':
  unittest.main()
