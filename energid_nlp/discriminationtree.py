# Copyright Energid Technologies 2012

"""Implementation of discrimination trees, a variant of trie.

Discrimination trees are useful for indexing terms of a propositional
database.  There's lots of info in the book "Term Indexing" by Peter
Graf: http://books.google.com/books?id=Zf1nlcplAAYC
"""

import sys


class Error(Exception):
  pass


class DiscriminationTree:
  "A discrimination tree (also a node in a tree)."
  def __init__(self, index, parent=None, var_test=None):
    if not (parent or var_test):
      raise Error('Need to specify var_test (or parent).')
    if not var_test:
      # Use the parent's var_test if one isn't specified.
      var_test = parent.var_test
    self.index = index
    self.parent = parent
    self.var_test = var_test
    self.leaves = []
    self.interiors = []
    self.ints = {}

  def __str__(self):
    return '<%s %s>' % (self.__class__.__name__, self.index)

  def __repr__(self):
    return self.__str__()

  def retrieve(self, path):
    """Yields all propositions in this tree matching the specified
    path.
    """
    if len(path) == 0:
      for leaf in self.leaves:
        yield leaf
    else:
      next_index = path[0]
      if self._is_variable(next_index):
        for leaf in self._retrieve_variable(path[1:]):
          yield leaf
      else:
        next_disc_tree = self._child_at_index(next_index)
        if next_disc_tree is None:
          return
        else:
          for leaf in next_disc_tree.retrieve(path[1:]):
            yield leaf

  def put(self, path, leaf):
    """Inserts leaf into the tree at the node specified by the
    path.
    """
    if len(path) == 0:
      if isinstance(leaf, DiscriminationTree):
        self.interiors.append(leaf)
        self.ints[leaf.index] = leaf
      else:
        self.leaves.append(leaf)
      return True
    else:
      next_index = path[0]
      next_disc_tree = self._child_at_index(next_index)
      if next_disc_tree is None:
        next_disc_tree = DiscriminationTree(next_index, parent=self)
        self.interiors.append(next_disc_tree)
        self.ints[next_disc_tree.index] = next_disc_tree
      next_disc_tree.put(path[1:], leaf)

  def erase(self, path):
    "Deletes a path from the tree."
    leaf_parent = self._position(path)
    if leaf_parent is not None:
      leaf_parent._erase_at()
    return False

  def dump(self, stream=sys.stdout, indent=0):
    "Dumps a printed representation of a complete tree."
    stream.write('\n' + ' ' * indent + str(self))
    for child in self.leaves:
      stream.write('\n' + ' ' * (indent + 3) + str(child))
    for child in self.interiors:
      child.dump(stream=stream, indent=indent + 3)

  def _erase_at(self):
    "Deletes an entire tree from the parent."
    for child in self.interiors:
      child._erase_at()
    if self.parent is not None:
      self.parent.interiors.remove(self)
      del self.parent.ints[self.index]
    return True

  def _position(self, path):
    if len(path) == 0:
      return self
    else:
      next_disc_tree = self._child_at_index(path[0])
      if next_disc_tree is None:
        return None
      else:
        return next_disc_tree._position(path[1:])

  def _retrieve_variable(self, path):
    for child in self.interiors:
      for leaf in child.retrieve(path):
        yield leaf

  def _child_at_index(self, index):
    "Returns the child node with the specified index, or None."
    return self.ints.get(index, None)

  def _is_variable(self, sym):
    return self.var_test(sym)


def make_root_discrimination_tree(var_test):
  "Creates and returns a brand new root node."
  return DiscriminationTree('ROOT', var_test=var_test)
