# Copyright Energid Technologies 2012

"""Utilities based on utils.py from the code in Artificial
Intelligence: A Modern Approach (AIMA), see
http://code.google.com/p/aima-python/source/browse/trunk/utils.py

Many are inspired by similar functions in Common Lisp.
"""

import operator


def is_number(x):
  """Is x a number? We say it is if it has a __int__ method."""
  # Note that NaN (i.e. float('NaN')) has an __int__ method that
  # throws an exception.  Not sure if that matters.
  return hasattr(x, '__int__')


def is_sequence(x):
  """Is x a sequence? We say it is if it has a __getitem__ method."""
  return hasattr(x, '__getitem__')


def num_or_str(x):
  """The argument is a string; convert to a number if possible, or strip it.
  >>> num_or_str('42')
  42
  >>> num_or_str(' 42x ')
  '42x'
  """
  if is_number(x):
    return x
  try:
    return int(x)
  except ValueError:
    try:
      return float(x)
    except ValueError:
      return str(x).strip()


def if_(test, result, alternative):
  """Like C++ and Java's (test ? result : alternative), except
  both result and alternative are always evaluated. However, if
  either evaluates to a function, it is applied to the empty arglist,
  so you can delay execution by putting it in a lambda.
  >>> if_(2 + 2 == 4, 'ok', lambda: expensive_computation())
  'ok'
  """
  if test:
    if callable(result):
      return result()
    return result
  else:
    if callable(alternative):
      return alternative()
    return alternative


# Functions on Sequences (mostly inspired by Common Lisp) NOTE:
# Sequence functions (count_if, find_if, every, some) take function
# argument first (like reduce, filter, and map).

def remove_all(item, seq):
  """Return a copy of seq (or string) with all occurences of item removed.
  >>> remove_all(3, [1, 2, 3, 3, 2, 1, 3])
  [1, 2, 2, 1]
  >>> remove_all(4, [1, 2, 3])
  [1, 2, 3]
  """
  if isinstance(seq, str):
    return seq.replace(item, '')
  else:
    return [x for x in seq if x != item]


def unique(seq):
  """Remove duplicate elements from seq. Assumes hashable elements.
  >>> unique([1, 2, 3, 2, 1])
  [1, 2, 3]
  """
  return list(set(seq))


# Some set-like functions that accept test arguments, Common Lisp-style.

def element_of(elt, sequence, test_fn=operator.eq):
  for e in sequence:
    if test_fn(elt, e):
      return True
  return False


def intersection(s1, s2, test_fn):
  result = []
  for e1 in s1:
    if element_of(e1, s2, test_fn):
      result.append(e1)
  return result


def set_difference(s1, s2, test_fn):
  result = []
  for e1 in s1:
    if not element_of(e1, s2, test_fn):
      result.append(e1)
  return result


# Does not require hashability of elements like unique, above, does.

def remove_duplicates(seq, test_fn=lambda a, b: a == b):
  result = []
  for e in seq:
    if not element_of(e, result, test_fn):
      result.append(e)
  return result


def product(numbers):
  """Return the product of the numbers.
  >>> product([1,2,3,4])
  24
  """
  return reduce(operator.mul, numbers, 1)


def count_if(predicate, seq):
  """Count the number of elements of seq for which the predicate is true.
  >>> count_if(callable, [42, None, max, min])
  2
  """
  f = lambda count, x: count + (not not predicate(x))
  return reduce(f, seq, 0)


def find_if(predicate, seq):
  """If there is an element of seq that satisfies predicate; return it.
  >>> find_if(callable, [3, min, max])
  <built-in function min>
  >>> find_if(callable, [1, 2, 3])
  """
  for x in seq:
    if predicate(x):
      return x
  return None


def every(predicate, seq):
  """True if every element of seq satisfies predicate.
  >>> every(callable, [min, max])
  1
  >>> every(callable, [min, 3])
  0
  """
  for x in seq:
    if not predicate(x):
      return False
  return True


def some(predicate, seq):
  """If some element x of seq satisfies predicate(x), return predicate(x).
  >>> some(callable, [min, 3])
  1
  >>> some(callable, [2, 3])
  0
  """
  for x in seq:
    px = predicate(x)
    if  px:
      return px
  return False


def is_in(elt, seq):
  """Like (elt in seq), but compares with is, not ==.
  >>> e = []; is_in(e, [1, e, 3])
  True
  >>> is_in(e, [1, [], 3])
  False
  """
  for x in seq:
    if elt is x:
      return True
  return False
