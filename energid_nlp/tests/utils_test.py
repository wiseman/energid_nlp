# Copyright 2012 Energid Technologies

import unittest

from energid_nlp import utils


class UtilsTests(unittest.TestCase):
  def test_element_of(self):
    """Test element_of."""
    self.assertTrue(utils.element_of(1, [1, 2, 3]))
    self.assertTrue(utils.element_of(2, (1, 2, 3)))
    self.assertTrue(utils.element_of('a', {'a': 0, 'b': 1}))
    self.assertFalse(utils.element_of(5, [1, 2, 3]))

    def equal_mod_2(a, b):
      return (a % 2) == (b % 2)

    self.assertTrue(utils.element_of(3, [1, 2], test_fn=equal_mod_2))
    self.assertFalse(utils.element_of(3, [0, 2], test_fn=equal_mod_2))

  def test_is_number(self):
    """Test is_number."""
    self.assertTrue(utils.is_number(0))
    self.assertTrue(utils.is_number(1))
    self.assertTrue(utils.is_number(1.5))
    self.assertTrue(utils.is_number(5j))
    self.assertTrue(utils.is_number(2 + 5j))
    self.assertTrue(utils.is_number(999999999999999L))
    # See utils.is_number source if wondering about NaNs.
    self.assertTrue(utils.is_number(float('NaN')))
    self.assertFalse(utils.is_number('a'))
    self.assertFalse(utils.is_number({}))

  def test_is_sequence(self):
    """Test is_sequence."""
    self.assertTrue(utils.is_sequence({}))
    self.assertTrue(utils.is_sequence([]))
    self.assertFalse(utils.is_sequence(5))
    self.assertFalse(utils.is_sequence(self))

  def test_if_(self):
    """Test if_."""
    def return_nine():
      return 9

    def return_ten():
      return 10

    self.assertEqual(3, utils.if_(True, 3, 4))
    self.assertEqual(4, utils.if_(False, 3, 4))
    self.assertEqual(9, utils.if_(True, return_nine, return_ten))
    self.assertEqual(10, utils.if_(False, return_nine, return_ten))

  def test_num_or_str(self):
    """Test num_or_str."""
    self.assertEqual(42, utils.num_or_str('42'))
    self.assertEqual('42x', utils.num_or_str('42x'))

  def test_remove_all(self):
    """Test remove_all."""
    self.assertEqual([1, 2, 2, 1], utils.remove_all(3, [1, 2, 3, 3, 2, 1, 3]))
    self.assertEqual([1, 2, 3], utils.remove_all(4, [1, 2, 3]))


if __name__ == '__main__':
  unittest.main()
