# Copyright 2012 Energid Technologies

import unittest

from energid_nlp import logic

from energid_nlp.tests import testutils


class LogicTests(testutils.KBTestCase):
  def test_simple_assertions(self):
    """Test simple propositional assertions."""
    kb = logic.PropKB()
    kb.tell(logic.expr('color(cat, black)'))
    self.assertAllBindingsEqual(
      kb.ask_all(logic.expr('color(cat, black)')),
      [{}])
    kb.tell(logic.expr('age(cat, 35)'))
    self.assertAllBindingsEqual(kb.ask_all(logic.expr('age(cat, 35)')), [{}])

  def test_simple_assertions2(self):
    """Test more simple propositional assertions."""
    kb = logic.PropKB()
    kb.tell(logic.expr('color(cat, coat, black)'))
    self.assertAllBindingsEqual(
      kb.ask_all(logic.expr('color(cat, coat, black)')),
      [{}])
    kb.tell(logic.expr('age(cat, toy, 35)'))
    self.assertAllBindingsEqual(
      kb.ask_all(logic.expr('age(cat, toy, 35)')),
      [{}])
    kb.tell(logic.expr('color(cat, mitten, left, black)'))
    self.assertAllBindingsEqual(
      kb.ask_all(logic.expr('color(cat, mitten, left, black)')),
      [{}])
    kb.tell(logic.expr('age(cat, toy, top, 35)'))
    self.assertAllBindingsEqual(
      kb.ask_all(logic.expr('age(cat, toy, top, 35)')),
      [{}])
    kb.tell(logic.expr('age(cat, toy, top, x, y, z, 35)'))
    self.assertAllBindingsEqual(
      kb.ask_all(logic.expr('age(cat, toy, top, x, y, z, 35)')),
      [{}])

  def test_simple_erasures(self):
    """Test simple variable-less retractions."""
    kb = logic.PropKB()
    kb.tell(logic.expr('color(cat, black)'))
    kb.tell(logic.expr('age(cat,35)'))
    self.assertAllBindingsEqual(
      kb.ask_all(logic.expr('color(cat, black)')),
      [{}])
    kb.retract(logic.expr('color(cat, black)'))
    self.assertBindingsEqual(
      kb.ask(logic.expr('color(cat, black)')),
      False)
    self.assertAllBindingsEqual(
      kb.ask_all(logic.expr('color(cat, black)')),
      [])

  def test_simple_erasures2(self):
    """Test more simple variable-less retractions."""
    kb = logic.PropKB()
    kb.tell(logic.expr('color(cat, coat, black)'))
    kb.tell(logic.expr('age(cat, toy, 35)'))
    self.assertAllBindingsEqual(
      kb.ask_all(logic.expr('color(cat, coat, black)')),
      [{}])
    kb.retract(logic.expr('color(cat, coat, black)'))
    self.assertBindingsEqual(
      kb.ask(logic.expr('color(cat, coat, black)')),
      False)
    self.assertAllBindingsEqual(
      kb.ask_all(logic.expr('color(cat, coat, black)')),
      [])

  def test_erasures(self):
    """Test retractions with variables."""
    kb = logic.PropKB()
    kb.tell(logic.expr('has(john, cat)'))
    kb.tell(logic.expr('has(john, car)'))
    kb.tell(logic.expr('has(cat, toy)'))
    kb.retract(logic.expr('has(john, ?t)'))
    self.assertBindingsEqual(kb.ask(logic.expr('has(john, ?t)')), False)
    self.assertAllBindingsEqual(kb.ask_all(logic.expr('has(john, ?t)')), [])
    self.assertAllBindingsEqual(kb.ask_all(logic.expr('has(cat, toy)')), [{}])

  def test_erasures2(self):
    """Test more retractions with variables."""
    kb = logic.PropKB()
    kb.tell(logic.expr('has(john, bag, cat)'))
    kb.tell(logic.expr('has(john, bag, car)'))
    kb.tell(logic.expr('has(john, bin, cat)'))
    kb.tell(logic.expr('has(john, bin, car)'))
    kb.tell(logic.expr('has(cat, bag, toy)'))
    kb.retract(logic.expr('has(john, bag, ?t)'))
    self.assertBindingsEqual(kb.ask(logic.expr('has(john, bag, ?t)')), False)
    self.assertAllBindingsEqual(kb.ask_all(logic.expr('has(john, bin, ?t)')),
                                [{'?t': 'cat'}, {'?t': 'car'}])
    self.assertAllBindingsEqual(kb.ask_all(logic.expr('has(john, ?t)')), [])
    self.assertAllBindingsEqual(
      kb.ask_all(logic.expr('has(cat, bag, toy)')),
      [{}])
    kb.tell(logic.expr('has(john, bag, cat)'))
    kb.tell(logic.expr('has(john, bag, car)'))
    kb.retract(logic.expr('has(john, ?t, cat)'))
    self.assertAllBindingsEqual(
      kb.ask_all(logic.expr('has(john, ?t, cat)')),
      [])
    self.assertAllBindingsEqual(
      kb.ask_all(logic.expr('has(john, ?t, car)')),
      [{'?t': 'bin'}, {'?t': 'bag'}])

  def test_simple_unification(self):
    """Test simple queries with variables."""
    kb = logic.PropKB()
    kb.tell(logic.expr('color(cat, black)'))
    kb.tell(logic.expr('age(cat, 35)'))
    self.assertAllBindingsEqual(
      kb.ask_all(logic.expr('color(cat, ?x)')),
      [{'?x': 'black'}])

  def test_simple_unification2(self):
    """Test more simple queries with variables."""
    kb = logic.PropKB()
    kb.tell(logic.expr('color(cat, fur, black)'))
    kb.tell(logic.expr('age(cat, mental, 3)'))
    self.assertAllBindingsEqual(kb.ask_all(logic.expr('color(cat, ?x)')), [])
    self.assertAllBindingsEqual(
      kb.ask_all(logic.expr('color(cat, fur, ?x)')),
      [{'?x': 'black'}])
    self.assertAllBindingsEqual(
      kb.ask_all(logic.expr('age(cat, ?type, 3)')),
      [{'?type': 'mental'}])
    self.assertAllBindingsEqual(
      kb.ask_all(logic.expr('age(cat, ?type, 4)')),
      [])
    self.assertAllBindingsEqual(
      kb.ask_all(logic.expr('color(cat, ?thing, ?c)')),
      [{'?c': 'black', '?thing': 'fur'}])
    self.assertAllBindingsEqual(
      kb.ask_all(logic.expr('color(?a, ?b, ?c)')),
      [{'?a': 'cat', '?b': 'fur', '?c': 'black'}])
    self.assertAllBindingsEqual(
      kb.ask_all(logic.expr('color(?a, ?b, ?c, ?d)')),
      [])

  def test_conjunction(self):
    """Test conjunctive queries."""
    kb = logic.PropKB()
    kb.tell(logic.expr('color(cat, black)'))
    kb.tell(logic.expr('age(cat, 35)'))
    kb.tell(logic.expr('name(cat, ted)'))
    self.assertBindingsEqual(
      kb.ask(logic.expr('(color(cat, black) & age(cat, 35))')),
      {})
    self.assertBindingsEqual(
      kb.ask(logic.expr(
        '(color(cat, black) & (age(cat, 35) & name(cat, ted)))')),
      {})
    self.assertBindingsEqual(
      kb.ask(logic.expr('(color(cat, white) & age(cat, 35))')),
      False)
    self.assertBindingsEqual(
      kb.ask(logic.expr('(color(cat, black) & age(cat, 34))')),
      False)
    self.assertAllBindingsEqual(
      kb.ask_all(logic.expr('(color(cat, ?c) & age(cat, ?a))')),
      [{'?a': '35', '?c': 'black'}])
    self.assertBindingsEqual(
      kb.ask(logic.expr('(color(cat, ?c) & age(cat, ?c))')),
      False)
    kb.tell(logic.expr('color(car, black)'))
    self.assertAllBindingsEqual(
      kb.ask_all(logic.expr('(color(cat, ?b) & color(car, ?b))')),
      [{'?b': 'black'}])

  def test_memory_functions(self):
    """Test memory functions."""
    kb = logic.PropKB()
    kb.tell(logic.expr('color(cat, black)'))
    kb.tell(logic.expr('color(car, black)'))
    # <=> (logical equivalence) is implemented as a memory function.
    self.assertBindingsEqual(kb.ask(logic.expr('(Cat <=> Cat)')), {})
    self.assertBindingsEqual(kb.ask(logic.expr('(Cat <=> Baseball)')), False)
    self.assertBindingsEqual(kb.ask(logic.expr('(cat <=> cat)')), {})
    self.assertBindingsEqual(kb.ask(logic.expr('(cat <=> Cat)')), False)
    self.assertBindingsEqual(kb.ask(logic.expr('(~(Cat <=> Cat))')), False)
    self.assertBindingsEqual(kb.ask(logic.expr('(~(Cat <=> Baseball))')), {})
    self.assertAllBindingsEqual(
      kb.ask_all(logic.expr(
        '((color(cat, ?x) & color(car, ?y)) & (?x <=> ?y))')),
      [{'?x': 'black', '?y': 'black'}])
    self.assertAllBindingsEqual(
      kb.ask_all(logic.expr(
        '((color(?a, black) & color(?b, black)) & (~(?a <=> ?b)))')),
      [{'?a': 'cat', '?b': 'car'}, {'?a': 'car', '?b': 'cat'}])

    # Bind is another memory function; it just binds a variable.
    self.assertAllBindingsEqual(
      kb.ask_all(logic.expr('Bind(?x, boo)')), [{'?x': 'boo'}])
    self.assertBindingsEqual(
      kb.ask(logic.expr('(Color(Cat, ?b) & Bind(?b, Boo))')), False)

    # Shouldn't be able to assert a 'Bind'.
    self.assertRaises(logic.Error, lambda: kb.tell('Bind(?x, boo)'))

  def test_disjunction(self):
    """Test disjunctive queries."""
    kb = logic.PropKB()
    kb.tell(logic.expr('Color(Cat, Black)'))
    kb.tell(logic.expr('Age(Cat, 35)'))
    kb.tell(logic.expr('Name(Cat, Ted)'))
    self.assertBindingsEqual(
      kb.ask(logic.expr('(Color(Cat, Black) | Age(Cat, 35))')), {})
    self.assertBindingsEqual(
      kb.ask(logic.expr('(Color(Cat, Black) | Age(Cat, 36))')), {})
    self.assertBindingsEqual(
      kb.ask(logic.expr('(Color(Cat, White) | Age(Cat, 35))')), {})
    self.assertBindingsEqual(
      kb.ask(logic.expr('(Color(Cat, White) | Age(Cat, 36))')), False)
    self.assertAllBindingsEqual(
      kb.ask_all(logic.expr('(Color(Cat, ?c) | Age(Cat, ?a))')),
      [{'?c': 'Black'}, {'?a': '35'}])
    self.assertAllBindingsEqual(
      kb.ask_all(logic.expr('(Color(Cat, ?c) | Age(Cat, 36))')),
      [{'?c': 'Black'}])

  def test_not(self):
    """Gotta have NOT."""
    kb = logic.PropKB()
    kb.tell(logic.expr('Color(Cat, Black)'))
    kb.tell(logic.expr('Age(Cat, 35)'))
    kb.tell(logic.expr('Name(Cat, Ted)'))
    self.assertBindingsEqual(
      kb.ask(logic.expr('(Color(Cat, Black) & (~Age(Cat, 36)))')), {})
    self.assertBindingsEqual(
      kb.ask(logic.expr('(Color(Cat, Black) & (~Age(Cat, 35)))')), False)
    self.assertBindingsEqual(
      kb.ask(logic.expr('((~Age(Cat, 35)) & Color(Cat, Black))')), False)
    self.assertBindingsEqual(
      kb.ask(logic.expr('((~Age(Cat, 36)) & Color(Cat, Black))')), {})
    self.assertBindingsEqual(
      kb.ask(logic.expr('(Color(Cat, Black) & (~Age(Cat, ?c)))')), False)
    self.assertBindingsEqual(
      kb.ask(logic.expr('((~Age(Cat, ?c)) & Color(Cat, Black))')), False)
    self.assertBindingsEqual(
      kb.ask(logic.expr('(Color(Cat, ?c) & (~Age(Cat, ?a)))')), False)
    self.assertBindingsEqual(
      kb.ask(logic.expr('((~Age(Cat, ?a)) & Color(Cat, ?c))')), False)
    self.assertBindingsEqual(
      kb.ask(logic.expr('((~Age(Cat, 36)) & Color(Cat, ?c))')),
      {'?c': 'Black'})
    self.assertBindingsEqual(
      kb.ask(logic.expr('(Color(Cat, ?c) & (~Age(Cat, 36)))')),
      {'?c': 'Black'})
    self.assertBindingsEqual(kb.ask(logic.expr('~Age(Cat, 35)')), False)
    self.assertBindingsEqual(kb.ask(logic.expr('~Age(Cat, 36)')), {})

  def test_conjuction_with_disjunction(self):
    """Test queries that combine conjunction and disjunction."""
    kb = logic.PropKB()
    kb.tell(logic.expr('Color(Cat, Black)'))
    kb.tell(logic.expr('Age(Cat, 35)'))
    kb.tell(logic.expr('Name(Cat, Ted)'))
    self.assertBindingsEqual(
      kb.ask(logic.expr(
               '(Color(Cat, Black) & (Age(Cat, 36) | Name(Cat, Ted)))')), {})
    self.assertBindingsEqual(
        kb.ask(logic.expr(
          '(Color(Cat, Black) & (Age(Cat, 36) | Name(Cat, John)))')), False)
    self.assertBindingsEqual(
        kb.ask(logic.expr(
          '((Age(Cat, 36) | Name(Cat, Ted)) & Color(Cat, Black))')), {})
    self.assertBindingsEqual(
        kb.ask(logic.expr(
          '((Age(Cat, 36) | Name(Cat, John)) & Color(Cat, Black))')), False)

  def test_fluents(self):
    """Test that fluents work correctly.  Fluents are propositions
    that can have only one value--e.g., people have only one age; I
    can't be 35 and 40 simultaneously, so Age(John, 35) and Age(John,
    40) can't both be true.
    """
    kb = logic.PropKB()
    kb.tell(logic.expr('Has(John, Cat)'))
    kb.tell(logic.expr('Has(John, Computer)'))
    self.assertAllBindingsEqual(
        kb.ask_all(logic.expr(
          'Has(John, ?w)')), [{'?w': 'Computer'}, {'?w': 'Cat'}])
    kb.define_fluent(logic.expr('Age'))
    kb.tell(logic.expr('Age(John, 35)'))
    kb.tell(logic.expr('Age(John, 40)'))
    self.assertAllBindingsEqual(
        kb.ask_all(logic.expr('Age(John, ?a)')), [{'?a': '40'}])
    self.assertRaises(logic.Error, lambda: kb.define_fluent('Foo'))

  def test_fluents2(self):
    """More fluent tests."""
    kb = logic.PropKB()
    kb.define_fluent(logic.expr('PartColor'))
    kb.tell(logic.expr('PartColor(john, hand, red)'))
    kb.tell(logic.expr('PartColor(john, hair, brown)'))
    kb.tell(logic.expr('PartColor(john, hand, green)'))
    kb.tell(logic.expr('PartColor(john, hair, black)'))
    self.assertAllBindingsEqual(
      kb.ask_all(logic.expr(
        'PartColor(john, hair, ?a)')), [{'?a': 'black'}])
    self.assertAllBindingsEqual(
      kb.ask_all(logic.expr('PartColor(john, hand, ?a)')), [{'?a': 'green'}])
    self.assertAllBindingsEqual(
      kb.ask_all(logic.expr('PartColor(john, hair, brown)')), [])
    self.assertAllBindingsEqual(
      kb.ask_all(logic.expr('PartColor(john, hand, red)')), [])
    self.assertAllBindingsEqual(
      kb.ask_all(logic.expr('PartColor(john, ?x, ?y)')),
      [{'?x': 'hand', '?y': 'green'}, {'?x': 'hair', '?y': 'black'}])

  def test_inheritance(self):
    """Test inheritance."""
    # Set up a simple inheritance hierarchy.
    kb = logic.PropKB()
    kb.tell(logic.expr('ISA(Mammal, Animal)'))
    kb.tell(logic.expr('ISA(Cat, Mammal)'))
    kb.tell(logic.expr('ISA(Petunia, Cat)'))
    kb.tell(logic.expr('ISA(Painter, Artist)'))

    # Query the hierarchy a bit
    self.assertAllBindingsEqual(
      kb.ask_all(logic.expr('ISA(Petunia, ?x)')),
      [{'?x': 'Petunia'}, {'?x': 'Cat'}, {'?x': 'Mammal'},
       {'?x': 'Animal'}])
    self.assertAllBindingsEqual(
      kb.ask_all(logic.expr('ISA(Cat, ?x)')),
      [{'?x': 'Cat'}, {'?x': 'Mammal'},
       {'?x': 'Animal'}])
    self.assertAllBindingsEqual(kb.ask_all(logic.expr('ISA(Cat, Cat)')), [{}])

    # Oh, this was nasty.
    self.assertAllBindingsEqual(
      kb.ask_all(logic.expr(
        '((Bind(?x1, Cat) & Bind(?x, Painter)) & '
        'ISA(?x1, Animal)) & ISA(?x, Artist)')),
      [{'?x': 'Painter', '?x1': 'Cat'}])

    # Define 'Size' as an inherited fluent.
    kb.define_fluent(logic.expr('Size'), inherited=True)
    kb.tell(logic.expr('Size(Mammal, Medium)'))
    kb.tell(logic.expr('Size(Petunia, Petite)'))

    # Define 'Color' as an inherited fluent.
    kb.define_fluent(logic.expr('Color'), inherited=True)
    kb.tell(logic.expr('Color(Petunia, Gray)'))

    # Define 'Name' as an non-inherited fluent.
    kb.define_fluent(logic.expr('Name'), inherited=False)
    kb.tell(logic.expr('Name(Animal, Peter)'))
    kb.tell(logic.expr('Name(Petunia, Petunia)'))
    self.assertBindingsEqual(kb.ask(logic.expr('ISA(Petunia, Animal)')), {})

    # Petunia ISA Petunia, Cat, Mammal and Animal.
    self.assertAllBindingsEqual(
      kb.ask_all(logic.expr('ISA(Petunia, ?x)')),
      [{'?x': 'Petunia'}, {'?x': 'Cat'}, {'?x': 'Mammal'},
       {'?x': 'Animal'}])

    # Check all_parents and git checkout pychall_children.
    self.assertEqual(
      kb.all_parents(logic.expr('Petunia')),
      map(logic.expr, ['Petunia', 'Cat', 'Mammal', 'Animal']))
    self.assertEqual(
      kb.all_parents(logic.expr('Cat')),
      map(logic.expr, ['Cat', 'Mammal', 'Animal']))
    self.assertEqual(
      kb.all_parents(logic.expr('XZZY')),
      map(logic.expr, ['XZZY']))
    self.assertEqual(
      kb.all_parents(logic.expr('Animal')),
      map(logic.expr, ['Animal']))

    self.assertEqual(
      kb.all_children(logic.expr('Petunia')),
      map(logic.expr, ['Petunia']))
    self.assertEqual(
      kb.all_children(logic.expr('Cat')),
      map(logic.expr, ['Cat', 'Petunia']))
    self.assertEqual(
      kb.all_children(logic.expr('XZZY')),
      map(logic.expr, ['XZZY']))
    self.assertEqual(
      kb.all_children(logic.expr('Animal')),
      map(logic.expr, ['Animal', 'Mammal', 'Cat', 'Petunia']))

    self.assertEqual(
      kb.isa(logic.expr('Petunia'), logic.expr('Petunia')),
      True)
    self.assertEqual(kb.isa(logic.expr('Petunia'), logic.expr('Mammal')), True)
    self.assertEqual(kb.isa(logic.expr('Petunia'), logic.expr('Cat')), True)
    self.assertEqual(kb.isa(logic.expr('Petunia'), logic.expr('Peter')), False)

    # Check that we're managing the inheritance caches correctly.
    self.assertEqual(
      kb.all_children(logic.expr('A')),
      map(logic.expr, ['A']))
    self.assertEqual(
      kb.all_parents(logic.expr('B')),
      map(logic.expr, ['B']))
    kb.tell(logic.expr('ISA(B, A)'))
    self.assertEqual(
      kb.all_children(logic.expr('A')),
      map(logic.expr, ['A', 'B']))
    self.assertEqual(
      kb.all_parents(logic.expr('B')),
      map(logic.expr, ['B', 'A']))
    self.assertEqual(
      kb.all_proper_children(logic.expr('A')),
      map(logic.expr, ['B']))
    self.assertEqual(
      kb.all_proper_parents(logic.expr('B')),
      map(logic.expr, ['A']))
    kb.tell(logic.expr('ISA(C, B)'))
    self.assertEqual(
      kb.all_proper_children(logic.expr('B')),
      map(logic.expr, ['C']))
    self.assertEqual(
      kb.all_proper_children(logic.expr('A')),
      map(logic.expr, ['B', 'C']))
    self.assertEqual(
      kb.all_proper_parents(logic.expr('C')),
      map(logic.expr, ['B', 'A']))

    # Check that we can query slot values correctly.
    self.assertBindingsEqual(kb.ask(logic.expr('Size(Animal, x)')), False)
    self.assertAllBindingsEqual(
      kb.ask_all(logic.expr('Size(Mammal, ?s)')),
      [{'?s': 'Medium'}])
    self.assertAllBindingsEqual(
      kb.ask_all(logic.expr('Size(Cat, ?s)')),
      [{'?s': 'Medium'}])
    self.assertAllBindingsEqual(
      kb.ask_all(logic.expr('Size(Petunia, ?s)')),
      [{'?s': 'Petite'}])
    self.assertAllBindingsEqual(
      kb.ask_all(logic.expr('Name(Animal, ?n)')),
      [{'?n': 'Peter'}])
    self.assertBindingsEqual(kb.ask(logic.expr('Name(Cat, ?n)')), False)
    self.assertAllBindingsEqual(
      kb.ask_all(logic.expr('Name(Petunia, ?n)')),
      [{'?n': 'Petunia'}])

    # Just checking that we're substituting variables correctly.
    self.assertAllBindingsEqual(
      kb.ask_all(logic.expr('(Bind(?x, Petunia) & Size(?x, ?s))')),
      [{'?x': 'Petunia', '?s': 'Petite'}])

    # Instead of asking what some thing's slot value is, here we ask
    # for all things that have the given value (including those things
    # for which we haven't explicitly asserted the slot value, but
    # have the value through inheritance).
    self.assertAllBindingsEqual(
      kb.ask_all(logic.expr('Size(?x, Petite)')),
      [{'?x': 'Petunia'}])
    self.assertAllBindingsEqual(kb.ask_all(logic.expr('Size(?x, Medium)')),
                                [{'?x': 'Cat'}, {'?x': 'Mammal'}])
    self.assertAllBindingsEqual(
      kb.ask_all(logic.expr('Name(?x, Peter)')),
      [{'?x': 'Animal'}])
    self.assertAllBindingsEqual(
      kb.ask_all(logic.expr('Name(?x, Petunia)')),
      [{'?x': 'Petunia'}])
    self.assertAllBindingsEqual(
      kb.ask_all(logic.expr('(Bind(?x, Petite) & Size(?t, ?x))')),
      [{'?x': 'Petite', '?t': 'Petunia'}])
    self.assertAllBindingsEqual(
      kb.ask_all(logic.expr('(Color(?x, Gray) & Size(?x, Petite))')),
      [{'?x': 'Petunia'}])
    self.assertAllBindingsEqual(
      kb.ask_all(logic.expr('(Color(?x, Black) & Size(?x, Petite))')),
      [])
    # Can't have two variables in queries of inheritable slots.
    self.assertRaises(logic.Error, lambda: kb.ask(logic.expr('Size(?x, ?y)')))

  def test_inheritance2(self):
    """More inheritance tests."""
    # Set up a simple inheritance hierarchy.
    kb = logic.PropKB()
    kb.tell(logic.expr('ISA(Mammal, Animal)'))
    kb.tell(logic.expr('ISA(Cat, Mammal)'))
    kb.tell(logic.expr('ISA(Petunia, Cat)'))

    # Define 'Constraint' as an inherited fluent.
    kb.define_fluent(logic.expr('Constraint'), inherited=True)
    kb.tell(logic.expr('Constraint(Mammal, Size, SizeThing)'))
    kb.tell(logic.expr('Constraint(Petunia, Size, PetuniaSize)'))

    # Check that we can query slot values correctly.
    self.assertBindingsEqual(
      kb.ask(logic.expr('Constraint(Animal, Size, foo)')), False)
    self.assertBindingsEqual(
      kb.ask(logic.expr('Constraint(Animal, Color, foo)')), False)
    self.assertAllBindingsEqual(
      kb.ask_all(logic.expr('Constraint(Mammal, Size, ?c)')),
      [{'?c': 'SizeThing'}])
    self.assertEqual(kb.slot_value(logic.expr('Mammal'),
                                   logic.expr('Constraint'),
                                   logic.expr('Size')),
                     logic.expr('SizeThing'))
    self.assertAllBindingsEqual(
      kb.ask_all(logic.expr('Constraint(Cat, Size, ?c)')),
      [{'?c': 'SizeThing'}])
    self.assertAllBindingsEqual(
      kb.ask_all(logic.expr('Constraint(Petunia, Size, ?c)')),
      [{'?c': 'PetuniaSize'}])

    # Just checking that we're substituting variables correctly.
    self.assertAllBindingsEqual(
      kb.ask_all(logic.expr('(Bind(?x, Petunia) & Constraint(?x, Size, ?s))')),
      [{'?x': 'Petunia', '?s': 'PetuniaSize'}])

    # Instead of asking what some thing's slot value is, here we ask
    # for all things that have the given value (including those things
    # for which we haven't explicitly asserted the slot value, but
    # have the value through inheritance).
    self.assertAllBindingsEqual(
      kb.ask_all(logic.expr('Constraint(?x, Size, PetuniaSize)')),
      [{'?x': 'Petunia'}])
    self.assertAllBindingsEqual(
      kb.ask_all(logic.expr('Constraint(?x, Color, PetuniaSize)')),
      [])
    self.assertAllBindingsEqual(
      kb.ask_all(logic.expr('Constraint(?x, Size, SizeThing)')),
      [{'?x': 'Cat'}, {'?x': 'Mammal'}])
    self.assertAllBindingsEqual(
      kb.ask_all(logic.expr(
          '(Bind(?x, PetuniaSize) & Constraint(?t, Size, ?x))')),
      [{'?x': 'PetuniaSize', '?t': 'Petunia'}])

    # Define 'Color1' as an inherited fluent.
    kb.define_fluent(logic.expr('Color1'), inherited=True)
    kb.tell(logic.expr('Color1(Petunia, Gray)'))

    self.assertAllBindingsEqual(
      kb.ask_all(
        logic.expr('(Color1(?x, Gray) & Constraint(?x, Size, PetuniaSize))')),
      [{'?x': 'Petunia'}])
    self.assertAllBindingsEqual(
      kb.ask_all(
        logic.expr('(Color1(?x, Black) & Constraint(?x, Size, PetuniaSize))')),
      [])

  def test_descriptions(self):
    """Test descriptions."""
    kb = logic.PropKB()
    kb.tell(logic.expr('ISA(Mammal, Animal)'))
    kb.tell(logic.expr('ISA(Cat, Mammal)'))
    kb.tell(logic.expr('ISA(Petunia, Cat)'))
    kb.tell(logic.expr('ISA(Dog, Animal)'))
    kb.tell(logic.expr('ISA(Nine, Dog)'))
    kb.define_fluent(logic.expr('Color'), inherited=True)
    kb.define_fluent(logic.expr('Size'), inherited=True)
    kb.tell(logic.expr('Size(Cat, Small)'))
    kb.tell(logic.expr('Color(Petunia, Black)'))
    kb.tell(logic.expr('Size(Dog, Large)'))
    kb.tell(logic.expr('Color(Gretchen, Black)'))
    kb.tell(logic.expr('IsEdible(Animal, true)'))
    kb.tell(logic.expr('Color(Animal, Purple)'))

    d = logic.Description('Cat', {'Size': 'Small'})
    self.assertEqual(d.find_all(kb),
                     [logic.expr('Cat'), logic.expr('Petunia')])
    d = logic.Description('Cat')
    self.assertEqual(d.find_all(kb),
                     [logic.expr('Cat'), logic.expr('Petunia')])
    d = logic.Description('Cat', {'Size': 'Small', 'Color': 'Black'})
    self.assertEqual(d.find_all(kb), [logic.expr('Petunia')])
    d = logic.Description('Cat',
                          {'Size': 'Small', 'Color': 'Black', 'Age': '35'})
    self.assertEqual(d.find_all(kb), [])
    d = logic.Description('Animal', {'Color': 'Black'})
    self.assertEqual(d.find_all(kb), [logic.expr('Petunia')])

    d = logic.Description('Animal', {'Color': 'Black'})
    query = logic.expr('Color')(d, logic.expr('?c'))
    self.assertAllBindingsEqual(kb.ask_all(query), [{'?c': 'Black'}])
    query = logic.expr('IsEdible')(d, logic.expr('?c'))
    self.assertAllBindingsEqual(kb.ask_all(query), [{'?c': 'true'}])

    self.assertEqual(d.has_slot('Color'), True)
    self.assertEqual(d.has_slot(logic.expr('Color')), True)
    self.assertEqual(d.has_slot('Size'), False)
    self.assertEqual(d.has_slot(logic.expr('Size')), False)
    self.assertEqual(d.slot_value('Color'), 'Black')
    self.assertRaises(KeyError, lambda: d.slot_value('Size'))

    d = logic.Description('Cat', {'IsEdible': 'false'})
    self.assertEqual(kb.slot_value(d, 'IsEdible'), logic.expr('false'))
    self.assertEqual(kb.slot_value(d, 'Color'), logic.expr('Purple'))

  def test_descriptions2(self):
    """More description tests."""
    kb = logic.PropKB()
    kb.tell(logic.expr('ISA(c-cat, c-animal)'))
    kb.tell(logic.expr('INSTANCEOF(i-petunia, c-cat)'))
    kb.tell(logic.expr('color(i-petunia, i-gray)'))
    kb.tell(logic.expr('ISA(i-gray, c-color)'))
    kb.tell(logic.expr('alternate-spelling(i-gray, grey)'))
    # Description of a cat whose color is something with an alternate
    # spelling 'grey'.
    d = logic.Description(
      'c-cat',
      {'color': logic.Description('c-color', {'alternate-spelling': 'grey'})})
    self.assertEqual(d.find_all(kb), [logic.expr('i-petunia')])
    self.assertEqual(d.find_instances(kb), [logic.expr('i-petunia')])

  def test_descriptions3(self):
    """Even more description tests."""
    kb = logic.PropKB()
    kb.tell(logic.expr('ISA(c-cat, c-animal)'))
    kb.tell(logic.expr('INSTANCEOF(i-petunia, c-cat)'))
    kb.tell(logic.expr('color(i-petunia, i-gray)'))
    kb.tell(logic.expr('ISA(i-gray, c-color)'))
    kb.tell(logic.expr('alternate-spelling(i-gray, grey)'))
    kb.tell(logic.expr('INSTANCEOF(i-domestic-short-hair, c-species)'))
    kb.tell(logic.expr('species(i-petunia, i-domestic-short-hair)'))
    kb.tell(logic.expr('origin(i-domestic-short-hair, i-egypt)'))
    kb.tell(logic.expr('INSTANCEOF(i-egypt, c-city)'))
    kb.tell(logic.expr('capital(i-egypt, cairo)'))
    # Description of a cat whose color is something with an alternate
    # spelling 'grey' and whose species has an origin whose capital is
    # cairo.
    species_d = logic.Description(
      'c-species',
      {'origin': logic.Description('c-city', {'capital': 'cairo'})})
    d = logic.Description(
      'c-cat',
      {'color': logic.Description('c-color', {'alternate-spelling': 'grey'}),
       'species': species_d})
    self.assertEqual(d.find_all(kb), [logic.expr('i-petunia')])
    self.assertEqual(d.find_instances(kb), [logic.expr('i-petunia')])

    self.assertEqual(kb.isa(logic.expr(d), logic.expr('c-animal')), True)
    self.assertEqual(kb.isa(logic.expr(d), logic.expr('c-cat')), True)
    self.assertBindingsEqual(kb.ask(
        logic.expr('ISA')(logic.expr(d), logic.expr('c-animal'))), {})
    self.assertBindingsEqual(kb.ask(
        logic.expr('ISA')(logic.expr(d), logic.expr('c-cat'))), {})

  def test_instances(self):
    """Test INSTANCEOF and ISINSTANCE."""
    kb = logic.PropKB()
    kb.tell(logic.expr('ISA(mammal, animal)'))
    kb.tell(logic.expr('ISA(cat, mammal)'))
    kb.tell(logic.expr('INSTANCEOF(petunia, cat)'))
    self.assertAllBindingsEqual(
      kb.ask_all(logic.expr('ISA(petunia, ?x)')),
      [{'?x': 'petunia'}, {'?x': 'cat'}, {'?x': 'mammal'}, {'?x': 'animal'}])
    self.assertAllBindingsEqual(
      kb.ask_all(logic.expr('ISINSTANCE(petunia)')), [{}])
    self.assertAllBindingsEqual(
      kb.ask_all(logic.expr('ISINSTANCE(?x)')), [{'?x': 'petunia'}])

    self.assertAllBindingsEqual(
      kb.ask_all(logic.expr('INSTANCEOF(petunia, ?x)')),
      [{'?x': 'cat'}])
    self.assertAllBindingsEqual(
      kb.ask_all(logic.expr('ISINSTANCE(?x)')), [{'?x': 'petunia'}])
    self.assertAllBindingsEqual(kb.ask_all(logic.expr('INSTANCEOF(?x, ?y)')),
                                [{'?x': 'petunia', '?y': 'cat'}])
    self.assertAllBindingsEqual(
      kb.ask_all(logic.expr('ISINSTANCE(petunia)')), [{}])

  def test_dupes(self):
    """Test duplicate propositions."""
    kb = logic.PropKB()
    kb.tell(logic.expr('Foo(x, y)'))
    kb.tell(logic.expr('Foo(x, y)'))
    kb.tell(logic.expr('Foo(x, y) & Foo(x, y)'))
    self.assertAllBindingsEqual(kb.ask_all(logic.expr('Foo(x, ?y)')),
                                [{'?y': 'y'}])


if __name__ == '__main__':
  unittest.main()
