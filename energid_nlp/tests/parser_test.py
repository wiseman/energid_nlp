# Copyright 2012 Energid Technologies

import unittest

from energid_nlp import parser
from energid_nlp import logic
from energid_nlp.tests import testutils


class ConceptualParserTest(testutils.KBTestCase):
  def test_phrasal_pattern_parser(self):
    """Test the phrasal pattern parser."""
    p = parser.PhrasalPatternParser()
    self.assertEqual(p.parse_tree('addressee'),
                     [':symbol', 'addressee'])
    self.assertEqual(p.parse_tree("why don't we stop"),
                     [':sequence',
                      [':symbol', 'why'],
                      [':symbol', "don't"],
                      [':symbol', 'we'],
                      [':symbol', 'stop']])
    self.assertEqual(p.parse_tree('<addressee>'),
                     [':slotref', [':symbol', 'addressee']])
    # Note that the :any is optimized away in the following parse.
    self.assertEqual(p.parse_tree('[hi there]'),
                     [':sequence', [':symbol', 'hi'], [':symbol', 'there']])
    self.assertEqual(p.parse_tree('[one|two]'),
                     [':any', [':symbol', 'one'], [':symbol', 'two']])
    self.assertEqual(
      p.parse_tree("[let's|let us|why don't we]"),
      [':any',
       [':symbol', "let's"],
       [':sequence', [':symbol', 'let'], [':symbol', 'us']],
       [':sequence',
        [':symbol', 'why'], [':symbol', "don't"], [':symbol', 'we']]])

    self.assertEqual(
      p.parse_tree("<addressee> ?:[let's|let us|why don't we] <action> "
                   "?:[with|on] ?:the <object>"),
      [':sequence',
       [':slotref', [':symbol', 'addressee']],
       [':optional',
        [':any',
         [':symbol', "let's"],
         [':sequence', [':symbol', 'let'], [':symbol', 'us']],
         [':sequence',
          [':symbol', 'why'],
          [':symbol', "don't"],
          [':symbol', 'we']]]],
       [':slotref', [':symbol', 'action']],
       [':optional', [':any', [':symbol', 'with'], [':symbol', 'on']]],
       [':optional', [':symbol', 'the']],
       [':slotref', [':symbol', 'object']]])
    self.assertRaises(parser.Error, lambda: p.parse_tree('<foo'))
    self.assertRaises(parser.Error, lambda: p.parse_tree('[foo'))
    self.assertRaises(parser.Error, lambda: p.parse_tree('foo]'))
    self.assertRaises(parser.Error, lambda: p.parse_tree('foo>'))

  def test_simple_parsing(self):
    """Simple parsing tests."""
    kb = logic.PropKB()
    p = parser.ConceptualParser(kb)
    p.add_phrasal_pattern(logic.expr('BigRedThing'), 'big red thing')
    self.assertParseEqual(p.parse_tokens(['big', 'red', 'thing']),
                          [logic.Description('BigRedThing')])
    self.assertParseEqual(
      p.parse('big red thing'), [logic.Description('BigRedThing')])
    self.assertParseEqual(
      p.parse('BIG red THING'), [logic.Description('BigRedThing')])
    self.assertRaises(TypeError, lambda: p.parse_tokens(0))
    self.assertRaises(TypeError, lambda: p.parse_tokens([0]))
    self.assertRaises(TypeError, lambda: p.parse(0))
    self.assertRaises(TypeError, lambda: p.parse([0]))

  def test_slot_filling(self):
    """Test slot filling."""
    kb = logic.PropKB()
    kb.tell(logic.expr('$constraint(BigRedThing, color, red)'))
    p = parser.ConceptualParser(kb)
    p.add_phrasal_pattern(logic.expr('BigRedThing'), 'big <color> thing')
    p.add_phrasal_pattern(logic.expr('red'), 'red')
    self.assertParseEqual(
      p.parse('big red thing'),
      [logic.Description('BigRedThing', {'color': 'red'})])
    p.add_phrasal_pattern(logic.expr('BigRedThing'), 'big <rudolph> thing')
    self.assertRaises(parser.Error, lambda: p.parse('big crazy thing'))

    p.add_phrasal_pattern(
      logic.expr('BigRedThing'), 'big <doesnotexist> thing')
    self.assertRaises(parser.Error, lambda: p.parse('big floppy thing'))

  def test_constraints_inherited(self):
    """Test inherited constraints."""
    kb = logic.PropKB()
    # We don't use the parser we create in the next line, but we
    # depend on its side-effect of making $constraint an heritable
    # fluent.
    unused_p = parser.ConceptualParser(kb)
    kb.tell(logic.expr('ISA(c-child, c-parent)'))
    kb.tell(logic.expr('$constraint(c-parent, color, c-color)'))
    self.assertEqual(
      kb.slot_value(logic.expr('c-parent'), logic.expr('$constraint'),
                    logic.expr('color')), logic.expr('c-color'))
    self.assertEqual(
      kb.slot_value(logic.expr('c-child'), logic.expr('$constraint'),
                    logic.expr('color')), logic.expr('c-color'))
    self.assertEqual(
      kb.ask_all(logic.expr('$constraint(c-parent, color, ?v)')),
      [{logic.expr('?v'): logic.expr('c-color')}])
    self.assertEqual(kb.ask_all(logic.expr('$constraint(c-child, color, ?v)')),
                     [{logic.expr('?v'): logic.expr('c-color')}])

  def test_slot_filling2(self):
    """More slot filling tests"""
    kb = logic.PropKB()

    kb.tell(logic.expr('ISA(c-animal, c-physobj)'))
    kb.tell(logic.expr('ISA(c-dog, c-animal)'))
    kb.tell(logic.expr('ISA(c-cat, c-animal)'))
    kb.tell(logic.expr('ISA(c-red, c-color)'))
    kb.tell(logic.expr('ISA(c-green, c-color)'))
    kb.tell(logic.expr('ISA(c-big, c-size)'))
    kb.tell(logic.expr('ISA(c-small, c-size)'))
    kb.tell(logic.expr('ISA(c-feed, c-action)'))

    kb.tell(logic.expr('$constraint(c-physobj, color, c-color)'))
    kb.tell(logic.expr('$constraint(c-physobj, size, c-size)'))
    kb.tell(logic.expr('$constraint(c-action-request, action, c-action)'))
    kb.tell(logic.expr(
        '$constraint(c-action-request, theta-object, c-physobj)'))

    p = parser.ConceptualParser(kb)
    p.add_phrasal_pattern(logic.expr('c-physobj'), '<size> <color> thing')
    p.add_phrasal_pattern(logic.expr('c-red'), 'red')
    p.add_phrasal_pattern(logic.expr('c-green'), 'green')
    p.add_phrasal_pattern(logic.expr('c-small'), 'small')
    p.add_phrasal_pattern(logic.expr('c-small'), 'tiny')
    p.add_phrasal_pattern(logic.expr('c-big'), 'big')
    p.add_phrasal_pattern(logic.expr('c-big'), 'large')
    p.add_phrasal_pattern(logic.expr('c-feed'), 'feed')
    p.add_phrasal_pattern(
      logic.expr('c-action-request'), '<action> the <theta-object>')

    self.assertParseEqual(
      p.parse_tokens(['big', 'red', 'thing']),
      [logic.Description('c-physobj', {'color': 'c-red', 'size': 'c-big'})])
    self.assertParseEqual(
      p.parse('big red thing'),
      [logic.Description('c-physobj', {'color': 'c-red', 'size': 'c-big'})])
    self.assertParseEqual(
      p.parse('big green thing'),
      [logic.Description('c-physobj', {'color': 'c-green', 'size': 'c-big'})])
    self.assertParseEqual(
      p.parse('small red thing'),
      [logic.Description('c-physobj', {'color': 'c-red', 'size': 'c-small'})])
    self.assertParseEqual(
      p.parse('small green thing'),
      [logic.Description('c-physobj',
                         {'color': 'c-green', 'size': 'c-small'})])
    self.assertParseEqual(
      p.parse('feed the small green thing'),
      [logic.Description(
         'c-action-request',
         {'action': 'c-feed',
          'theta-object': logic.Description('c-physobj',
                                            {'color': 'c-green',
                                             'size': 'c-small'})})])

  def test_non_directives(self):
    """Test that we get expected errors when using non-directives."""
    kb = logic.PropKB()
    p = parser.ConceptualParser(kb)
    self.assertRaises(
      parser.Error,
      lambda: p.add_phrasal_pattern_object(
        logic.expr('c-thing'), [':not-a-directive', 'a', 'b', 'c']))

  def test_sequence(self):
    """Test :sequence patterns."""
    kb = logic.PropKB()
    p = parser.ConceptualParser(kb)
    p.add_phrasal_pattern_object(
      logic.expr('c-thing'), [':sequence', 'a', 'b', 'c'])
    self.assertParseEqual(p.parse('a b c'), [logic.Description('c-thing')])
    self.assertParseEqual(p.parse('a b c d'), [])
    self.assertParseEqual(p.parse('g a b c'), [])
    p.add_phrasal_pattern_object(
      logic.expr('c-thing'), [':sequence', 'a', [':sequence', 'x', 'y', 'z']])
    p.add_phrasal_pattern_object(
      logic.expr('c-thing'), [':sequence', 'a', [':sequence', 'x']])
    p.add_phrasal_pattern_object(
      logic.expr('c-thing'), [':sequence', [':sequence', 'x', 'y', 'z'], 'a'])
    self.assertParseEqual(p.parse('a x y z'), [logic.Description('c-thing')])
    self.assertParseEqual(p.parse('a x'), [logic.Description('c-thing')])
    self.assertParseEqual(p.parse('x y z a'), [logic.Description('c-thing')])

    p.add_phrasal_pattern_object(
      logic.expr('c-thing'), [':sequence', [':sequence', 'x', 0, 'z'], 'a'])
    self.assertRaises(TypeError, lambda: p.parse_tokens(['x', 0, 'z', 'a']))

  def test_optional(self):
    """Test :optional patterns."""
    kb = logic.PropKB()
    p = parser.ConceptualParser(kb)
    p.add_phrasal_pattern_object(
      logic.expr('c-thing'),
      [':sequence', 'the', [':optional', 'big'], 'thing'])
    self.assertParseEqual(p.parse('the thing'), [logic.Description('c-thing')])
    self.assertParseEqual(
      p.parse('the big thing'), [logic.Description('c-thing')])
    self.assertParseEqual(p.parse('the'), [])
    self.assertParseEqual(p.parse('thing'), [])

    p = parser.ConceptualParser(kb)
    p.add_phrasal_pattern_object(
      logic.expr('c-thing'), [':sequence', [':optional', 'the'], 'thing'])
    self.assertParseEqual(p.parse('thing'), [logic.Description('c-thing')])
    self.assertParseEqual(p.parse('the thing'), [logic.Description('c-thing')])

    p = parser.ConceptualParser(kb)
    p.add_phrasal_pattern_object(
      logic.expr('c-thing'), [':sequence', 'the', [':optional', 'thing']])
    self.assertParseEqual(p.parse('the'), [logic.Description('c-thing')])
    self.assertParseEqual(p.parse('the thing'), [logic.Description('c-thing')])

  def test_any(self):
    """Test :any patterns."""
    kb = logic.PropKB()
    p = parser.ConceptualParser(kb)
    p.add_phrasal_pattern_object(
      logic.expr('c-thing'), [':sequence', 'a', [':any', 'b', 'c', 'd'], 'e'])
    self.assertParseEqual(p.parse('a b e'), [logic.Description('c-thing')])
    self.assertParseEqual(p.parse('a c e'), [logic.Description('c-thing')])
    self.assertParseEqual(p.parse('a d e'), [logic.Description('c-thing')])
    self.assertParseEqual(p.parse('a z e'), [])

    p = parser.ConceptualParser(kb)
    p.add_phrasal_pattern_object(
      logic.expr('c-thing'), [':sequence', [':optional', 'the'], 'thing'])
    self.assertParseEqual(p.parse('thing'), [logic.Description('c-thing')])
    self.assertParseEqual(p.parse('the thing'), [logic.Description('c-thing')])

  def test_combo_directives(self):
    """Test more combinations of directives."""
    kb = logic.PropKB()
    p = parser.ConceptualParser(kb)
    p.add_phrasal_pattern_object(
      logic.expr('c-thing'),
      [':sequence', 'a', [':any',
                          [':sequence',
                           'b',
                           [':optional', [':any', 'c', 'd']],
                           'e'],
                          'z']])
    self.assertParseEqual(p.parse('a b e'), [logic.Description('c-thing')])
    self.assertParseEqual(p.parse('a b c e'), [logic.Description('c-thing')])
    self.assertParseEqual(p.parse('a b d e'), [logic.Description('c-thing')])
    self.assertParseEqual(p.parse('a b d e e'), [])
    self.assertParseEqual(p.parse('a z'), [logic.Description('c-thing')])
    self.assertParseEqual(p.parse('a z e'), [])

  def test_head_directive(self):
    """Test :head directive."""
    kb = logic.PropKB()
    kb.tell(logic.expr('ISA(c-animal, c-physobj)'))
    kb.tell(logic.expr('ISA(c-dog, c-animal)'))
    kb.tell(logic.expr('ISA(c-cat, c-animal)'))
    kb.tell(logic.expr('ISA(c-red, c-color)'))
    kb.tell(logic.expr('ISA(c-green, c-color)'))
    kb.tell(logic.expr('ISA(c-big, c-size)'))
    kb.tell(logic.expr('ISA(c-small, c-size)'))
    kb.tell(logic.expr('ISA(c-flea-bitten, c-flea-status)'))
    kb.tell(logic.expr('ISA(c-flea-free, c-flea-status)'))
    kb.tell(logic.expr('ISA(c-feed, c-action)'))
    kb.tell(logic.expr('$constraint(c-physobj, color, c-color)'))
    kb.tell(logic.expr('$constraint(c-physobj, size, c-size)'))
    kb.tell(logic.expr('$constraint(c-action-request, action, c-action)'))
    kb.tell(
      logic.expr('$constraint(c-action-request, theta-object, c-physobj)'))
    kb.tell(logic.expr('$constraint(c-dog, flea-status, c-flea-status)'))

    p = parser.ConceptualParser(kb)
    p.add_phrasal_pattern(logic.expr('c-physobj'), '<size> <color> <:head>')
    p.add_phrasal_pattern(logic.expr('c-physobj'), 'thing')
    p.add_phrasal_pattern(logic.expr('c-cat'), 'cat')
    p.add_phrasal_pattern(logic.expr('c-dog'), 'dog')
    p.add_phrasal_pattern(logic.expr('c-dog'), '<flea-status> dog')
    p.add_phrasal_pattern(logic.expr('c-red'), 'red')
    p.add_phrasal_pattern(logic.expr('c-green'), 'green')
    p.add_phrasal_pattern(logic.expr('c-small'), 'small')
    p.add_phrasal_pattern(logic.expr('c-small'), 'tiny')
    p.add_phrasal_pattern(logic.expr('c-big'), 'big')
    p.add_phrasal_pattern(logic.expr('c-big'), 'large')
    p.add_phrasal_pattern(logic.expr('c-flea-bitten'), 'itchy')
    p.add_phrasal_pattern(logic.expr('c-flea-free'), 'happy')
    p.add_phrasal_pattern(logic.expr('c-feed'), 'feed')
    p.add_phrasal_pattern(
      logic.expr('c-action-request'), '<action> <theta-object>')
    p.add_phrasal_pattern(
      logic.expr('c-action-request'), '<action> the <theta-object>')

    self.assertParseEqual(
      p.parse('big red thing'),
      [logic.Description('c-physobj', {'color': 'c-red', 'size': 'c-big'})])
    self.assertParseEqual(
      p.parse('big green thing'),
      [logic.Description('c-physobj', {'color': 'c-green', 'size': 'c-big'})])
    self.assertParseEqual(
      p.parse('small red thing'),
      [logic.Description('c-physobj', {'color': 'c-red', 'size': 'c-small'})])
    self.assertParseEqual(
      p.parse('small green thing'),
      [logic.Description('c-physobj',
                         {'color': 'c-green', 'size': 'c-small'})])
    cat_desc = logic.Description(
      'c-cat', {'color': 'c-green', 'size': 'c-small'})
    self.assertParseEqual(
      p.parse('feed the small green cat'),
      [logic.Description('c-action-request',
                         {'action': 'c-feed', 'theta-object': cat_desc})])
    green_dog_desc = logic.Description(
      'c-dog', {'color': 'c-green', 'size': 'c-small'})
    self.assertParseEqual(
      p.parse('feed the small green dog'),
      [logic.Description('c-action-request',
                         {'action': 'c-feed',
                          'theta-object': green_dog_desc})])
    green_itchy_dog_desc = logic.Description(
      'c-dog',
      {'flea-status': 'c-flea-bitten', 'color': 'c-green', 'size': 'c-small'})
    self.assertParseEqual(
      p.parse('feed the small green itchy dog'),
      [logic.Description('c-action-request',
                         {'action': 'c-feed',
                          'theta-object': green_itchy_dog_desc})])

  def test_examples(self):
    """More tests."""
    kb = logic.PropKB()
    kb.tell(logic.expr('ISA(c-start-over, c-action)'))
    kb.tell(logic.expr('ISA(c-hog, c-name)'))
    kb.tell(logic.expr('ISA(c-process, c-thing)'))
    kb.tell(logic.expr('ISA(c-nine-line, c-process)'))
    kb.tell(logic.expr('ISA(c-talk-on, c-process)'))
    kb.tell(logic.expr('$constraint(c-action-request, addressee, c-name)'))
    kb.tell(logic.expr('$constraint(c-action-request, action, c-action)'))
    kb.tell(logic.expr('$constraint(c-action-request, object, c-thing)'))
    p = parser.ConceptualParser(kb)
    p.add_phrasal_pattern(logic.expr('c-hog'), 'hog')
    p.add_phrasal_pattern(logic.expr('c-start-over'), 'start [over|again]')
    p.add_phrasal_pattern(logic.expr('c-start-over'), 'restart')
    p.add_phrasal_pattern(logic.expr('c-nine-line'), '[nine|9] line')
    p.add_phrasal_pattern(logic.expr('c-nine-line'), '[nineline|9line]')
    p.add_phrasal_pattern(logic.expr('c-talk-on'), 'talk on')
    p.add_phrasal_pattern(logic.expr('c-talk-on'), 'talkon')
    p.add_phrasal_pattern(
      logic.expr('c-action-request'),
      ("?:<addressee> ?:[let's|let us|lets|why don't we] "
       "<action> ?:[with|on] ?:the <object>"))
    self.assertParseEqual(p.parse("hog, let's start over with the nine-line"),
                          [logic.Description('c-action-request',
                                             {'addressee': 'c-hog',
                                              'action': 'c-start-over',
                                              'object': 'c-nine-line'})])
    self.assertParseEqual(p.parse("hog, why don't we start over with talk-on"),
                          [logic.Description('c-action-request',
                                             {'addressee': 'c-hog',
                                              'action': 'c-start-over',
                                              'object': 'c-talk-on'})])
    self.assertParseEqual(p.parse('hog, start over with the nine-line'),
                          [logic.Description('c-action-request',
                                             {'addressee': 'c-hog',
                                              'action': 'c-start-over',
                                              'object': 'c-nine-line'})])
    self.assertParseEqual(p.parse('restart the nine-line'),
                          [logic.Description('c-action-request',
                                             {'action': 'c-start-over',
                                              'object': 'c-nine-line'})])
    self.assertParseEqual(p.parse('restart the 9-line'),
                          [logic.Description('c-action-request',
                                             {'action': 'c-start-over',
                                              'object': 'c-nine-line'})])
    self.assertParseEqual(p.parse('restart the 9line'),
                          [logic.Description('c-action-request',
                                             {'action': 'c-start-over',
                                              'object': 'c-nine-line'})])

  def test_examples2(self):
    """Even more tests."""
    kb = logic.PropKB()
    kb.tell(logic.expr('ISA(c-start-over, c-action)'))
    kb.tell(logic.expr('ISA(c-hog, c-name)'))
    kb.tell(logic.expr('ISA(c-process, c-thing)'))
    kb.tell(logic.expr('ISA(c-nine-line, c-process)'))
    kb.tell(logic.expr('ISA(c-talk-on, c-process)'))
    kb.tell(logic.expr('$constraint(c-action-request, addressee, c-name)'))
    kb.tell(logic.expr('$constraint(c-action-request, action, c-action)'))
    kb.tell(logic.expr('$constraint(c-action-request, object, c-thing)'))
    p = parser.ConceptualParser(kb)
    p.add_phrasal_pattern(logic.expr('c-hog'), 'hog')
    p.add_phrasal_pattern(logic.expr('c-start-over'), 'start [over|again]')
    p.add_phrasal_pattern(logic.expr('c-start-over'), 'restart')
    p.add_phrasal_pattern(logic.expr('c-nine-line'), '[nine|9] line')
    p.add_phrasal_pattern(logic.expr('c-nine-line'), 'nineline')
    p.add_phrasal_pattern(logic.expr('c-nine-line'), '9line')
    p.add_phrasal_pattern(logic.expr('c-talk-on'), 'talk on')
    p.add_phrasal_pattern(logic.expr('c-talk-on'), 'talkon')
    p.add_phrasal_pattern(
      logic.expr('c-action-request'),
      ("?:?addressee ?:[let's|let us|lets|why don't we] "
       "?action ?:[with|on] ?:the ?object"))
    self.assertParseEqual(p.parse('hog'), [logic.Description('c-hog')])
    self.assertParseEqual(
      p.parse('9 line'), [logic.Description('c-nine-line')])
    self.assertParseEqual(
      p.parse('start again'), [logic.Description('c-start-over')])
    self.assertParseEqual(p.parse("hog, let's start over with the nine-line"),
                          [logic.Description('c-action-request',
                                             {'addressee': 'c-hog',
                                              'action': 'c-start-over',
                                              'object': 'c-nine-line'})])
    self.assertParseEqual(p.parse("hog, why don't we start over with talk-on"),
                          [logic.Description('c-action-request',
                                             {'addressee': 'c-hog',
                                              'action': 'c-start-over',
                                              'object': 'c-talk-on'})])
    self.assertParseEqual(p.parse('hog, start over with the nine-line'),
                          [logic.Description('c-action-request',
                                             {'addressee': 'c-hog',
                                              'action': 'c-start-over',
                                              'object': 'c-nine-line'})])
    self.assertParseEqual(p.parse('restart the nine-line'),
                          [logic.Description('c-action-request',
                                             {'action': 'c-start-over',
                                              'object': 'c-nine-line'})])
    self.assertParseEqual(p.parse('restart the 9-line'),
                          [logic.Description('c-action-request',
                                             {'action': 'c-start-over',
                                              'object': 'c-nine-line'})])
    self.assertParseEqual(p.parse('restart the 9line'),
                          [logic.Description('c-action-request',
                                             {'action': 'c-start-over',
                                              'object': 'c-nine-line'})])

  def test_pre_parser(self):
    """Test the pre-parser."""
    kb = logic.PropKB()
    p = parser.ConceptualParser(kb)
    self.assertParseEqual(p.parse('123'),
                          [logic.Description('c-number',
                                             {'value': 123})])
    self.assertParseEqual(p.parse('0'),
                          [logic.Description('c-number',
                                             {'value': 0})])
    self.assertParseEqual(p.parse('01'),
                          [logic.Description('c-number',
                                             {'value': 1})])
    self.assertParseEqual(p.parse('019'),
                          [logic.Description('c-number',
                                             {'value': 19})])
    self.assertParseEqual(p.parse('5'),
                          [logic.Description('c-number',
                                             {'value': 5})])

    # FIXME: These parse into digits--need to fix this inconsistency.
    # self.assertParseEqual(p.parse('one'),
    #                       [logic.Description('c-number',
    #                                          {'value': 1})])
    # self.assertParseEqual(p.parse('seven'),
    #                       [logic.Description('c-number',
    #                                          {'value': 7})])

    self.assertParseEqual(p.parse('HK'),
                          [logic.Description('c-mgrs-square',
                                             {'letters': 'hk'})])
    self.assertParseEqual(p.parse('XY'),
                          [logic.Description('c-mgrs-square',
                                             {'letters': 'xy'})])

  def test_fill_slots_at_most_once(self):
    """Test that slots get filled at most once."""
    kb = logic.PropKB()
    p = parser.ConceptualParser(kb)
    kb.tell(logic.expr('ISA(c-red, c-color)'))
    p.add_phrasal_pattern(logic.expr('c-red'), 'red')
    kb.tell(logic.expr('ISA(c-blue, c-color)'))
    p.add_phrasal_pattern(logic.expr('c-blue'), 'blue')
    kb.tell(logic.expr('$constraint(c-thing, color, c-color)'))
    p.add_phrasal_pattern(logic.expr('c-thing'), '[thing|{color} {:head}]')
    self.assertParseEqual(
      p.parse('red thing'), [logic.Description('c-thing', {'color': 'c-red'})])
    self.assertParseEqual(p.parse('red blue thing'), [])


class IndexedConceptParserTest (testutils.KBTestCase):
  def testIndexSetPatternParser(self):
    """Test indexset parsing."""
    p = parser.IndexSetPatternParser(None)
    # Basic stuff -- simple tokens, concepts and required tokens.
    index_set = p.parse(None, 'big dog')
    self.assertEqual(index_set.indices, ['big', 'dog'])
    self.assertEqual(index_set.required_indices, [])

    index_set = p.parse(None, 'big $dog')
    self.assertEqual(index_set.indices, ['big', logic.expr('dog')])
    self.assertEqual(index_set.required_indices, [])

    index_set = p.parse(None, '!big !dog')
    self.assertEqual(index_set.indices, ['big', 'dog'])
    self.assertEqual(index_set.required_indices, ['big', 'dog'])

    index_set = p.parse(None, '!big black !$dog')
    self.assertEqual(index_set.indices, ['big', 'black', logic.expr('dog')])
    self.assertEqual(index_set.required_indices, ['big', logic.expr('dog')])

    # Try something a little fancier -- slot references.
    kb = logic.PropKB()
    p = parser.IndexSetPatternParser(kb)
    kb.tell(logic.expr('$constraint(c-dog, color, c-color)'))
    kb.tell(logic.expr('$constraint(c-dog, size, c-size)'))

    index_set = p.parse('c-dog', '!{size} {color} dog')
    self.assertEqual(index_set.target_concept, logic.expr('c-dog'))
    self.assertEqual(index_set.indices,
                     [logic.expr('c-size'), logic.expr('c-color'), 'dog'])
    self.assertEqual(index_set.required_indices, [logic.expr('c-size')])
    self.assertEqual(
      index_set.slots,
      [['size', logic.expr('c-size')], ['color', logic.expr('c-color')]])

    # Now test various weird cases
    index_set = p.parse(None, '   !big black     !$dog   ')
    self.assertEqual(index_set.indices, ['big', 'black', logic.expr('dog')])
    self.assertEqual(index_set.required_indices, ['big', logic.expr('dog')])

    # FIXME: Some of these raise SyntaxError because they fail inside
    # of eval.
    self.assertRaises(parser.Error, lambda: p.parse(None, '!'))
    self.assertRaises(SyntaxError, lambda: p.parse(None, '$'))
    self.assertRaises(SyntaxError, lambda: p.parse(None, '!$$!'))
    self.assertRaises(SyntaxError, lambda: p.parse(None, '$!$!'))

    self.assertRaises(SyntaxError, lambda: p.parse('c-dog', '{}'))
    self.assertRaises(parser.Error, lambda: p.parse('c-dog', '{!}'))
    self.assertRaises(parser.Error, lambda: p.parse('c-dog', '{foo} {foo}'))
    self.assertEqual(p.parse(None, ''), parser.IndexSet())
    self.assertEqual(p.parse(None, '   '), parser.IndexSet())

  def test_icp_with_literal_tokens(self):
    """Very simple tests with simple ICP index sets containing only
    literal tokens.
    """
    kb = logic.PropKB()
    p = parser.IndexedConceptParser(kb)
    p.add_index_set('c-number', 'one two three numberorjohn')
    p.add_index_set('c-john', 'john numberorjohn')

    # Check some of the lower-level computations on which scoring is
    # based.
    self.assertEqual(p.target_concept_cardinality('john'), 1)
    self.assertEqual(p.target_concept_cardinality('numberorjohn'), 2)
    self.assertApproxEqual(p.probability_of_index('john'), 0.5, 0.001)
    self.assertApproxEqual(p.probability_of_index('numberorjohn'), 1.0, 0.001)
    self.assertApproxEqual(p.information_value('john'), 1.0, 0.001)
    self.assertApproxEqual(p.information_value('numberorjohn'), 0.0, 0.001)
    self.assertEqual(len(p.unique_target_concepts), 2)

    # Parsing nothing should result in nothing.
    results = p.parse('')
    self.assertEqual(len(results), 0)

    # Should result in a c-number.
    results = p.parse('one')
    self.assertEqual(len(results), 1)
    self.assertEqual(results[0].target_concept, logic.expr('c-number'))

    # 'john' should have a better score than 'john boy' for c-john.
    results1 = p.parse('john')
    self.assertEqual(len(results1), 1)
    self.assertEqual(results1[0].target_concept, logic.expr('c-john'))

    results2 = p.parse('john boy woo bloo')
    self.assertEqual(len(results2), 1)
    self.assertEqual(results2[0].target_concept, logic.expr('c-john'))
    self.assertTrue(results1[0].score > results2[0].score)

    # 'one two' should have a better score than 'one' for c-number.
    results1 = p.parse('one')
    self.assertEqual(len(results1), 1)
    self.assertEqual(results1[0].target_concept, logic.expr('c-number'))

    results2 = p.parse('one two')
    self.assertEqual(len(results2), 1)
    self.assertEqual(results2[0].target_concept, logic.expr('c-number'))
    self.assertTrue(results1[0].score < results2[0].score)

    # 'numberorjohn' should result in both c-number and c-john, but
    # c-john should have a higher score.
    results = p.parse('numberorjohn')
    self.assertEqual(len(results), 2)
    self.assertEqual(results[0].target_concept, logic.expr('c-john'))
    self.assertEqual(results[1].target_concept, logic.expr('c-number'))
    self.assertTrue(results[0].score > results[1].score)

  def test_icp_with_concepts(self):
    """Slightly more complicated tests with ICP index sets that
    contain terminals and concepts.
    """
    kb = logic.PropKB()
    p = parser.IndexedConceptParser(kb)
    kb.tell(logic.expr('ISA(c-red, c-color)'))
    kb.tell(logic.expr('ISA(c-green, c-color)'))
    kb.tell(logic.expr('ISA(c-blue, c-color)'))
    kb.tell(logic.expr('ISA(c-purple, c-color)'))
    p.add_phrasal_pattern(logic.expr('c-red'), 'red')
    p.add_phrasal_pattern(logic.expr('c-green'), 'green')
    p.add_phrasal_pattern(logic.expr('c-blue'), 'blue')
    p.add_phrasal_pattern(logic.expr('c-purple'), 'purple')
    p.add_index_set('c-color', '$c-red $c-green $c-blue $c-purple')

    results1 = p.parse('red')
    self.assertEqual(len(results1), 1)
    self.assertEqual(results1[0].target_concept, logic.expr('c-color'))

    results2 = p.parse('red green blue purple')
    self.assertEqual(len(results2), 1)
    self.assertEqual(results2[0].target_concept, logic.expr('c-color'))
    self.assertTrue(results2[0].score > results1[0].score)

    # Setup a little hierarchy, C -> B -> A.
    kb.tell(logic.expr('ISA(c-b, c-a)'))
    kb.tell(logic.expr('ISA(c-c, c-b)'))
    p.add_phrasal_pattern(logic.expr('c-a'), 'a')
    p.add_phrasal_pattern(logic.expr('c-b'), 'b')
    p.add_phrasal_pattern(logic.expr('c-c'), 'c')
    p.add_index_set('c-1', '$c-a')
    p.add_index_set('c-2', '$c-b')
    p.add_index_set('c-3', '$c-c')

    r1 = p.parse('a')
    r2 = p.parse('b')
    r3 = p.parse('c')
    self.assertEqual(len(r1), 1)
    self.assertEqual(len(r2), 2)
    self.assertEqual(len(r3), 3)

    # Most specific result should always have the highest score and
    # appear first, followed by less specific results.
    self.assertEqual(map(lambda r: r.target_concept, r1),
                     [logic.expr('c-1')])
    self.assertEqual(map(lambda r: r.target_concept, r2),
                     [logic.expr('c-2'), logic.expr('c-1')])
    self.assertEqual(map(lambda r: r.target_concept, r3),
                     [logic.expr('c-3'), logic.expr('c-2'), logic.expr('c-1')])

  def test_icp_slot_filling(self):
    """Slightly more complicated tests with ICP index sets that
    contain terminals and concepts.
    """
    kb = logic.PropKB()
    p = parser.IndexedConceptParser(kb)
    kb.tell(logic.expr('ISA(c-red, c-color)'))
    kb.tell(logic.expr('ISA(c-green, c-color)'))
    kb.tell(logic.expr('ISA(c-big, c-size)'))
    kb.tell(logic.expr('ISA(c-small, c-size)'))
    kb.tell(logic.expr('$constraint(c-dog, color, c-color)'))
    kb.tell(logic.expr('$constraint(c-dog, size, c-size)'))
    p.add_phrasal_pattern(logic.expr('c-red'), 'red')
    p.add_phrasal_pattern(logic.expr('c-green'), 'green')
    p.add_phrasal_pattern(logic.expr('c-big'), 'big')
    p.add_phrasal_pattern(logic.expr('c-small'), 'small')
    p.add_index_set('c-dog', '{color} !{size} !dog')

    results = p.parse('dog')
    self.assertEqual(len(results), 0)
    results = p.parse('big red')
    self.assertEqual(len(results), 0)
    results = p.parse('big red dog')
    self.assertEqual(len(results), 1)
    self.assertParseEqual(
      results,
      [logic.Description('c-dog', {'size': 'c-big', 'color': 'c-red'})])
    results = p.parse('big dog')
    self.assertEqual(len(results), 1)
    self.assertParseEqual(
      results, [logic.Description('c-dog', {'size': 'c-big'})])

    # Now multiple slots of the same type.
    kb.tell(logic.expr('$constraint(c-thing, slot-a, c-size)'))
    kb.tell(logic.expr('$constraint(c-thing, slot-b, c-size)'))
    p.add_index_set('c-thing', '{slot-a} {slot-b}')
    results = p.parse('big small')
    self.assertParseEqual(
      results,
      [logic.Description('c-thing', {'slot-a': 'c-big', 'slot-b': 'c-small'})])

  def test_simple_icp3(self):
    """Test indexset parsing."""
    kb = logic.PropKB()
    p = parser.IndexedConceptParser(kb)
    kb.tell(logic.expr('ISA(c-red, c-color)'))
    kb.tell(logic.expr('ISA(c-green, c-color)'))
    kb.tell(logic.expr('ISA(c-blue, c-color)'))
    p.add_phrasal_pattern(logic.expr('c-red'), 'red')
    p.add_phrasal_pattern(logic.expr('c-green'), 'green')
    p.add_phrasal_pattern(logic.expr('c-blue'), 'blue')
    p.add_index_set('c-color', 'color $c-red $c-green $c-blue')

    results1 = p.parse('color')
    self.assertEqual(len(results1), 1)
    self.assertEqual(results1[0].target_concept, logic.expr('c-color'))
    results2 = p.parse('color red green blue')
    self.assertEqual(len(results2), 1)
    self.assertEqual(results2[0].target_concept, logic.expr('c-color'))
    self.assertTrue(results2[0].score > results1[0].score)

    p.add_index_set('c-john', '!john boy')
    results = p.parse('boy')
    self.assertEqual(len(results), 0)

    results1 = p.parse('john')
    self.assertEqual(len(results1), 1)
    self.assertEqual(results1[0].target_concept, logic.expr('c-john'))

    results2 = p.parse('john boy')
    self.assertEqual(len(results2), 1)
    self.assertEqual(results2[0].target_concept, logic.expr('c-john'))
    self.assertTrue(results2[0].score > results1[0].score)

    kb.tell(logic.expr('ISA(c-purple, c-color)'))
    p.add_phrasal_pattern(logic.expr('c-purple'), 'purple')
    # Tricky!  Note that we're adding a second indexset to c-color,
    # and this one has a required concept.
    p.add_index_set('c-color', '!$c-purple blah')

    # If we have one token from the new indexset but not the required
    # token, it should not parse at all.
    results = p.parse('blah')
    self.assertEqual(len(results), 0)

    # If we have the required token, it should parse.
    results = p.parse('purple')
    self.assertEqual(len(results), 1)
    self.assertEqual(results[0].target_concept, logic.expr('c-color'))

    # If we have a token from the old indexset, we don't need the
    # required token from the new indexset.
    results = p.parse('red')
    self.assertEqual(len(results), 1)
    self.assertEqual(results[0].target_concept, logic.expr('c-color'))


if __name__ == '__main__':
  unittest.main()
