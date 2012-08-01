# Copyright 2012 Energid Technologies

import os.path
import unittest

from energid_nlp import parser
from energid_nlp import logic
from energid_nlp import fdl
from energid_nlp import generation

from energid_nlp.tests import testutils


class FDLTest (testutils.KBTestCase):
  def test_string_parsing(self):
    """Test parsing FDL from strings."""
    xml = """
<fdl>

<lexicon>
  <lexeme>
    <!-- Short for 'roger'. -->
    <spelling>rog</spelling>
    <part-of-speech>interjection</part-of-speech>
    <phonemes>r ao jh</phonemes>
    <phonemes>r ao d</phonemes>
  </lexeme>
  <lexeme>
    <!-- Whatever, this is just a test. -->
    <spelling>john</spelling>
    <part-of-speech>noun</part-of-speech>
    <phonemes>j uh n</phonemes>
  </lexeme>
</lexicon>


<frame id='c-action-request'>
  <parent id='c-utterance' />

  <constraint slot='addressee' type='c-name' />
  <constraint slot='action' type='c-action' />
  <constraint slot='object' type='c-thing' />

  <phrase>
    {addressee} ?:[let's|let us|why don't we]
    {action} ?:[with|on] ?:the {object}
  </phrase>
</frame>

<frame id='c-hog'>
  <parent id='c-name' />
  <phrase>hog</phrase>
</frame>

<frame id='c-restart'>
  <parent id='c-action' />
  <phrase>[restart|start over]</phrase>
</frame>

<frame id='c-talkon'>
  <parent id='c-thing' />
  <phrase>[talk on|talkon]</phrase>
</frame>

<frame id='i-petunia'>
  <parent id='c-cat' instanceof='true' />
  <slot name='color' value='i-gray' />
  <generate>Petunia</generate>
</frame>

</fdl>
"""
    kb = logic.PropKB()
    cp_parser = parser.ConceptualParser(kb)
    icp_parser = parser.IndexedConceptParser(kb)
    fdl_handler = fdl.BaseFrameHandler(kb, cp_parser, icp_parser)
    fdl_parser = fdl.FDLParser(fdl_handler)
    self.assertEqual(fdl_parser.parse_fdl_string(xml), True)
    self.assertEqual(
      kb.isa(logic.expr('c-action-request'), logic.expr('c-utterance')), True)
    self.assertEqual(
      cp_parser.slot_constraint(
        logic.expr('c-action-request'), logic.expr('addressee')),
      logic.expr('c-name'))
    self.assertEqual(
      cp_parser.slot_constraint(
        logic.expr('c-action-request'), logic.expr('action')),
      logic.expr('c-action'))
    self.assertEqual(
      cp_parser.slot_constraint(
        logic.expr('c-action-request'), logic.expr('object')),
      logic.expr('c-thing'))
    self.assertParseEqual(cp_parser.parse('hog'), [logic.Description('c-hog')])
    self.assertParseEqual(
      cp_parser.parse('restart'), [logic.Description('c-restart')])
    self.assertParseEqual(
      cp_parser.parse('start over'), [logic.Description('c-restart')])
    self.assertParseEqual(
      cp_parser.parse('talkon'), [logic.Description('c-talkon')])
    self.assertParseEqual(
      cp_parser.parse('talk on'), [logic.Description('c-talkon')])
    self.assertParseEqual(cp_parser.parse('hog let us restart the talk on'),
                          [logic.Description('c-action-request',
                                             {'action': 'c-restart',
                                              'addressee': 'c-hog',
                                              'object': 'c-talkon'})])
    self.assertParseEqual(
      cp_parser.parse("hog why don't we start over with the talk on"),
      [logic.Description('c-action-request',
                         {'action': 'c-restart',
                          'addressee': 'c-hog',
                          'object': 'c-talkon'})])
    self.assertAllBindingsEqual(kb.ask_all(logic.expr('ISINSTANCE(?x)')),
                                [{'?x': 'i-petunia'}])
    self.assertAllBindingsEqual(
      kb.ask_all(logic.expr('INSTANCEOF(i-petunia, ?x)')),
      [{'?x': 'c-cat'}])
    self.assertAllBindingsEqual(kb.ask_all(logic.expr('color(i-petunia, ?c)')),
                                [{'?c': 'i-gray'}])
    self.assertEqual('color' in kb.heritable, True)

    generator = generation.Generator(kb)
    self.assertEqual(generator.generate(logic.expr('i-petunia')), 'Petunia')
    self.assertEqual(len(fdl_handler.lexemes), 2)
    self.assertEqual(fdl_handler.lexemes[0].spelling, 'rog')
    self.assertEqual(fdl_handler.lexemes[0].part_of_speech, 'interjection')
    self.assertEqual(
      fdl_handler.lexemes[0].phonetic_pronunciations, ['r ao jh', 'r ao d'])
    self.assertEqual(fdl_handler.lexemes[1].spelling, 'john')
    self.assertEqual(fdl_handler.lexemes[1].part_of_speech, 'noun')
    self.assertEqual(
      fdl_handler.lexemes[1].phonetic_pronunciations, ['j uh n'])

  def test_file_parsing(self):
    """Test parsing FDL from files."""
    kb = logic.PropKB()
    cp_parser = parser.ConceptualParser(kb)
    icp_parser = parser.IndexedConceptParser(kb)
    fdl_handler = fdl.BaseFrameHandler(kb, cp_parser, icp_parser)
    fdl_parser = fdl.FDLParser(fdl_handler)
    test_fdl_path = os.path.join(os.path.dirname(__file__), 'test.fdl')
    self.assertEqual(fdl_parser.parse_fdl_file(test_fdl_path), True)
    self.assertParseEqual(cp_parser.parse('hog let us restart the talk on'),
                          [logic.Description('c-action-request',
                                             {'action': 'c-restart',
                                              'addressee': 'c-hog',
                                              'object': 'c-talkon'})])
    self.assertParseEqual(
      cp_parser.parse("hog why don't we start over with the talk on"),
      [logic.Description('c-action-request',
                         {'action': 'c-restart',
                          'addressee': 'c-hog',
                          'object': 'c-talkon'})])


if __name__ == '__main__':
  unittest.main()
