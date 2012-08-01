# Copyright 2012 Energid Technologies

"""Implements a parser and reader for the Frame Description Language
(FDL).
"""

import logging
import os.path
import sys
from xml.dom import minidom
import xml.parsers.expat

from energid_nlp import logic


class Error(Exception):
  pass


class CPTest:
  """A test for the ConceptualParser."""
  EXACT = 'exact'
  ONEOF = 'one-of'

  def __init__(self, phrase, result, match_type):
    self.phrase = phrase
    self.result = result
    self.match_type = match_type

  def succeeded(self, parses):
    if self.match_type == CPTest.EXACT:
      return len(parses) == 1 and parses[0].base == logic.expr(self.result)
    elif self.match_type == CPTest.ONEOF:
      for parse in parses:
        if parse.base == logic.expr(self.result):
          return True
      return False


class ICPTest:
  """A test for the IndexedConceptParser."""
  def __init__(self, phrase, result):
    self.phrase = phrase
    self.result = result

  def succeeded(self, parses):
    return len(parses) > 0 and parses[0].base == logic.expr(self.result)


class Lexeme:
  PARTS_OF_SPEECH = (
    'unknown', 'noun', 'verb', 'modifier', 'function', 'interjection')

  def __init__(self, spelling, part_of_speech):
    self.spelling = spelling
    part_of_speech = part_of_speech.lower()
    if not part_of_speech in self.PARTS_OF_SPEECH:
      raise Error('Part of speech is %r, but must be one of %s.' % (
        part_of_speech, self.PARTS_OF_SPEECH))
    self.part_of_speech = part_of_speech
    self.phonetic_pronunciations = []

  def add_phonetic_pronunciation(self, phonemes):
    self.phonetic_pronunciations.append(phonemes)


class FDLParser:
  def __init__(self, frame_handler):
    self.frame_handler = frame_handler
    self.debug = 0
    self.cp_tests = []
    self.icp_tests = []
    self.loaded_files = []  # keep track of files loaded on this run

  def get_text(self, node):
    rc = ''
    for child in node.childNodes:
      if child.nodeType == child.TEXT_NODE:
        rc = rc + child.data
    return rc

  def parse_fdl_file(self, path, debug=0):
    # record the file we started with
    self.loaded_files = [os.path.realpath(path)]
    # parse and run
    doc = minidom.parse(path).documentElement
    return self.parse_fdl_doc(doc, debug, path)

  def parse_fdl_string(self, string, debug=0):
    doc = minidom.parseString(string).documentElement
    return self.parse_fdl_doc(doc, debug)

  def parse_fdl_doc(self, doc, debug, source_file=''):
    self.debug = debug

    # call helper function.
    self._parse_fdl_doc_helper(doc, source_file)

    if debug > 0:
      print 'Processed %s files.' % (len(self.loaded_files),)

    return True

  def handle_include(self, include_node, source_file):
    raw_file = include_node.attributes['file'].value
    merge_file = os.path.join(os.path.dirname(source_file), raw_file)
    include_file = os.path.realpath(merge_file)
    # sanity check, does file exist?
    if (not os.path.exists(include_file)):
      logging.warning(
        'Include file %r does not exist, knowledge may be incomplete.',
        raw_file)
      logging.warning('Expected location: %r', include_file)
    elif (include_file in self.loaded_files):
      if self.debug > 0:
        print 'Skipping include file \'%s\' (already processed).' % (raw_file,)

    else:
      if self.debug > 0:
        print 'Processing include file: \'%s\'' % (raw_file,)
      # record new file
      self.loaded_files += [include_file]
      # parse it
      try:
        new_doc = minidom.parse(include_file).documentElement
      except xml.parsers.expat.ExpatError, e:
        raise Error('Error while parsing %r: %s' % (include_file, e))
      self._parse_fdl_doc_helper(new_doc, merge_file)

  def handle_lexicon(self, lexicon):
    if self.debug > 0:
      print 'Processing lexicon'

    lexemes = []
    for lexeme_node in lexicon.getElementsByTagName('lexeme'):
      # Get spelling
      spelling_nodes = lexeme_node.getElementsByTagName('spelling')
      if len(spelling_nodes) != 1:
        raise Error('Must have exactly one <spelling> per <lexeme>.')
      spelling = self.get_text(spelling_nodes[0])

      # Get part-of-speech
      part_of_speech_nodes = lexeme_node.getElementsByTagName('part-of-speech')
      if len(part_of_speech_nodes) > 1:
        raise Error(
          'There are %s <part-of-speech> tags; there must be only 1.' % (
            len(part_of_speech_nodes),))
      for pos_node in part_of_speech_nodes:
        pos = self.get_text(pos_node)

      lexeme = Lexeme(spelling, pos)

      # Get phonemes
      phonemes_nodes = lexeme_node.getElementsByTagName('phonemes')
      if len(phonemes_nodes) == 0:
        raise Error('There must be at least one <phonemes> tag.')
      for phonemes_node in phonemes_nodes:
        phonemes = self.get_text(phonemes_node)
        lexeme.add_phonetic_pronunciation(phonemes)
      lexemes.append(lexeme)

    if self.frame_handler is not None:
      self.frame_handler.handle_lexicon(lexemes)

  def handle_frame_parent(self, parent_node):
    parent = parent_node.attributes['id'].value
    is_instance = False
    if (parent_node.hasAttribute('instanceof') and
        parent_node.attributes['instanceof'].value == 'true'):
      if self.debug > 1:
        print '  is an instance'
      is_instance = True
    if self.debug > 1:
      print '  Parent is %s' % (parent,)
    if self.debug > 1:
      print '  Is an instance of %s' % (parent,)
    return [parent, is_instance]

  def handle_frame_slots(self, slot_nodes):
    slots = {}
    for slot_node in slot_nodes:
      if (not slot_node.hasAttribute('name')) or \
         (not slot_node.hasAttribute('value')):
        raise Error('%s is not a valid slot.' % (slot_node.toxml(),))
      slot = slot_node.attributes['name'].value
      value = slot_node.attributes['value'].value
      slots[slot] = value
    if self.debug > 1:
      print '  Slots are %s' % (slots,)
    return slots

  def handle_frame_constraints(self, constraint_nodes):
    constraints = {}
    for constraint_node in constraint_nodes:
      if (not constraint_node.hasAttribute('slot')) or \
         (not constraint_node.hasAttribute('type')):
        raise Error('%s is not a valid constraint.' % (
          constraint_node.toxml(),))
      slot = constraint_node.attributes['slot'].value
      req_type = constraint_node.attributes['type'].value
      constraints[slot] = req_type
    if self.debug > 1:
      print '  Constraints are %s' % (constraints,)
    return constraints

  def handle_frame_phrases(self, phrase_nodes):
    phrases = []
    for phrase_node in phrase_nodes:
      phrases.append(self.get_text(phrase_node))
    if self.debug > 1:
      print '  Phrases are %s' % (phrases,)
    return phrases

  def handle_frame_indexsets(self, indexset_nodes):
    indexsets = []
    for indexset_node in indexset_nodes:
      indexsets.append(self.get_text(indexset_node))
    if self.debug > 1:
      print '  Index sets are %s' % (indexsets,)
    return indexsets

  def handle_frame_generates(self, generate_nodes):
    generates = []
    for generate_node in generate_nodes:
      generates.append(self.get_text(generate_node))
    if self.debug > 1:
      print '  Generation templates are %s' % (generates,)
    return generates

  def handle_frame_testphrases(self, class_name, testphrase_nodes):
    conceptual_parser_tests = []
    for testphrase_node in testphrase_nodes:
      phrase = self.get_text(testphrase_node)
      result = class_name
      if testphrase_node.hasAttribute('match-type'):
        match_type = testphrase_node.attributes['match-type'].value
      else:
        match_type = CPTest.EXACT
      conceptual_parser_tests.append(CPTest(phrase, result, match_type))
    return conceptual_parser_tests

  def handle_frame_testindices(self, class_name, testindex_nodes):
    icp_tests = []
    for testindex_node in testindex_nodes:
      phrase = self.get_text(testindex_node)
      result = class_name
      icp_tests.append(ICPTest(phrase, result))
    return icp_tests

  def handle_frame(self, frame):
    class_name = frame.attributes['id'].value

    if self.debug > 0:
      print 'Processing frame %s' % (class_name,)

    # Handle <parent>
    parent_nodes = frame.getElementsByTagName('parent')
    if len(parent_nodes) > 1:
      raise Error('%s cannot have more than one parent.' % (class_name,))
    [parent, is_instance] = [None, False]
    if len(parent_nodes) > 0:
      [parent, is_instance] = self.handle_frame_parent(parent_nodes[0])

    # Handle <slot>
    slots = self.handle_frame_slots(frame.getElementsByTagName('slot'))

    # Handle <constraint>
    constraints = self.handle_frame_constraints(
      frame.getElementsByTagName('constraint'))

    # Handle <phrase>
    phrases = self.handle_frame_phrases(frame.getElementsByTagName('phrase'))

    # Handle <indexset>
    indexsets = self.handle_frame_indexsets(
      frame.getElementsByTagName('indexset'))

    # Handle <generate>
    generates = self.handle_frame_generates(
      frame.getElementsByTagName('generate'))

    # Handle <testphrase>
    cp_tests = self.handle_frame_testphrases(
      class_name, frame.getElementsByTagName('testphrase'))

    # Handle <testindex>
    icp_tests = self.handle_frame_testindices(
      class_name, frame.getElementsByTagName('testindex'))

    self.do_handle_frame(
      class_name=class_name, parent=parent, is_instance=is_instance,
      slots=slots, constraints=constraints, phrases=phrases,
      indexsets=indexsets, generates=generates, cp_tests=cp_tests,
      icp_tests=icp_tests)

  def _parse_fdl_doc_helper(self, doc, source_file):
    """Helper method which allows parse_fdl_doc to recurse while
    keeping timing and total counts for all objects straight, and only
    executing the tests once at end.
    """
    # As first step, handle included files.  It would probably be
    # better to handle these in-line, so you could set up some
    # definitions and then include another file which required them,
    # but this will do for now.
    #
    # Files are relative to current file.
    for node in doc.childNodes:
      if node.nodeType == node.ELEMENT_NODE:
        if node.localName == 'lexicon':
          self.handle_lexicon(node)
        elif node.localName == 'frame':
          self.handle_frame(node)
        elif node.localName == 'include':
          self.handle_include(node, source_file)
        else:
          logging.warning('Unknown element %r', node.localName)

  def do_handle_frame(self, **args):
    if self.frame_handler is not None:
      self.frame_handler.handle_fdl_frame(args)
    self.cp_tests = self.cp_tests + args['cp_tests']
    self.icp_tests = self.icp_tests + args['icp_tests']

  def run_test_phrases(self, parents, cp_parser, icp_parser):
    test_count = 0
    fail_count = 0

    results = self.run_test_cp_phrases(parents, cp_parser)
    test_count += results[0]
    fail_count += results[1]

    results = self.run_test_icp_phrases(parents, icp_parser)
    test_count += results[0]
    fail_count += results[1]

    if test_count > 0:
      print 'Ran %s parse tests, %s failed.' % (test_count, fail_count)

  def class_is_one_of(self, kb, c, parents):
    for parent in parents:
      if kb.isa(c, parent):
        return True
    return False

  def run_test_icp_phrases(self, parents, parser):
    test_count = 0
    fail_count = 0
    last_class = None
    if parents:
      parents = [logic.expr(parent) for parent in parents]
    for test in self.icp_tests:
      class_name = test.result
      if (len(parents) > 0 and
          not self.class_is_one_of(parser.kb,
                                   logic.expr(class_name),
                                   parents)):
        continue
      phrase = test.phrase
      if class_name != last_class:
        sys.stdout.write('Class: %s\n' % class_name)
      last_class = class_name
      sys.stdout.write('  %-70s ==> ' % (phrase,))
      test_count = test_count + 1
      parses = parser.parse(str(phrase))
      if not test.succeeded(parses):
        sys.stdout.write('FAIL\n')
        print 'ICP parse test failure, %r should have parsed to a' % (phrase,)
        print 'Description of type %s.' % (class_name,)
        if len(parses) > 0:
          for parse in parses:
            sys.stdout.write('  score: %s ' % (parse.score,))
            parse.pprint()
        else:
          print 'None'
        fail_count = fail_count + 1
      else:
        sys.stdout.write('ok')
        if len(parses) == 1:
          sys.stdout.write(' [%f]\n' % (parses[0].score,))
        else:
          sys.stdout.write(' [delta %f]\n' % (
              parses[0].score - parses[1].score,))
    return [test_count, fail_count]

  def run_test_cp_phrases(self, parents, parser):
    test_count = 0
    fail_count = 0
    last_class = None
    if parents:
      parents = [logic.expr(parent) for parent in parents]
    for test in self.cp_tests:
      class_name = test.result
      if len(parents) > 0 and not self.class_is_one_of(parser.kb,
                                                       logic.expr(class_name),
                                                       parents):
        continue
      phrase = test.phrase
      if class_name != last_class:
        sys.stdout.write('Class: %s\n' % class_name)
      last_class = class_name
      sys.stdout.write('  %-70s ==> ' % (phrase,))
      test_count = test_count + 1
      parses = parser.parse(str(phrase))
      if not test.succeeded(parses):
        sys.stdout.write('FAIL\n')
        print 'CP parse test failure, %r should have parsed to a' % (phrase,)
        print 'Description of type %s.' % (class_name,)
        if len(parses) > 0:
          for parse in parses:
            parse.pprint()
        else:
          print 'None'
        fail_count = fail_count + 1
      else:
        sys.stdout.write('ok\n')

    return [test_count, fail_count]


class BaseFrameHandler:
  """A helpful base class for anything that needs to parse FDL and
  assert stuff to a KnowledgeBase and a Conceptual or Indexed Concept
  Parser.
  """
  def __init__(self, kb, cp, icp):
    self.kb = kb
    self.cp = cp
    self.icp = icp
    self.lexemes = []

  def handle_slots(self, frame, slots):
    class_name = frame['class_name']
    for name in slots:
      self.kb.define_fluent(logic.expr(name), inherited=True)
      slot_value_str = str(slots[name])
      if len(slot_value_str) > 0 and slot_value_str[0] == '{':
        slot_frame = parse_frame_literal(slot_value_str)
        self.handle_fdl_frame(slot_frame)
        slot_value = logic.Expr(slot_frame['class_name'])
      else:
        slot_value = logic.Expr(slot_value_str)
      self.kb.tell(logic.expr(name)(logic.expr(class_name), slot_value))

  def handle_constraints(self, frame, constraints):
    class_name = frame['class_name']
    for name in constraints:
      self.kb.tell(logic.expr('$constraint')(logic.expr(class_name),
                                             logic.expr(name),
                                             logic.expr(constraints[name])))

  def handle_indexsets(self, frame, indexsets):
    class_name = frame['class_name']
    for indexset in indexsets:
      self.icp.add_index_set(class_name, indexset)

  def handle_generates(self, frame, generates):
    class_name = frame['class_name']
    self.kb.define_fluent(logic.expr('$generate'), inherited=True)
    for template in generates:
      self.kb.tell(
        logic.expr('$generate')(logic.expr(class_name),
                                logic.Expr(str(template))))

  def handle_phrases(self, frame, phrases):
    class_name = frame['class_name']
    for phrase in phrases:
      self.cp.add_phrasal_pattern(logic.expr(class_name), phrase)
      self.icp.add_phrasal_pattern(logic.expr(class_name), phrase)

  def handle_fdl_frame(self, frame):
    class_name = frame['class_name']
    parent = frame.get('parent', None)
    slots = frame.get('slots', [])
    constraints = frame.get('constraints', [])

    if parent is not None:
      if frame['is_instance']:
        self.kb.tell(
          logic.expr('INSTANCEOF')(logic.expr(class_name),
                                   logic.expr(parent)))
      else:
        self.kb.tell(
          logic.expr('ISA')(logic.expr(class_name),
                            logic.expr(parent)))

    self.handle_slots(frame, slots)
    self.handle_constraints(frame, constraints)
    self.handle_phrases(frame, frame.get('phrases', []))
    self.handle_indexsets(frame, frame.get('indexsets', []))
    self.handle_generates(frame, frame.get('generates', []))

  def handle_lexicon(self, lexemes):
    self.lexemes = self.lexemes + lexemes


g_literal_counter = 0


def parse_frame_literal(frame_spec):
  global g_literal_counter
  frame = eval(frame_spec)
  if not 'class_name' in frame:
    frame['class_name'] = 'literal-%s' % (g_literal_counter,)
  g_literal_counter += 1
  return frame
