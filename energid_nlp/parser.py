#!/usr/bin/env python2.5

# Copyright 2012 Energid Technologies

"""Implements natural language parsers.

This module provides two types of natural language parsers:

1. ConceptualParser, a conceptual, frame-based parser in the style of
Charles Martin's Direct Memory Access Parser.

2. IndexedConceptParser, a less strict parser based on the parser in
Will Fitzgerald's thesis 'Building Embedded Conceptual Parsers'.
"""


from energid_nlp import logic
from energid_nlp import fdl
from energid_nlp import utils
from energid_nlp import stemmer

import logging
import re
import string
import sys
from copy import copy
import math
import StringIO

import pprint
import getopt


# We'll be using $constraint internally.
CONSTRAINT_EXPR = logic.expr('$constraint')


def tokenize(text):
  """Tokenizes a string.  Tokens consist of characters that are
  letters, digits or underscores.  This function is primarily intended
  for text typed directly by users or from a speech recognizer.
  """
  regex = re.compile(r'\W+')
  tokens = regex.split(text.lower())
  tokens = [token for token in tokens if token]
  return tokens


class Error(Exception):
  pass


class ParserBase:
  """A simple base class for the ConceptualParser and the
  IndexedConceptParser.  Really just provides a tokenize method and a
  parse method.
  """
  def __init__(self):
    self.debug = 0

  def tokenize(self, text):
    """Tokenizes a string."""
    tokens = tokenize(text.lower())
    if self.debug > 0:
      print 'Tokens: %s' % (tokens,)
    return tokens

  def parse_tokens(self, text, debug=0):
    raise NotImplementedError

  def parse(self, text, debug=0):
    """Parses a string.  Returns the list of valid parses."""
    if not isinstance(text, basestring):
      raise TypeError('%r is not a string.' % (text,))
    results = self.parse_tokens(self.tokenize(text), debug=debug)
    if len(results) > 1:
      results = utils.remove_duplicates(results)
    for result in results:
      result.text = text
    return results

Cardinals = {
  'zero': 0, 'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5,
  'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'niner': 9, 'ten': 10
  }


# ----------------------------------------
# Conceptual Parser
# ----------------------------------------

class ConceptualParser(ParserBase):
  """A DMAP-inspired conceptual memory parser."""
  def __init__(self, kb, stem=False):
    ParserBase.__init__(self)
    self.kb = kb
    self.kb.define_fluent(CONSTRAINT_EXPR, inherited=True)
    self.syntax_functions = {}
    self.preparsers = []
    self.anytime_predictions = {}
    self.dynamic_predictions = {}
    self.phrasal_patterns = {}
    self.phrasal_pattern_objects = {}
    self.references = []
    self.position = 0
    self.reference_callback = None
    # Stemming in the ConceptualParser is not well tested; It's really
    # only used when we're a servant of the ICP.
    self.stem = stem
    if stem:
      self.stemmer = stemmer.PorterStemmer()
    self.install_syntax_functions()
    self.install_preparsers()
    self.reset()

  def install_syntax_functions(self):
    """Installs the standard syntax directives."""
    self.syntax_functions[':head'] = head_prediction_generator
    self.syntax_functions[':optional'] = optional_prediction_generator
    self.syntax_functions[':sequence'] = sequence_prediction_generator
    self.syntax_functions[':any'] = any_prediction_generator

  def install_preparsers(self):
    def parse_c_number(token):
      return logic.Description(logic.expr('c-number'),
                               {logic.expr('value'): logic.expr(int(token))})
    self.add_preparser('[0-9]+', parse_c_number)

    def parse_cardinal(token):
      return logic.Description(
        logic.expr('c-digit'),
        {logic.expr('value'): logic.expr(Cardinals[token])})
    self.add_preparser('one|two|three|four|five|six|seven|eight|nine|ten|zero',
                       parse_cardinal)

    def parse_mgrs_square(token):
      return logic.Description(logic.expr('c-mgrs-square'),
                               {logic.expr('letters'): logic.expr(token)})
    self.add_preparser('[a-z][a-z]', parse_mgrs_square)

  def add_preparser(self, regex, function):
    matcher = re.compile(regex)
    self.preparsers.append([matcher, function])

  def check_preparsers(self, token):
    """Checks to see if any preparsers would like to handle the token.
    If not, None is returned, otherwise the result of preparsing the
    token is returned.
    """
    for [matcher, function] in self.preparsers:
      match = matcher.match(token)
      if match is None or match.end() != len(token):
        pass
      else:
        return function(token)
    return None

  def preparse(self, token):
    """Runs the token through any relevant preparser."""
    result = self.check_preparsers(token)
    if result is None:
      return token
    else:
      return result

  def reset(self):
    """Resets the parser state as if it hadn't yet parsed anything."""
    self.dynamic_predictions = {}
    self.position = 0
    self.references = []

  def add_phrasal_pattern(self, base, phrasal_pattern):
    """Adds a phrasal pattern to a class.  The phrasal_pattern
    argument is a string using the phrasal pattern syntax,
    e.g. '<action> the ?:[dang|darn] <object>' (see the
    PhrasalPatternParser class).
    """
    if not base in self.phrasal_patterns:
      self.phrasal_patterns[base] = [phrasal_pattern]
    else:
      self.phrasal_patterns[base].append(phrasal_pattern)
    pattern_parser = PhrasalPatternParser(stem=self.stem)
    pp_obj = pattern_parser.parse(phrasal_pattern)
    self.add_phrasal_pattern_object(base, pp_obj)

  def add_phrasal_pattern_object(self, base, phrasal_pattern_obj):
    if not base in self.phrasal_pattern_objects:
      self.phrasal_pattern_objects[base] = [phrasal_pattern_obj]
    else:
      self.phrasal_pattern_objects[base].append(phrasal_pattern_obj)
    for pred in self.generate_predictions(
      base, [phrasal_pattern_obj], None, None, {}, 0.0):
      self.index_anytime_prediction(pred)

  def index_prediction(self, table, prediction):
    target = self.prediction_target(prediction)
    if target in table:
      table[target].append(prediction)
    else:
      table[target] = [prediction]

  def index_anytime_prediction(self, prediction):
    """Adds a prediction to the set of anytime predictions."""
    if self.debug > 1:
      print 'Indexing anytime prediction %s' % (prediction,)
    self.index_prediction(self.anytime_predictions, prediction)

  def index_dynamic_prediction(self, prediction):
    """Adds a prediction to the set of dynamic predictions."""
    if self.debug > 1:
      print 'Indexing dynamic prediction %s' % (prediction,)
    self.index_prediction(self.dynamic_predictions, prediction)

  def predictions_on(self, item):
    preds = (predictions_on(self.dynamic_predictions, item) +
             predictions_on(self.anytime_predictions, item))
    if self.debug > 1:
      print 'Predictions on %s are %s' % (item, preds)
    return preds

  def clear_predictions(self):
    self.anytime_predictions = {}
    self.dynamic_predictions = {}

  def pparse(self, text, debug=0):
    """Parses a string and pretty-prints the results."""
    pprint.pprint(map(logic.Description.dictify,
                      self.parse(text, debug)))

  def parse_tokens(self, tokens, debug=0):
    """Parses a sequence of tokens. Returns the list of valid parses."""
    self.reset()
    self.debug = debug
    for position, token in enumerate(tokens):
      if self.stem:
        token = self.stemmer.stem(token)
      if not isinstance(token, basestring):
        raise TypeError(
          'Only string tokens are allowed; %s is not a string.' % (token,))
      self.reference(token, self.position, self.position, 0.0)
      preparse = self.check_preparsers(token)
      if preparse:
        self.reference(preparse, self.position, self.position, 0.0)
      self.position = position + 1
    return self.complete_parses(len(tokens))

  def complete_parses(self, pos):
    """Returns a list of complete parses given the current parser
    state.
    """
    parses = []
    for [item, start, end, unused_value] in self.references:
      if start == 0 and end == pos - 1 and isinstance(item, logic.Description):
        parses.append(item)
    return parses

  def reference(self, item, start, end, value):
    """References an item (a token string or a class)."""
    assert isinstance(item, basestring) or isinstance(item, logic.Description)
    if self.debug > 0:
      print 'referencing %s' % ((item, start, end),)
    self.references.append([item, start, end, value])
    for abst in self.all_abstractions(item):
      if self.reference_callback:
        apply(self.reference_callback, [abst, start, end, value])
      for prediction in self.predictions_on(abst):
        self.advance_prediction(prediction, item, start, end)

  def advance_prediction(self, prediction, item, start, end):
    """Advances a prediction."""
    if self.debug > 2:
      print 'Advancing prediction %s' % ((prediction, item, start, end),)
    if prediction.next is None or prediction.next == start:
      phrasal_pattern = prediction.phrasal_pattern[1:]
      if prediction.start is not None:
        start = prediction.start
      if is_head(prediction.phrasal_pattern[0]):
        base = item.base
        try:
          slots = self.merge_slots(prediction.slots, item.slots)
        except DuplicateSlotError:
          return
      else:
        base = prediction.base
        slots = self.extend_slots(prediction, item)
      if phrasal_pattern == []:
        # Prediction has been used up.
        self.reference(
          self.find_frame(base, slots), start, end, prediction.value)
      else:
        for prediction in self.generate_predictions(
          base, phrasal_pattern, start, self.position + 1, slots,
          prediction.value):
          if len(prediction.phrasal_pattern) > 0:
            self.index_dynamic_prediction(prediction)
          else:
            self.reference(
              self.find_frame(prediction.base, slots), start, end,
              prediction.value)

  def generate_predictions(self, base, phrasal_pattern, start, position, slots,
                           value):
    predictions = list(self.generate_predictions2(
        base, phrasal_pattern, start, position, slots, value))
    return predictions

  def generate_predictions2(self, base, phrasal_pattern, start, position,
                            slots, value):
    # If there's no syntax directive, it's an implicit sequence.  Make
    # explicit what's implicit.
    if not self.is_syntax_directive(phrasal_pattern[0]):
      phrasal_pattern = [[':sequence'] + phrasal_pattern]

    new_predictions = apply(self.syntax_functions[phrasal_pattern[0][0]],
                            [base, phrasal_pattern, start, position, slots])
    for pred in new_predictions:
      pred.value = pred.value + value
      if (len(pred.phrasal_pattern) > 0 and
          self.is_syntax_directive(pred.phrasal_pattern[0])):
        for p in self.generate_predictions2(base, pred.phrasal_pattern,
                                            start, position, slots, value):
          yield p
      else:
        yield pred

  def is_syntax_directive(self, term):
    """Checks whether a term in a phrasal pattern is a syntax
    directive, e.g. [':optional' ...].
    """
    if isinstance(term, list):
      if term[0] in self.syntax_functions:
        return True
      raise Error('%s is not a valid syntax function.' % (term[0],))
    else:
      return False

  def merge_slots(self, pred_slots, item_slots):
    """Merges two sets of slots into one superduper collection of slots."""
    for slot in pred_slots:
      if slot in item_slots:
        raise DuplicateSlotError('Slot %s already has the value %s.' % (
            slot, item_slots[slot]))
    slots = {}
    for slot in pred_slots:
      slots[slot] = pred_slots[slot]
    for slot in item_slots:
      slots[slot] = item_slots[slot]
    return slots

  def find_frame(self, base, slots):
    """Creates a description with the specified base class and slots."""
    return logic.Description(base, slots)

  def extend_slots(self, prediction, item):
    """If the prediction is waiting for a slot-filler, and the item we
    saw can fill the slot, add the slot with filler to the predictions
    slots.
    """
    spec = prediction.phrasal_pattern[0]
    slots = prediction.slots
    if is_role_specifier(spec):
      new_slots = copy(slots)
      new_slot = self.role_specifier(spec)
      if new_slot in new_slots:
        raise DuplicateSlotError('Slot %s already exists in %s.' % (
            new_slot, prediction))
      new_slots[new_slot] = item
      return new_slots
    else:
      return slots

  def prediction_target(self, prediction):
    spec = prediction.phrasal_pattern[0]
    if self.is_syntax_directive(spec):
      raise Error('Cannot index on syntax directive %s.' % (spec,))
    if is_role_specifier(spec):
      base = prediction.base
      value = self.slot_constraint(base, self.role_specifier(spec))
      if value is not None:
        return value
      else:
        raise Error('%s has no constraint in %s.' % (spec, base))
    elif is_head(spec):
      return prediction.base
    else:
      return spec

  def all_abstractions(self, item):
    if isinstance(item, basestring):
      return [item]
    elif isinstance(item, logic.Expr):
      return self.kb.all_parents(item)
    elif isinstance(item, logic.Description):
      return self.kb.all_parents(logic.expr(item.base))
    else:
      raise Error('%s must be a string or Expr.' % (repr(item,)))

  def slot_constraint(self, item, role_spec):
    """Looks up the constraint on the specified slot for item."""
    return self.kb.slot_value(
      logic.expr(item),
      CONSTRAINT_EXPR,
      logic.expr(role_spec))

  def role_specifier(self, item):
    return logic.expr(item[1:])


class Prediction:
  """Represents a prediction the parser has made about what the next
  token might be and what frame it is part of.
  """
  def __init__(self, base, phrasal_pattern, start, next_pos, slots, value):
    self.base = base
    self.phrasal_pattern = phrasal_pattern
    self.start = start
    self.next = next_pos
    self.slots = slots
    self.value = value

  def __repr__(self):
    return '<%s base: %s start: %s next: %s slots: %s pat: %s value: %s>' % (
      self.__class__.__name__, repr(self.base), self.start, self.next,
      repr(self.slots), self.phrasal_pattern, self.value)

  # We make phrasal_pattern a somewhat fancy attribute of this class;
  # When it's set to a sequence, we automatically tokenize the first
  # element.
  #
  # pred.phrasal_pattern = ['how are you', 'john?']
  #
  # pred.phrasal_pattern -> ['how', 'are', 'you', 'john?']
  #
  # FIXME: This doesn't feel like the best place to do this.
  def __setattr__(self, name, value):
    if name == 'phrasal_pattern':
      if (len(value) > 0 and isinstance(value[0], basestring) and
          value[0][0] != ':' and value[0][0] != '?'):
        tokens = self.tokenize(value[0])
        self.__dict__[name] = tokens + value[1:]
      else:
        self.__dict__[name] = value
    else:
      self.__dict__[name] = value

  def __getattr__(self, name):
    if name == 'phrasal_pattern':
      return self._phrasal_pattern
    else:
      raise AttributeError(name)

  def tokenize(self, text):
    return tokenize(text)


def predictions_on(prediction_table, item):
  if item in prediction_table:
    predictions = prediction_table[item]
  else:
    predictions = []
  return predictions


# These *_prediction_generator functions are used for syntax directive
# processing.  They return a list of new predictions.

def head_prediction_generator(base, phrasal_pattern, start, position, slots):
  """Generates predictions for :head."""
  return [Prediction(
      base, [':head'] + phrasal_pattern[1:], start, position, slots)]


def sequence_prediction_generator(base, phrasal_pattern, start, position,
                                  slots):
  """Generates predictions for :sequence."""
  return [Prediction(base, phrasal_pattern[0][1:] + phrasal_pattern[1:],
                     start, position, slots, 1.0)]


def optional_prediction_generator(base, phrasal_pattern, start, position,
                                  slots):
  """Generates predictions for :optional."""
  return [Prediction(base, phrasal_pattern[1:], start, position, slots, 0.2),
          Prediction(base, phrasal_pattern[0][1:] + phrasal_pattern[1:],
                     start, position, slots, 0.0)]


def any_prediction_generator(base, phrasal_pattern, start, position, slots):
  """Generates predictions for :any."""
  preds = map(lambda pat: Prediction(base, [pat] + phrasal_pattern[1:],
                                     start, position, slots, 0.0),
              phrasal_pattern[0][1:])
  return preds


class DuplicateSlotError(Error):
  pass


def is_role_specifier(item):
  return item[0] == '?'


def is_head(item):
  return item == ':head'


class PhrasalPatternParser:
  """Parses phrasal patterns.

  <color> is a reference to a slot named color.
  <:head> is special, and refers to the phrase head.
  ?:thing means that thing is optional.
  [thing-a|thing-b] means that either thing-a or thing-b are acceptable.

  Examples:

  'pick up ?:the <object>'
  '?:<name>, ?:[please|would you] clean my <object>'
  """

  def __init__(self, stem=False):
    # We do the stemming of literal tokens in this class.  Is that
    # weird?
    if stem:
      self.stemmer = stemmer.PorterStemmer()
    else:
      self.stemmer = False

  # We do this thing where we parse into an intermediate
  # representation (as returned by parse_tree), then convert that into
  # the final form.  I don't remember how that came about and it
  # should perhaps be examined in the future.
  #
  # Pattern string:
  #   '?:[let's|let us] <action>'
  #
  # Intermediate representation:
  #   [':sequence',
  #    [':optional',
  #     [':any',
  #      [':symbol', 'let's'],
  #      [':sequence', [':symbol', 'let'], [':symbol', 'us']]]],
  #    [':slotref', [':symbol', 'action']]]
  #
  # Final form:
  #   [':sequence',
  #    [':optional',
  #     [':any', [':sequence', 'let', 's'], [':sequence', 'let', 'us']]],
  #    '?action']

  def parse(self, pattern):
    """Parses a string containing a phrasal pattern into a tree
    representation.
    """
    phrasal_pattern = self.convert_parse_tree_to_phrasal_pattern(
      self.parse_tree(pattern))
    return phrasal_pattern

  def parse_tree(self, input_str):
    [obj, unused_position] = self.read_sequence(input_str, 0)
    return obj

  def read(self, input_str, position):
    position = self.skip_whitespace(input_str, position)
    if position >= len(input_str):
      return [None, position]

    char = input_str[position]
    if char == '<':
      return self.read_slot(input_str, position + 1)
    elif char == '{':
      return self.read_slot(input_str, position + 1, '{', '}')
    elif char == '?' and input_str[position + 1] == ':':
      return self.read_optional(input_str, position + 2)
    elif char == '[':
      return self.read_choice(input_str, position + 1)
    elif self.is_symbol_char(char):
      return self.read_token(input_str, position)
    else:
      raise Error(
        'Illegal character %r at position %s in %r.' % (
          char, position, repr(input_str)))

  def read_sequence(self, input_str, position, terminators=''):
    objects = []
    [obj, position] = self.read(input_str, position)
    if obj is not None:
      objects.append(obj)
    while (obj is not None and
           (position >= len(input_str) or
            not input_str[position] in terminators)):
      [obj, position] = self.read(input_str, position)
      if obj is not None:
        objects.append(obj)
    return [self.make_sequence(objects), position]

  def read_slot(self, input_str, position, slot_char='<', terminator='>'):
    [symbol, position] = self.read_symbol(input_str, position)
    position = self.skip_whitespace(input_str, position)
    if not position < len(input_str):
      raise Error(
            'Unterminated %r in phrasal pattern %r.' % (
            slot_char, input_str))
    if input_str[position] != terminator:
      raise Error(
            ('Unexpected character %r in slot reference in phrasal '
             'pattern %r') % (
            input_str[position], input_str))
    return [self.make_slot(symbol), position + 1]

  def read_optional(self, input_str, position):
    [obj, position] = self.read(input_str, position)
    return [self.make_optional(obj), position]

  def read_choice(self, input_str, position):
    choices = []
    while input_str[position] != ']':
      [obj, position] = self.read_sequence(input_str, position, '|]')
      position = self.skip_whitespace(input_str, position)
      if position >= len(input_str):
        raise Error("Unterminated '[' in %r." % (input_str,))
      if not (input_str[position] == ']' or input_str[position] == '|'):
        raise Error('Illegal character %r in %r.' % (
            input_str[position], input_str))
      if input_str[position] == '|':
        position = position + 1
      choices.append(obj)
    return [self.make_choice(choices), position + 1]

  def read_symbol(self, input_str, position):
    position = self.skip_whitespace(input_str, position)
    start_position = position
    while (position < len(input_str) and
           self.is_symbol_char(input_str[position])):
      position = position + 1
    return [self.make_symbol(input_str[start_position:position]), position]

  def read_token(self, input_str, position):
    position = self.skip_whitespace(input_str, position)
    start_position = position
    while (position < len(input_str) and
           self.is_symbol_char(input_str[position])):
      position = position + 1
      symbol = self.make_symbol(
        self.maybe_stem(input_str[start_position:position]))
    return symbol, position

  def make_symbol(self, s):
    return [':symbol', s]

  def make_sequence(self, objects):
    if len(objects) == 1:
      return objects[0]
    else:
      return [':sequence'] + objects

  def make_optional(self, obj):
    return [':optional', obj]

  def make_choice(self, objects):
    if len(objects) == 1:
      return objects[0]
    else:
      return [':any'] + objects

  def make_slot(self, symbol):
    return [':slotref', symbol]

  def skip_whitespace(self, input_str, position):
    while (position < len(input_str) and
           (input_str[position] == ' ' or input_str[position] == '\n')):
      position = position + 1
    return position

  def is_symbol_char(self, char):
    return char in string.digits or char in string.letters or char in "-'?:"

  def convert_parse_tree_to_phrasal_pattern(self, tree):
    node_type = tree[0]
    if node_type == ':sequence':
      return [':sequence'] + map(
        self.convert_parse_tree_to_phrasal_pattern, tree[1:])
    elif node_type == ':symbol':
      if tree[1][0] == '?' and tree[1][1] in string.letters:
        return tree[1]
      else:
        symbols = tokenize(tree[1])
        if len(symbols) == 1:
          return symbols[0]
        else:
          return [':sequence'] + symbols
    elif node_type == ':slotref':
      symbol_str = tree[1][1]
      if symbol_str == ':head':
        return ':head'
      else:
        return '?' + symbol_str
    elif node_type == ':optional':
      return [':optional'] + map(
        self.convert_parse_tree_to_phrasal_pattern, tree[1:])
    elif node_type == ':any':
      return [':any'] + map(
        self.convert_parse_tree_to_phrasal_pattern, tree[1:])
    else:
      raise Error('Unknown element %s. (%s)' % (node_type, tree))

  def maybe_stem(self, token):
    if self.stemmer:
      return self.stemmer.stem(token)
    else:
      return token


class FrameHandler(fdl.BaseFrameHandler):
  def __init__(self, kb, cp, icp):
    fdl.BaseFrameHandler.__init__(self, kb, cp, icp)
    self.constraints = {}

  def handle_constraints(self, frame, constraints):
    fdl.BaseFrameHandler.handle_constraints(self, frame, constraints)
    if len(constraints) > 0:
      self.constraints[frame['class_name']] = constraints


class InteractiveParserApp:
  """Lets you interactively play with a ConceptualParser."""
  def __init__(self, argv):
    self.kb = None
    self.cp_parser = None
    self.icp_parser = None
    self.fdl_handler = None
    self.fdl_parser = None
    self.debug = 0
    self.run_tests = False
    self.transcript_path = None
    self.test_classes = []
    optlist, args = getopt.getopt(argv[1:], 'td:f:c:')
    for o, v in optlist:
      if o == '-d':
        self.debug = v
      elif o == '-t':
        self.run_tests = True
      elif o == '-c':
        self.test_classes = v.split(',')
      elif o == '-f':
        self.transcript_path = v
    self.fdl_file = args[0]

  def run(self):
    self.kb = logic.PropKB()
    self.cp_parser = ConceptualParser(self.kb)
    self.icp_parser = IndexedConceptParser(self.kb)
    self.fdl_handler = FrameHandler(self.kb, self.cp_parser, self.icp_parser)
    self.fdl_parser = fdl.FDLParser(self.fdl_handler)
    self.fdl_parser.parse_fdl_file(self.fdl_file, self.debug)

    if self.run_tests:
      self.check_constraints()
      self.fdl_parser.run_test_phrases(
        self.test_classes, self.cp_parser, self.icp_parser)

    if self.transcript_path:
      for line in open(self.transcript_path):
        line = line[0:-1]
        if len(line) > 0:
          print '\n*** %s' % (line,)
          parses = self.cp_parser.parse(line)
          print '  %s:' % (len(parses),)
          pprint.pprint(parses)

    if not self.run_tests and not self.transcript_path:
      self.do_parse_loop()

  def check_constraints(self):
    def can_be_constraint(concept):
      for c in self.kb.all_children(logic.expr(concept)):
        if c in self.cp_parser.phrasal_patterns:
          return True
      return False

    for class_name in self.fdl_handler.constraints:
      constraints = self.fdl_handler.constraints[class_name]
      for name in constraints:
        req_type = constraints[name]
        if not can_be_constraint(req_type):
          logging.warning(
            "%s has constraint '%s IS-A %s' which has no phrasal patterns",
            class_name, name, req_type)
        sv = self.kb.slot_value(logic.expr(class_name), logic.expr(name))
        if sv:
          if not self.kb.isa(sv, logic.expr(req_type)):
            logging.warning(
              ("%s has constraint '%s IS-A %s' which is not consistent "
               'with slot value %s'),
              class_name, name, req_type, sv)

  def do_parse_loop(self):
    while True:
      sys.stdout.write('? ')
      input_str = sys.stdin.readline()
      if len(input_str) == 0:
        break
      if input_str[0] == '#':
        print 'CP: ' + self.cp_parser.predictions_on(eval(input_str[1:]))
        #print 'ICP: ' + self.icp_parser.predictions_on(eval(input_str[1:]))
      elif input_str[0] == '%':
        print 'CP: ' + self.cp_parser.anytime_predictions
        #print 'ICP: ' + self.icp_parser.anytime_predictions
      else:
        cp_results = self.cp_parser.parse(input_str, debug=self.debug)
        if cp_results:
          print 'CP:'
          pprint.pprint(map(lambda d: d.dictify(), cp_results))
        icp_results = self.icp_parser.parse(input_str, debug=self.debug)
        if icp_results:
          print 'ICP:'
          pprint.pprint(map(lambda d: d.dictify(), icp_results))

# --------------------------------------------------
# Indexed Concept Parser
# --------------------------------------------------

# Only parse results with a score greater than this will be returned.
CUTOFF_ICP_SCORE = -1000000

MIN_PROBABILITY = -100.0 / CUTOFF_ICP_SCORE

MIN_INFORMATION_VALUE = -100.0 / CUTOFF_ICP_SCORE


class IndexedConceptParser(ParserBase):
  """A Will Fitzgerald-style Indexed Concept Parser."""
  def __init__(self, kb):
    ParserBase.__init__(self)
    self.debug = 0
    self.kb = kb
    self.cp_parser = ConceptualParser(kb, stem=True)
    self.index_sets = {}
    self.target_concepts = {}
    self.appraisers = []
    self.total_appraiser_votes = 0
    self.stemmer = stemmer.PorterStemmer()
    self.index_set_pattern_parser = IndexSetPatternParser(kb)
    self.install_standard_appraisers()
    self.unique_target_concepts = {}

  def add_appraiser(self, appraiser, votes):
    self.appraisers.append([appraiser, votes])
    self.total_appraiser_votes = self.total_appraiser_votes + votes

  def install_standard_appraisers(self):
    self.add_appraiser(PredictedScore(self), 1)
    self.add_appraiser(UnpredictedScore(self), 1)
    self.add_appraiser(UnseenScore(self), 1)
    self.add_appraiser(RequiredScore(self), 10)

  def stem(self, token):
    if isinstance(token, basestring):
      return self.stemmer.stem(token)
    else:
      return token

  def parse_tokens(self, tokens, debug=0):
    self.debug = debug
    self.cp_parser.debug = debug
    indices = self.find_indices(tokens, self.match_function)
    if debug > 0:
      print 'ICP parsing tokens %s' % (tokens,)
    if debug > 1:
      print 'ICP found indices %s' % (indices,)
    results = self.score_index_sets(indices)
    results.sort(key=lambda x: x.score, reverse=True)
    results = [result for result in results if result.score > CUTOFF_ICP_SCORE]
    results = utils.remove_duplicates(
      results,
      lambda r1, r2: r1.target_concept == r2.target_concept)
    if debug > 0:
      print 'ICP results: %s' % (results,)
    return results

  def find_indices(self, tokens, match_fn):
    return apply(match_fn, [tokens])

  def match_function(self, tokens):
    """Uses the CP to parse tokens and returns a list of the tokens
    and any concepts that were referenced.
    """
    items = []

    def add_ref(item, unused_start, unused_end, unused_value):
      if isinstance(item, logic.Description):
        items.append(logic.expr(item))
      else:
        items.append(item)

    self.cp_parser.reference_callback = add_ref
    self.cp_parser.parse_tokens(tokens, debug=self.debug)
    return items

  def score_index_sets(self, found_indices):
    results = []
    for index_set in self.candidate_index_sets(found_indices):
      result = ICPResult(None,
                         self.index_set_score(index_set, found_indices),
                         index_set.target_concept,
                         index_set.indices,
                         self.extract_slot_fillers(index_set, found_indices))
      results.append(result)
    return results

  def extract_slot_fillers(self, index_set, found_indices):
    # found_indices may be something like ['big', c-big, c-size], in
    # which case we want the most specific concept (c-big) to fill our
    # slot.
    def maybe_use_filler(indices, current_filler, candidate_filler):
      if current_filler is None:
        return candidate_filler
      elif self.kb.isa(logic.expr(indices[candidate_filler]),
                       logic.expr(indices[current_filler])):
        return candidate_filler
      else:
        return current_filler

    result_slots = {}
    for [slot, constraint] in index_set.slots:
      filler = None
      for i, index in enumerate(found_indices):
        if (isinstance(index, logic.Expr) and
            self.kb.isa(index, constraint) and
            not (i in result_slots.values())):
          filler = maybe_use_filler(found_indices, filler, i)
      if filler is not None:
        result_slots[slot] = filler
    for k, unused_v in result_slots.items():
      result_slots[k] = found_indices[result_slots[k]]
    return result_slots

  def candidate_index_sets(self, found_indices):
    candidates = []
    abstractions = []
    for index in found_indices:
      abstractions = abstractions + self.all_abstractions(index)
    for index in abstractions:
      candidates = candidates + self.index_sets.get(index, [])
#    print 'candidates for %s are %s' % (found_indices, candidates)
    return utils.remove_duplicates(candidates)

  def all_abstractions(self, item):
    if isinstance(item, basestring):
      return [item]
    elif isinstance(item, logic.Expr):
      return self.kb.all_parents(item)
    elif isinstance(item, logic.Description):
      return self.kb.all_parents(logic.expr(item.base))
    else:
      raise Error('%s must be a string or Expr.' % (repr(item,)))

  def install(self, index_set):
    """Installs an index set."""
    index_set.indices = map(self.stem, index_set.indices)
    index_set.required_indices = map(self.stem, index_set.required_indices)
    self.unique_target_concepts[index_set.target_concept] = True
    for index in index_set.indices:
      if not index in self.target_concepts.get(index, []):
        self.target_concepts[index] = ([index_set.target_concept] +
                                       self.target_concepts.get(index, []))
      if not index_set in self.index_sets.get(index, []):
        self.index_sets[index] = [index_set] + self.index_sets.get(index, [])

  def add_phrasal_pattern(self, base, phrasal_pattern):
    # We keep track of indexsets while we let the CP keep track of
    # phrasal patterns.
    self.cp_parser.add_phrasal_pattern(base, phrasal_pattern)

  def add_index_set(self, target_concept, indexsetpattern):
    """Adds an index set to the target concept.  The indexsetpattern
    must be a string containing an indexset pattern (see
    IndexSetPatternParser).
    """
    indexset = self.index_set_pattern_parser.parse(
      logic.expr(target_concept), indexsetpattern)
    self.install(indexset)

  def index_set_score(self, index_set, found_indices):
    score = 0
    for (appraiser, votes) in self.appraisers:
      if votes > 0:
        appraiser_score = self.call_appraiser(
          appraiser,
          votes / float(self.total_appraiser_votes),
          index_set,
          found_indices)
        score = score + appraiser_score
    return score

  def call_appraiser(self, appraiser, weight, index_set, found_indices):
    score = weight * appraiser.score(index_set, found_indices)
    return score

  def probability_of_index(self, index):
    cardinality = self.target_concept_cardinality(index)
    if cardinality == 0:
      # Very small, but > 0
      return MIN_PROBABILITY
    else:
      return float(cardinality) / len(self.unique_target_concepts)

  def target_concept_cardinality(self, index):
    return len(self.target_concepts.get(index, []))

  def summed_value(self, unused_base, predicted_set):
    if_sum = 0.0
    for item in predicted_set:
      if_sum += self.information_value(item)
    return if_sum

  def information_value(self, index):
    value = -math.log(self.probability_of_index(index), 2)
    if value == 0.0:
      value = MIN_INFORMATION_VALUE
    return value


class IndexSet:
  """Represents a set of indices for the IndexedConceptParser.
  Includes the target concept, the indices, and required indices.
  """
  def __init__(self, target=None, indices=None, required_indices=None,
               slots=None):
    def cond(test, a, b):
      if test:
        return a
      else:
        return b

    if isinstance(target, basestring):
      self.target_concept = logic.expr(target)
    else:
      self.target_concept = target
    self.indices = cond(indices is None, [], indices)
    self.required_indices = cond(
      required_indices is None, [], required_indices)
    self.slots = cond(slots is None, [], slots)

  def __repr__(self):
    s = StringIO.StringIO()
    s.write('<%s target: %s indices: %s' % (
      self.__class__.__name__, self.target_concept, self.indices))
    if len(self.required_indices) > 0:
      s.write(' required: %s' % (self.required_indices,))
    if len(self.slots) > 0:
      s.write(' slots: %s' % (self.slots,))
    s.write('>')
    return s.getvalue()

  def __cmp__(self, other):
    if (other is self) or (isinstance(other, IndexSet) and
                           self.target_concept == other.target_concept and
                           self.indices == other.indices and
                           self.required_indices == other.required_indices):
      return 0
    else:
      return -1


class ICPResult(logic.Description):
  """Holds an Indexed Concept Parser parse result, which consists of
  the target concept, the score, the index concepts and the input
  text.
  """
  def __init__(self, text, score, target_concept, index_concepts, slots):
    logic.Description.__init__(self, target_concept, slots)
    self.text = text
    self.score = score
    self.index_concepts = index_concepts
    self.target_concept = target_concept

  def __repr__(self):
    return '<%s score: %s target: %s slots: %s>' % \
           (self.__class__.__name__, self.score, self.target_concept,
            self.slots)


# --------------------
# ICP Appraisers
# --------------------

class PredictedScore:
  """ICP appraiser that scores up for indices that we've seen that are
  in the indexset.
  """
  def __init__(self, parser):
    self.parser = parser

  def score(self, index_set, found_indices):
    predicted = index_set.indices
    pred_items = predicted_items(self.parser.kb, found_indices, predicted)
    score = (self.parser.summed_value(index_set.target_concept, pred_items) /
             self.parser.summed_value(index_set.target_concept, predicted))
    return score


class UnpredictedScore:
  """ICP appraiser that penalizes for indices that we've seen that
  were not part of the indexset.
  """
  def __init__(self, parser):
    self.parser = parser

  def score(self, index_set, found_indices):
    predicted = index_set.indices
    unpred_items = unpredicted_items(self.parser.kb, found_indices, predicted)
    unpredicted_score = self.parser.summed_value(
      index_set.target_concept, unpred_items)
    seen_score = self.parser.summed_value(
      index_set.target_concept, found_indices)
    score = 1.0 - (unpredicted_score / seen_score)
    return score


class UnseenScore:
  """ICP appraiser that penalizes for indices we wanted to see but
  didn't.
  """
  def __init__(self, parser):
    self.parser = parser

  def score(self, index_set, found_indices):
    predicted = index_set.indices
    unseed_items = unseen_items(self.parser.kb, found_indices, predicted)
    unseed_score = self.parser.summed_value(
      index_set.target_concept, unseed_items)
    seed_score = self.parser.summed_value(
      index_set.target_concept, found_indices)
    score = 1.0 - (unseed_score / seed_score)
    return score


class RequiredScore:
  """ICP appraiser that nukes your score if there are required indices
  that were not seen.
  """
  def __init__(self, parser):
    self.parser = parser

  def score(self, index_set, found_indices):
    # Make a copy.
    found_indices = found_indices[:]
    for requirement in index_set.required_indices:
      if not requirement in found_indices:
        # Return something nice and low.
        return CUTOFF_ICP_SCORE * 10
      else:
        # Don't want to use a single index to satisfy multiple
        # requirements.
        del found_indices[found_indices.index(requirement)]
    return 0.0


# --------------------
# ICP utilities
# --------------------

def is_concept(thing):
  return isinstance(thing, logic.Description) or isinstance(thing, logic.Expr)


def abst_or_whole_of(kb, big, small):
  if is_concept(big) and is_concept(small):
    return kb.isa(big, small)
  else:
    return big == small


def spec_or_part_of(kb, big, small):
  if is_concept(big) and is_concept(small):
    return kb.isa(small, big)
  else:
    return big == small


def predicted_items(kb, seen_set, predicted_set):
  return utils.intersection(predicted_set,
                            seen_set,
                            lambda e1, e2: abst_or_whole_of(kb, e1, e2))


def unpredicted_items(kb, seen_set, predicted_set):
  return utils.set_difference(seen_set,
                              predicted_set,
                              lambda e1, e2: spec_or_part_of(kb, e1, e2))


def unseen_items(kb, seen_set, predicted_set):
  return utils.set_difference(predicted_set,
                              seen_set,
                              lambda e1, e2: spec_or_part_of(kb, e1, e2))


class IndexSetPatternParser:
  """Parses indexset patterns.

  word     is a literal token.
  $concept is a concept reference.
  !thing   is a required element.
  {slot}   is a slot reference.

  Examples:

  'big $c-dog'
  '!big !$c-dog'
  '{size} !{color}'
  """
  def __init__(self, kb):
    self.kb = kb

  def parse(self, target, pattern):
    """Parses a string containing a indexset pattern and returns an
    IndexSet.
    """
    indexset = IndexSet(target)
    return self.read(indexset, pattern, 0)

  def read(self, indexset, input_str, position):
    [unused_index, position] = self.read_one(indexset, input_str, position)
    while position < len(input_str):
      [unused_index, position] = self.read_one(indexset, input_str, position)
    return indexset

  def read_one(self, indexset, input_str, position):
    position = self.skip_whitespace(input_str, position)
    if position >= len(input_str):
      return [None, position]
    char = input_str[position]
    if char == '$':
      return self.parse_concept(indexset, input_str, position)
    elif char == '!':
      return self.parse_required(indexset, input_str, position)
    elif char == '{':
      return self.parse_slot(indexset, input_str, position)
    elif self.is_symbol_char(char):
      return self.parse_token(indexset, input_str, position)
    else:
      raise Error(
        'Illegal character %r at position %s in indexset %r.' % (
          (char, position, input_str)))

  def parse_token(self, indexset, input_str, position):
    # -- Token
    [index, position] = self.read_symbol(input_str, position)
    indexset.indices = indexset.indices + [index]
    return [index, position]

  def parse_concept(self, indexset, input_str, position):
    [index, position] = self.read_concept(input_str, position + 1)
    if index is None:
      raise Error('Lone ! in indexset %s.' % (repr(input_str),))
    indexset.indices = indexset.indices + [index]
    return [index, position]

  def parse_required(self, indexset, input_str, position):
    # -- Required
    [index, position] = self.read_one(indexset, input_str, position + 1)
    if index is None:
      raise Error('Lone ! in indexset %s.' % (repr(input_str),))
    indexset.required_indices = indexset.required_indices + [index]
    return [index, position]

  def parse_slot(self, indexset, input_str, position):
    # -- Slot reference
    [slot_name, position] = self.read_slot(input_str, position + 1)
    if slot_name is None:
      raise Error('Empty slot reference in indexset %s.' % (repr(input_str),))
    if slot_name in [slot_ref[0] for slot_ref in indexset.slots]:
      raise Error(
        'Duplicate slot reference %s in indexset %s.' % (
          slot_name, repr(input_str)))
    slot_type = self.slot_constraint(indexset.target_concept, slot_name)
    indexset.slots.append([slot_name, slot_type])
    indexset.indices = indexset.indices + [slot_type]
    return [slot_type, position]

  def read_symbol(self, input_str, position):
    position = self.skip_whitespace(input_str, position)
    start_position = position
    while (position < len(input_str) and
           self.is_symbol_char(input_str[position])):
      position += 1
    return [input_str[start_position:position], position]

  def read_concept(self, input_str, position):
    [symbol, position] = self.read_symbol(input_str, position)
    return [logic.expr(symbol), position]

  def read_slot(self, input_str, position):
    [symbol, position] = self.read_symbol(input_str, position)
    position = self.skip_whitespace(input_str, position)
    if not position < len(input_str):
      raise Error("Unterminated '{' in indexset %r" % (input_str,))
    if input_str[position] != '}':
      raise Error(
        "Unexpected character '%s' in slot reference in indexset %r." % (
          input_str[position], input_str))
    return [symbol, position + 1]

  def skip_whitespace(self, input_str, position):
    while (position < len(input_str) and
           (input_str[position] == ' ' or input_str[position] == '\n')):
      position += 1
    return position

  def is_symbol_char(self, char):
    return char in string.digits or char in string.letters or char in "-'?:"

  def slot_constraint(self, item, slot):
    item = logic.expr(item)
    slot = logic.expr(slot)
    return self.kb.slot_value(item, CONSTRAINT_EXPR, slot)


def main():
  p = InteractiveParserApp(sys.argv)
  p.run()


if __name__ == '__main__':
  main()
