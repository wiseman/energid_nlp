# Copyright Energid Technologies 2012

# Based on code from Artificial Intelligence: A Modern Approach
# http://aima.cs.berkeley.edu/

"""Implements propositional databases and logical expressions."""

import re
import pprint

from energid_nlp import discriminationtree
from energid_nlp import utils


class Error(Exception):
  pass


class KB:
  """A mostly abstract interface to knowledge bases."""
  def __init__(self, sentence=None):
    pass

  def tell(self, sentence):
    """Adds the logical sentence to the KB."""
    raise NotImplementedError

  def ask(self, query):
    """Returns a substitution that makes the query true, or it returns
    False. It is implemented in terms of ask_generator.
    """
    try:
      answer = self.ask_generator(query).next()
      return answer
    except StopIteration:
      return False

  def ask_all(self, query):
    """Returns a list of all substitutions that make the query
    true.
    """
    answer = list(self.ask_generator(query))
    return answer

  def ask_generator(self, query, bindings=None):
    """Yields all the substitutions that make query true."""
    raise NotImplementedError

  def retract(self, sentence):
    """Remove the sentence from the KB."""
    raise NotImplementedError


class Proposition:
  # A logical proposition.  Stored in PropKBs.  Users never see
  # these.

  def __init__(self, form):
    # Creates a Proposition from a form (a sequence of Exprs).
    self.form = form

  def __str__(self):
    return '<Proposition %s>' % (self.form,)

  def __repr__(self):
    return '<Proposition %s>' % (self.form,)

  def clause(self):
    # Returns the Expr the Proposition represents.
    return apply(Expr, [self.form[0]] + self.form[1:])


def reuse_proposition(old_prop, form):
  old_prop.form = form


class PropKB(KB):
  """A KB for propositional logic.  Uses a discrimination tree to
  efficiently index the propositions.
  """
  def __init__(self):
    KB.__init__(self)
    self.tree = discriminationtree.make_root_discrimination_tree(is_variable)
    self.functions = {}   # Memory functions
    self.fluents = []     # Fluent relations
    self.heritable = []   # Heritable (default) relations
    self.install_standard_functions()

  def tell(self, sentence):
    """Add the sentence's clauses to the KB."""
    if not isinstance(sentence, Expr):
      raise Error('%s must be an Expr.' % (repr(sentence),))
    else:
      clauses = conjuncts(to_cnf(sentence))
      for clause in clauses:
        if clause.op in self.functions:
          self.tell_function(clause)
        else:
          self.add_proposition(clause)

  def tell_function(self, sentence):
    # Handles assertions that involve a memory function.
    function = self.functions[sentence.op]
    function.tell(self, sentence)

  def assert_isa(self, child, parent):
    """Adds an inheritance relationship between child and parent."""
    if not isinstance(child, Expr):
      raise Error('%s must be an Expr.' % (child,))
    if not isinstance(parent, Expr):
      raise Error('%s must be an Expr.' % (parent,))
    # Use $ISA internally to record explicit ISA links.
    self.invalidate_isa_cache()
    self.tell(expr('$ISA(%s, %s)' % (child.op, parent.op)))

  def assert_instanceof(self, child, parent):
    """Adds an inheritance relationship between child and parent, and
    marks child as an instance.
    """
    if not isinstance(child, Expr):
      raise Error('%s must be an Expr.' % (child,))
    if not isinstance(parent, Expr):
      raise Error('%s must be an Expr.' % (parent,))
    # First assert the ISA-ness.
    self.assert_isa(child, parent)
    # Then use $INSTANCE internally to record explicit
    # instance-ness.
    self.tell(expr('$INSTANCE(%s)' % (child.op)))

  def ask_generator(self, query, bindings=None):
    if bindings is None:
      bindings = {}
    # Returns a generator for all the substitutions that make the
    # query true.
    #
    # AND (&) and OR (|) could be implemented as memory functions just
    # like NOT.
    if query.op == '&':
      binding_generator = self.ask_and(query, bindings)
    elif query.op == '|':
      binding_generator = self.ask_or(query, bindings)
    elif query.op in self.functions:
      binding_generator = self.ask_function(query, bindings)
    else:
      # It's not & or | or a memory function.
      binding_generator = self.ask_simple(query, bindings)
    return binding_generator

  def retract(self, clause):
    """Removes the sentence's clauses from the KB."""
    if clause.op in self.functions:
      self.retract_function(clause)
    else:
      self.retract_propositions_with_form(clause.form())

  def retract_function(self, query):
    # Calls a memory function's retract method.
    function = self.functions[query.op]
    function.retract(self, query)

  def ask_simple(self, query, active_bindings):
    # Handles simple queries of single relations, e.g. Color(Thingy,
    # Value).  Can handle variables, and inheritance.
    if query.op in self.heritable:
      for bindings in self.ask_with_inheritance(query, active_bindings):
        yield bindings
    else:
      for bindings in self.unify_with_propositions(query, active_bindings):
        yield bindings

  def unify_with_propositions(self, query, active_bindings):
    # Unifies the specified query with all matching propositions
    # in the discrimination tree.  Yields new substitutions that
    # extend the active_bindings.
    for bindings in self.retrieve_one_form(query, active_bindings):
      yield bindings

  def retrieve_one_form(self, query, active_bindings):
    # Returns a generator for all binding solutions that let query
    # unify with the propositions in the discrimination tree.
    query = subst(active_bindings, query)
    is_fluent = query.op in self.fluents
    is_heritable = query.op in self.heritable

    (inner_relations, obj, value) = deconstruct_query(query)
    if len(inner_relations) == 1 and isinstance(obj.op, Description):
      # Querying a slot value on a description.
      return self.retrieve_description_form(inner_relations[0], obj,
                                            value, active_bindings)
    generator = self.retrieve_bindings_generator(query, active_bindings)
    if generator is None:
      # Didn't find anything specific to this object, check for
      # inherited values.
      generator2 = self.check_for_inherited_solutions(query, active_bindings)
      if generator2 is not None:
        generator = generator2
    if generator is not None:
      # Check for more defaults from non-fluent defaults.
      if (not is_fluent) and is_heritable:
        generator3 = self.check_for_inherited_solutions(query, active_bindings)
        if generator3 is not None:
          pipe = append_generators(generator, generator3)
    return generator

#    def extend_heritables_to_children(self, query, generator):
#        (inner_relations, object, value) = self.deconstruct_query(query)
#        found_bindings = False

  def match_against_one_proposition(self, query, proposition, active_bindings):
    return unify(query, proposition.clause(), active_bindings)

  def retrieve_bindings_generator(self, query, active_bindings):
    possible_matches = self.retrieve(query.form())
    if possible_matches is not None:
      return self.filter_possible_matches(self.match_against_one_proposition,
                                          possible_matches,
                                          query,
                                          active_bindings)
    else:
      return None

  def filter_possible_matches(self, match_fn, possible_matches, query,
                              active_bindings):
    # Takes a generator of possible matches and returns a new
    # generator containing actual matches (actually yields binding
    # sets).
    if possible_matches is None:
      return
    else:
      next_match = possible_matches.next()
      new_bindings = match_fn(query, next_match, active_bindings)
      if new_bindings is not None:
        yield new_bindings
        for bindings in self.filter_possible_matches(match_fn,
                                                     possible_matches,
                                                     query,
                                                     active_bindings):
          yield bindings
      else:
        for bindings in self.filter_possible_matches(match_fn,
                                                     possible_matches,
                                                     query,
                                                     active_bindings):
          yield bindings

  def retrieve_description_form(self, relation, obj, value,
                                active_bindings):
    # Handles queries on Descriptions.  We first check to see if the
    # Description has a slot satisfying the query, and if it does not
    # then we do the same query on the base class of the description.
    def generator(bindings):
      yield bindings

    description = obj.op
    if is_variable(value):
      variable = value
      if description.has_slot(relation):
        return generator(extend(active_bindings, variable,
                                expr(description.slot_value(relation))))
      else:
        return self.retrieve_one_form(expr(relation)(description.base, value),
                                      active_bindings)
    else:
      if description.has_slot(relation):
        if description.slot_value(relation) == value.op:
          return generator(active_bindings)
        else:
          return None
      else:
        return self.retrieve_one_form(expr(relation)(description.base, value),
                                      active_bindings)

  def ask_with_inheritance(self, query, active_bindings):
    # Handles queries of slots that can be inherited.
    thing = subst(active_bindings, query.args[0])
    slot_path = ([subst(active_bindings, expr(query.op))] +
                 map(lambda s: subst(active_bindings, s), query.args[1:-1]))
    slot_value = subst(active_bindings, query.args[-1])
    if is_var_symbol(slot_value.op) and is_var_symbol(thing.op):
      raise Error(
        'Cannot have both an unbound object variable and an unbound value '
        'variable: %s.' % ((query, active_bindings),))
    elif is_var_symbol(slot_value.op):
      # Find the value of a slot
      var = expr('?var')
      for parent in self.all_parents(thing):
        for binding in self.unify_with_propositions(
          apply(slot_path[0],
                [parent] + slot_path[1:] + [var]), {}):
          yield extend(active_bindings, slot_value, binding[var])
          return
    elif is_var_symbol(thing.op):
      # Find all things that have a particular slot-value, including
      # through inheritance.
      answers = map(
        lambda b: b[expr('?x')],
        self.unify_with_propositions(
          apply(slot_path[0], [expr('?x')] + slot_path[1:] + [slot_value]),
          {}))
      for answer in answers:
        for child in self.all_children(answer):
          child_slot_value = self.slot_value2(child, slot_path)
          if child_slot_value is None or child_slot_value == slot_value:
            yield extend(active_bindings, thing, child)
          else:
            return
    else:
      # Just check whether the specified slot has the specified value,
      # no new bindings required.
      if self.slot_value2(thing, slot_path) == slot_value:
        yield active_bindings
      else:
        return

  def slot_value(self, obj, *relations):
    """Returns the value of the specified relations."""
    return self.slot_value2(obj, relations)

  def slot_value2(self, obj, relations):
    """Returns the value of the specified relations."""
    if (not (isinstance(obj, str) or
             isinstance(obj, Expr) or
             isinstance(obj, Description))):
      raise Error('%r is neither a string, a Description nor an Expr.' % (
          obj),)
    for relation in relations:
      if not (isinstance(relation, str) or isinstance(relation, Expr)):
        raise Error('%s is neither a string nor an Expr.' % (repr(relation),))

    obj = expr(obj)
    relations = map(expr, relations)
    if not isinstance(obj, Expr):
      raise Error('Object %s must be an Expr.' % (obj,))
    for relation in relations:
      if not isinstance(relation, Expr):
        raise Error('Slots argument %s contains a non-Expr: %s.' % (
            relations, relation))
    var = expr('?v')
    query = apply(relations[0], (obj,) + tuple(relations[1:]) + (var,))
    bindings = self.ask(query)
    if bindings is not False and var in bindings:
      value = bindings[var]
    else:
      value = None
    return value

  def slot_path_value(self, obj, *relations):
    for relation in relations:
      obj = self.slot_value(obj, relation)
    return obj

  # Cache for memoizing all_parents.
  parent_cache = {}

  def all_parents(self, obj):
    """Returns a list of all parents of obj, from most specific to
    least specific.
    """
    # Check the memoization cache first.
    if obj in self.parent_cache:
      return self.parent_cache[obj]

    if not isinstance(obj, Expr):
      raise Error('%s must be an Expr.' % (obj,))
    var = expr('?x')
    query = expr('ISA')(obj, var)
    solutions = self.ask_all(query)
    parents = map(lambda b: b[var], solutions)
    self.parent_cache[obj] = parents
    return parents

  def all_proper_parents(self, obj):
    return self.all_parents(obj)[1:]

  # Cache for memoizing all_children.
  child_cache = {}

  def all_children(self, obj):
    """Returns a list of all children of obj."""
    # Check the memoization cache first.
    if obj in self.child_cache:
      return self.child_cache[obj]

    if not isinstance(obj, Expr):
      raise Error('%s must be an Expr.' % (obj,))
    children = list(self.all_children_generator(obj))
    self.child_cache[obj] = children
    return children

  def all_children_generator(self, obj):
    if not isinstance(obj, Expr):
      raise Error('%s must be an Expr.' % (obj,))
    for binding in self.ask_generator(expr('ISA(?x, %s)' % (obj.op,))):
      yield binding[expr('?x')]

  def invalidate_isa_cache(self):
    # Clears the memoization caches for all_parents and all_children.
    self.parent_cache = {}
    self.child_cache = {}

  def all_proper_children(self, obj):
    """Returns a list of all proper children of obj, i.e. the list
    does not contain obj itself.
    """
    return self.all_children(obj)[1:]

  def isa(self, a, b):
    """Checks that a is a child of b."""
    if not isinstance(b, Expr):
      raise Error('%s must be an Expr.' % (b,))
    if not isinstance(a, Expr):
      raise Error('%s must be an Expr.' % (a,))
    # We do this normalization stuff because of the scenario where a
    # is a description, and we're checking whether it's isa its base
    # class.  Then the first element in the list of parents is the
    # description itself, followed by the immediate parent of its base
    # class.  When we go to check whether b is in the list of parents,
    # we end up comparing b to a description of b, and we determine
    # that it is not ISA.
    normalized_parents = map(normalize_description, self.all_parents(a))
    normalized_b = normalize_description(b)
    return normalized_b in normalized_parents

  def install_function(self, op, function):
    """Registers a memory function."""
    self.functions[op] = function

  def ask_function(self, query, s):
    # Handles queries that involve a memory function.
    function = self.functions[query.op]
    bindings = function.ask(self, query, s)
    for bindings in bindings:
      yield bindings

  def ask_and(self, query, s):
    # Handles conjunctive queries (could be implemented as a
    # memory function).
    arg1 = query.args[0]
    arg2 = query.args[1]
    for bindings in self.ask_generator(arg1, s):
      for more_bindings in self.ask_generator(arg2, bindings):
        if more_bindings is not None:
          yield more_bindings

  def ask_or(self, query, s):
    # Handles disjunctive queries (could be implemented as a
    # memory function).
    arg1 = query.args[0]
    arg2 = query.args[1]
    # Perform both subqueries and yield all bindings from each.
    for bindings in self.ask_generator(arg1, s):
      yield bindings
    for bindings in self.ask_generator(arg2, s):
      yield bindings

  def install_standard_functions(self):
    # Installs a set of standard memory functions.
    self.install_function('<=>', EqualFunction())
    self.install_function('~', NotFunction())
    self.install_function('Bind', BindFunction())
    self.install_function('ISA', ISAFunction())
    self.install_function('INSTANCEOF', INSTANCEOFFunction())
    self.install_function('ISINSTANCE', ISINSTANCEFunction())

  def install_standard_propositions(self):
    # Installs a set of standard propositions and fluents.
    pass

  def define_fluent(self, relation, inherited=False):
    """Defines a relation as a fluent."""
    if not isinstance(relation, Expr):
      raise Error('%s must be an Expr.' % (relation,))
    self.fluents.append(relation.op)
    if inherited:
      self.heritable.append(relation.op)

  def retrieve(self, form):
    # Retrieves matching propositions from the discrimination tree.
    return self.tree.retrieve(form)

  def add_proposition(self, clause):
    # Adds a proposition to the discrimination tree.
    form = clause.form()
    if form[0] in self.fluents:
      # If it's a fluent, need to erase any previous value.
      self.retract_propositions_with_form(form[0:-1] + [expr('?x')])
    try:
      old_prop = self.retrieve(form).next()
      reuse_proposition(old_prop, form)
    except StopIteration:
      new_prop = Proposition(form)
      self.tree.put(form, new_prop)

  def retract_propositions_with_form(self, form):
    # Removes matching propositions from the discrimination tree.
    for proposition in list(self.tree.retrieve(form)):
      self.tree.erase(proposition.form)


class MemoryFunction(object):
  """MemoryFunctions are ways to hook arbitrary code into queries (and
  assertions and retractions).  MemoryFunctions may define ask, tell
  and retract methods which will be called in the appropriate
  contexts.
  """
  def __init__(self):
    pass

  def ask(self, kb, clause, s):
    raise NotImplementedError

  def tell(self, kb, clause):
    raise NotImplementedError

  def retract(self, kb, clause):
    raise NotImplementedError


class NotFunction(MemoryFunction):
  """A memory function implementation of logical NOT (~)."""
  def ask(self, kb, expression, s):
    try:
      kb.ask_generator(expression.args[0], s).next()
      return
    except StopIteration:
      yield s

  def tell(self, kb, clause):
    raise NotImplementedError

  def retract(self, kb, clause):
    raise NotImplementedError


class EqualFunction(MemoryFunction):
  """A memory function implementation of logical equivalence."""
  def ask(self, kb, expression, s):
    arg1 = subst(s, expression.args[0])
    arg2 = subst(s, expression.args[1])
    if arg1 == arg2:
      yield s
    else:
      return

  def tell(self, kb, clause):
    raise NotImplementedError

  def retract(self, kb, clause):
    raise NotImplementedError


class BindFunction(MemoryFunction):
  """A memory function that creates a binding (assigns a variable a
  value).
  """
  def ask(self, kb, expression, s):
    if expression.args[0] in s:
      return
    else:
      yield extend(s, expression.args[0], expression.args[1])

  def tell(self, kb, clause):
    raise NotImplementedError

  def retract(self, kb, clause):
    raise NotImplementedError


class ISAFunction(MemoryFunction):
  """A memory function implementing ISA queries and assertions."""
  def tell(self, kb, sentence):
    kb.assert_isa(sentence.args[0], sentence.args[1])

  def retract(self, kb, clause):
    raise NotImplementedError

  def _direct_children(self, kb, concept):
    v = expr('?x')
    isaexpr = expr('$ISA')(v, concept)
    for s in kb.ask_generator(isaexpr):
      yield s[v]

  def _direct_parents(self, kb, concept):
    v = expr('?x')
    isaexpr = expr('$ISA')(concept, v)
    for s in kb.ask_generator(isaexpr):
      yield s[v]

  def _all_children(self, kb, concept):
    yield concept
    for child in self._direct_children(kb, concept):
      for subchild in self._all_children(kb, child):
        yield subchild

  def _all_parents(self, kb, concept):
    yield concept
    for parent in self._direct_parents(kb, concept):
      for superparent in self._all_parents(kb, parent):
        yield superparent

  def ask(self, kb, query, s):
    subst_query = subst(s, query)
    subst_child = subst_query.args[0]
    subst_parent = subst_query.args[1]

    if is_var_symbol(subst_child.op) and not is_var_symbol(subst_parent.op):
      for child in self._all_children(kb, subst_parent):
        yield extend(s, subst_child, child)
    elif ((not is_var_symbol(subst_child.op)) and
          is_var_symbol(subst_parent.op)):
      for parent in self._all_parents(kb, subst_child):
        yield extend(s, subst_parent, parent)
    elif ((not is_var_symbol(subst_child.op)) and
          (not is_var_symbol(subst_parent.op))):
      if (normalize_description(subst_child) in
          self._all_children(kb, subst_parent)):
        yield s
      else:
        pass
    else:
      raise Error('Have two unbound variables in ISA query: %s.' % (query,))


class INSTANCEOFFunction(MemoryFunction):
  """Memory function INSTANCEOF."""
  def tell(self, kb, sentence):
    kb.assert_instanceof(sentence.args[0], sentence.args[1])

  def ask(self, kb, query, s):
    subst_query = subst(s, query)
    subst_child = subst_query.args[0]
    subst_parent = subst_query.args[1]
    newquery = (expr('$ISA')(subst_child, subst_parent) &
                expr('ISINSTANCE')(subst_child))
    for bindings in kb.ask_generator(newquery, s):
      yield bindings

  def retract(self, kb, clause):
    raise NotImplementedError


class ISINSTANCEFunction(MemoryFunction):
  """Memory function ISINSTANCE."""
  def ask(self, kb, query, s):
    if len(query.args) > 1:
      raise Error(
        'ISINSTANCE has arity 1 but was used with %s arguments: %s' % (
          len(query.args), query))
    return kb.ask_generator(expr('$INSTANCE')(query.args[0]), s)

  def tell(self, kb, clause):
    raise NotImplementedError

  def retract(self, kb, clause):
    raise NotImplementedError


def newuniquevar(sentence, bindings, count=1):
  """Returns a new variable that does not appear in the sentence."""
  var = expr('?x%s' % (count,))
  if occur_check(var, sentence) or var in bindings:
    return newuniquevar(sentence, bindings, count + 1)
  else:
    return var


def newuniquevar2(usedvars, count=1):
  """Returns a new variable that isn't in usedvars."""
  var = expr('?x%s' % (count,))
  if var in usedvars:
    return newuniquevar2(usedvars, count + 1)
  else:
    return var


def removevars(s, variables):
  """Removes a set of variables from a substitution."""
  for su in s:
    for var in variables:
      if var in su:
        del su[var]
    yield su

#______________________________________________________________________________


class Description:
  """Descriptions are sort of like 'frame literals'.  They consist of
  a class and a set of slots and slot values, but none of the values
  are asserted to memory.

  The find_all method can be used to find all frames in a KB that
  match the Description.  To match, a frame must be of a class that
  ISA the Description class and must have at least the same slot
  values that the Description has.
  """
  def __init__(self, base, slots=None):
    if slots is None:
      slots = {}
    self.base = expr(base)
    self.slots = {}
    for s in slots:
      self.slots[expr(s)] = expr(slots[s])

  def __repr__(self):
    if len(self.slots) > 0:
      return '<%s base: %s slots: %s>' % (
        self.__class__.__name__, self.base, self.slots)
    else:
      return '<%s base: %s>' % (self.__class__.__name__, self.base)

  def __hash__(self):
    # Need a hash method so Exprs can live in dicts.
    return hash(self.base) ^ hash(tuple(self.slots.keys()))

  def __cmp__(self, other):
    if (other is self) or (isinstance(other, Description) and
                           self.base == other.base and
                           self.slots == other.slots):
      return 0
    else:
      # Not quite complete here.
      return -1

  def dictify(self):
    newslots = {}
    for slot in self.slots:
      slotval = self.slots[slot].op
      if isinstance(slotval, Description):
        newslots[slot.op] = slotval.dictify()
      else:
        newslots[slot.op] = slotval
    if len(newslots) > 0:
      return {'base': self.base.op, 'slots': newslots}
    else:
      return {'base': self.base.op}

  def pprint(self):
    pprint.pprint(self.dictify(), width=80)

  # --------------------
  # The more public part of the Description API
  # --------------------

  def slot_value(self, slot_name):
    """Returns the value of a slot."""
    slot_name = expr(slot_name)
    return self.slots[slot_name].op

  def has_slot(self, slot_name):
    """Checks whether the description has the specified slot."""
    slot_name = expr(slot_name)
    return slot_name in self.slots

  def find_all(self, kb):
    """Finds all objects matching this description."""
    return list(self.find_generator(kb))

  def find_instances(self, kb):
    """Finds all instances matching this description."""
    return list(self.find_generator(kb, instances_only=True))

  # --------------------
  # The less public part of the Description API.
  # --------------------

  def find_generator(self, kb, instances_only=False):
    objvar = expr('?x')
    query = self.query(objvar)
    if instances_only:
      query = query & expr('ISINSTANCE')(objvar)
    for bindings in kb.ask_generator(query):
      yield bindings[objvar]

  def query(self, objvar):
    """Returns a query that will find objects matching the
    description.
    """
    return self.make_query(objvar, [])[0]

  def make_query(self, objvar, usedvars):
    # Returns a query to find objects matching the description.
    [slot_query, usedvars] = self.slots_query(objvar, usedvars)
    isa_query = expr('ISA')(objvar, self.base)
    if slot_query is None:
      return [isa_query, usedvars]
    else:
      return [isa_query & slot_query, usedvars]

  def slots_query(self, objvar, usedvars):
    # Returns a query for the slots of the description.
    if len(self.slots) > 0:
      slot_queries = []
      for slot in self.slots:
        [query, usedvars] = self.slot_query(slot, objvar, usedvars)
        slot_queries.append(query)
      return [reduce(Expr.__and__, slot_queries), usedvars]
    else:
      return [None, usedvars]

  def slot_query(self, slot, objvar, usedvars):
    # Returns a query for a single slot.
    slot_value = self.slots[slot]
    if isinstance(slot_value.op, Description):
      dvar = newuniquevar2(usedvars)
      [dquery, newusedvars] = slot_value.op.make_query(dvar, usedvars + [dvar])
      return [slot(objvar, dvar) & dquery, usedvars + newusedvars]
    else:
      return [slot(objvar, slot_value), usedvars]


class Expr:
  """A symbolic mathematical expression.  We use this class for
  logical expressions, and for terms within logical expressions. In
  general, an Expr has an op (operator) and a list of args.  The op
  can be:

    Null-ary (no args) op:
      A number, representing the number itself.  (e.g. Expr(42) => 42)
      A symbol, representing a variable or constant (e.g. Expr('F') => F)
    Unary (1 arg) op:
      '~', '-', representing NOT, negation (e.g. Expr('~', Expr('P')) => ~P)
    Binary (2 arg) op:
      '>>', '<<', representing forward and backward implication
      '+', '-', '*', '/', '**', representing arithmetic operators
      '<', '>', '>=', '<=', representing comparison operators
      '<=>', '^', representing logical equality and XOR
    N-ary (0 or more args) op:
      '&', '|', representing conjunction and disjunction
      A symbol, representing a function term or FOL proposition

  Exprs can be constructed with operator overloading: if x and y are
  Exprs, then so are x + y and x & y, etc.  Also, if F and x are
  Exprs, then so is F(x); it works by overloading the __call__ method
  of the Expr F.  Note that in the Expr that is created by F(x), the
  op is the str 'F', not the Expr F.  See
  http://www.python.org/doc/current/ref/specialnames.html to learn
  more about operator overloading in Python.

  WARNING: x == y and x != y are NOT Exprs.  The reason is that we
  want to write code that tests 'if x == y:' and if x == y were the
  same as Expr('==', x, y), then the result would always be true; not
  what a programmer would expect.  But we still need to form Exprs
  representing equalities and disequalities.  We concentrate on
  logical equality (or equivalence) and logical disequality (or XOR).
  You have 3 choices:

    (1) Expr('<=>', x, y) and Expr('^', x, y)
      Note that ^ is bitwose XOR in Python (and Java and C++)
    (2) expr('x <=> y') and expr('x =/= y').
      See the doc string for the function expr.
    (3) (x % y) and (x ^ y).
      It is very ugly to have (x % y) mean (x <=> y), but we need
      SOME operator to make (2) work, and this seems the best choice.

  WARNING: if x is an Expr, then so is x + 1, because the int 1 gets
  coerced to an Expr by the constructor.  But 1 + x is an error,
  because 1 doesn't know how to add an Expr.  (Adding an __radd__
  method to Expr wouldn't help, because int.__add__ is still called
  first.) Therefore, you should use Expr(1) + x instead, or ONE + x,
  or expr('1 + x').
  """

  def __init__(self, op, *args):
    """Op is a string or number; args are Exprs (or are coerced to Exprs)."""
    assert (isinstance(op, str) or
            isinstance(op, Description) or
            (utils.is_number(op) and not args))
    if isinstance(op, Description):
      self.op = op
    else:
      self.op = utils.num_or_str(op)
    self.args = map(expr, args)  # Coerce args to Exprs

  def __call__(self, *args):
    """Self must be a symbol with no args, such as Expr('F').  Create
    a new Expr with 'F' as op and the args as arguments.
    """
    if not(is_symbol(self.op) and not self.args):
      raise Error('%s is not a symbol or there are args: %s' % (
          self.op, self.args))
    return Expr(self.op, *args)

  def __repr__(self):
    return "<%s: '%s'>" % (x.__class__.__name__, str(self))

  def __str__(self):
    """Show something like 'P' or 'P(x, y)', or '~P' or '(P | Q | R)'"""
    if len(self.args) == 0:  # Constant or proposition with arity 0
      return str(self.op)
    elif is_symbol(self.op):  # Functional or Propositional operator
      return '%s(%s)' % (self.op, ', '.join(map(str, self.args)))
    elif len(self.args) == 1:  # Prefix operator
      return self.op + repr(self.args[0])
    else:  # Infix operator
      return '(%s)' % (' ' + self.op + ' ').join(map(str, self.args))

  def __cmp__(self, other):
    """x and y are equal iff their ops and args are equal."""
    if (other is self or
        (isinstance(other, Expr) and
         self.op == other.op and self.args == other.args)):
      return 0
    else:
      return -1

  def __hash__(self):
    """Need a hash method so Exprs can live in dicts."""
    return hash(self.op) ^ hash(tuple(self.args))

  # See http://www.python.org/doc/current/lib/module-operator.html
  # Not implemented: not, abs, pos, concat, contains, *item, *slice
  def __lt__(self, other):
    return Expr('<', self, other)

  def __le__(self, other):
    return Expr('<=', self, other)

  def __ge__(self, other):
    return Expr('>=', self, other)

  def __gt__(self, other):
    return Expr('>', self, other)

  def __add__(self, other):
    return Expr('+', self, other)

  def __sub__(self, other):
    return Expr('-', self, other)

  def __and__(self, other):
    return Expr('&', self, other)

  def __div__(self, other):
    return Expr('/', self, other)

  def __truediv__(self, other):
    return Expr('/', self, other)

  def __invert__(self):
    return Expr('~', self)

  def __lshift__(self, other):
    return Expr('<<', self, other)

  def __rshift__(self, other):
    return Expr('>>', self, other)

  def __mul__(self, other):
    return Expr('*', self, other)

  def __neg__(self):
    return Expr('-', self)

  def __or__(self, other):
    return Expr('|', self, other)

  def __pow__(self, other):
    return Expr('**', self, other)

  def __xor__(self, other):
    return Expr('^', self, other)

  def __mod__(self, other):
    return Expr('<=>', self, other)  # (x % y)

  def form(self):
    return [self.op] + self.args


EXPR_TABLE = {}


def expr(s):
  # This little memoization cuts our calls to eval by about 99%.
  if s in EXPR_TABLE:
    return EXPR_TABLE[s]
  e = expr2(s)
  EXPR_TABLE[s] = e
  return e


def expr2(s):
  """Create an Expr representing a logic expression by parsing the
  input string. Symbols and numbers are automatically converted to
  Exprs.  In addition you can use alternative spellings of these
  operators:
    'x ==> y'   parses as   (x >> y)    # Implication
    'x <== y'   parses as   (x << y)    # Reverse implication
    'x <=> y'   parses as   (x % y)     # Logical equivalence
    'x =/= y'   parses as   (x ^ y)     # Logical disequality (xor)
  But BE CAREFUL; precedence of implication is wrong. expr('P & Q ==> R & S')
  is ((P & (Q >> R)) & S); so you must use expr('(P & Q) ==> (R & S)').
  >>> expr('P <=> Q(1)')
  (P <=> Q(1))
  >>> expr('P & Q | ~R(x, F(x))')
  ((P & Q) | ~R(x, F(x)))
  """
  if isinstance(s, Expr):
    return s
  if utils.is_number(s):
    return Expr(s)
  if isinstance(s, Description):
    return Expr(s)
  ## Replace the alternative spellings of operators with canonical spellings
  s = s.replace('==>', '>>').replace('<==', '<<')
  s = s.replace('<=>', '%').replace('=/=', '^')
  ## Replace a symbol or number, such as 'P' with 'Expr('P')'
  s = re.sub(r'([a-zA-Z0-9_\-$.?]+)', r"Expr('\1')", s)
  ## Now eval the string.  (A security hole; do not use with an adversary.)
#    print 'EVALLING: %s' % (s,)
  return eval(s, {'Expr': Expr})


def is_symbol(s):
  """A string s is a symbol if it starts with an alphabetic char, $ or ?."""
  return isinstance(s, str) and (s[0].isalpha() or s[0] == '$' or s[0] == '?')


def is_var_symbol(s):
  """A logic variable symbol is a string that begins with '?'."""
  return is_symbol(s) and s[0] == '?'


def is_prop_symbol(s):
  """A proposition logic symbol is an initial-uppercase string other than
  TRUE or FALSE."""
  return is_symbol(s) and s[0] != '?' and s != 'TRUE' and s != 'FALSE'


## Useful constant Exprs used in examples and code:
TRUE, FALSE, ZERO, ONE, TWO = map(Expr, ['TRUE', 'FALSE', 0, 1, 2])
A, B, C, F, G, P, Q, x, y, z = map(Expr, 'ABCFGPQxyz')


def tt_entails(kb, alpha):
  """Use truth tables to determine if KB entails sentence alpha. [Fig. 7.10]
  >>> tt_entails(expr('P & Q'), expr('Q'))
  True
  """
  return tt_check_all(kb, alpha, prop_symbols(kb & alpha), {})


def tt_check_all(kb, alpha, symbols, model):
  """Auxiliary routine to implement tt_entails."""
  if not symbols:
    if pl_true(kb, model):
      result = pl_true(alpha, model)
      assert result in (True, False)
      return result
    else:
      return True
  else:
    P, rest = symbols[0], symbols[1:]
    return (tt_check_all(kb, alpha, rest, extend(model, P, True)) and
            tt_check_all(kb, alpha, rest, extend(model, P, False)))


def prop_symbols(obj):
  """Return a list of all propositional symbols in obj."""
  if not isinstance(obj, Expr):
    return []
  elif is_prop_symbol(obj.op):
    return [obj]
  else:
    s = set(())
    for arg in obj.args:
      for symbol in prop_symbols(arg):
        s.add(symbol)
    return list(s)


def tt_true(alpha):
  """Is the sentence alpha a tautology? (alpha will be coerced to an expr.)
  >>> tt_true(expr('(P >> Q) <=> (~P | Q)'))
  True
  """
  return tt_entails(TRUE, expr(alpha))


def pl_true(exp, model=None):
  """Return True if the propositional logic expression is true in the
  model, and False if it is false. If the model does not specify the
  value for every proposition, this may return None to indicate 'not
  obvious'; this may happen even when the expression is
  tautological.
  """
  if model is None:
    model = {}
  op, args = exp.op, exp.args
  if exp == TRUE:
    return True
  elif exp == FALSE:
    return False
  elif is_prop_symbol(op):
    return model.get(exp)
  elif op == '~':
    p = pl_true(args[0], model)
    if p is None:
      return None
    else:
      return not p
  elif op == '|':
    result = False
    for arg in args:
      p = pl_true(arg, model)
      if p is True:
        return True
      if p is None:
        result = None
    return result
  elif op == '&':
    result = True
    for arg in args:
      p = pl_true(arg, model)
      if p is False:
        return False
      if p is None:
        result = None
    return result
  p, q = args
  if op == '>>':
    return pl_true(~p | q, model)
  elif op == '<<':
    return pl_true(p | ~q, model)
  pt = pl_true(p, model)
  if pt is None:
    return None
  qt = pl_true(q, model)
  if qt is None:
    return None
  if op == '<=>':
    return pt == qt
  elif op == '^':
    return pt != qt
  else:
    raise Error('illegal operator in logic expression' + str(exp))


## Convert to Conjunctive Normal Form (CNF)

def to_cnf(s):
  """Convert a propositional logical sentence s to conjunctive normal form.
  That is, of the form ((A | ~B | ...) & (B | C | ...) & ...) [p. 215]
  >>> to_cnf('~(B|C)')
  (~B & ~C)
  >>> to_cnf('B <=> (P1|P2)')
  ((~P1 | B) & (~P2 | B) & (P1 | P2 | ~B))
  >>> to_cnf('a | (b & c) | d')
  ((b | a | d) & (c | a | d))
  >>> to_cnf('A & (B | (D & E))')
  (A & (D | B) & (E | B))
  """
  if isinstance(s, str):
    s = expr(s)
  s = eliminate_implications(s)  # Steps 1, 2 from p. 215
  s = move_not_inwards(s)  # Step 3
  return distribute_and_over_or(s)  # Step 4


def eliminate_implications(s):
  """Change >>, <<, and <=> into &, |, and ~. That is, return an Expr
  that is equivalent to s, but has only &, |, and ~ as logical operators.
  >>> eliminate_implications(A >> (~B << C))
  ((~B | ~C) | ~A)
  """
  if not s.args or is_symbol(s.op):
    return s  # (Atoms are unchanged.)
  args = map(eliminate_implications, s.args)
  a, b = args[0], args[-1]
  if s.op == '>>':
    return (b | ~a)
  elif s.op == '<<':
    return (a | ~b)
  elif s.op == '<=>':
    return (a | ~b) & (b | ~a)
  else:
    return Expr(s.op, *args)


def move_not_inwards(s):
  """Rewrite sentence s by moving negation sign inward.
  >>> move_not_inwards(~(A | B))
  (~A & ~B)
  >>> move_not_inwards(~(A & B))
  (~A | ~B)
  >>> move_not_inwards(~(~(A | ~B) | ~~C))
  ((A | ~B) & ~C)
  """
  if s.op == '~':
    NOT = lambda b: move_not_inwards(~b)
    a = s.args[0]
    if a.op == '~':
      return move_not_inwards(a.args[0])  # ~~A ==> A
    if a.op == '&':
      return NaryExpr('|', *map(NOT, a.args))
    if a.op == '|':
      return NaryExpr('&', *map(NOT, a.args))
    return s
  elif is_symbol(s.op) or not s.args:
    return s
  else:
    return Expr(s.op, *map(move_not_inwards, s.args))


def distribute_and_over_or(s):
  """Given a sentence s consisting of conjunctions and disjunctions
  of literals, return an equivalent sentence in CNF.
  >>> distribute_and_over_or((A & B) | C)
  ((A | C) & (B | C))
  """
  if s.op == '|':
    s = NaryExpr('|', *s.args)
    if len(s.args) == 0:
      return FALSE
    if len(s.args) == 1:
      return distribute_and_over_or(s.args[0])
    conj = utils.find_if((lambda d: d.op == '&'), s.args)
    if not conj:
      return NaryExpr(s.op, *s.args)
    others = [a for a in s.args if a is not conj]
    if len(others) == 1:
      rest = others[0]
    else:
      rest = NaryExpr('|', *others)
    return NaryExpr('&', *map(distribute_and_over_or,
                              [(c | rest) for c in conj.args]))
  elif s.op == '&':
    return NaryExpr('&', *map(distribute_and_over_or, s.args))
  else:
    return s


_NaryExprTable = {
  '&': TRUE,
  '|': FALSE,
  '+': ZERO,
  '*': ONE
  }


def NaryExpr(op, *args):
  """Create an Expr, but with an nary, associative op, so we can promote
  nested instances of the same op up to the top level.
  >>> NaryExpr('&', (A&B),(B|C),(B&C))
  (A & B & (B | C) & B & C)
  """
  arglist = []
  for arg in args:
    if arg.op == op:
      arglist.extend(arg.args)
    else:
      arglist.append(arg)
  if len(args) == 1:
    return args[0]
  elif len(args) == 0:
    return _NaryExprTable[op]
  else:
    return Expr(op, *arglist)


def conjuncts(s):
  """Return a list of the conjuncts in the sentence s.
  >>> conjuncts(A & B)
  [A, B]
  >>> conjuncts(A | B)
  [(A | B)]
  """
  if isinstance(s, Expr) and s.op == '&':
    return s.args
  else:
    return [s]


def disjuncts(s):
  """Return a list of the disjuncts in the sentence s.
  >>> disjuncts(A | B)
  [A, B]
  >>> disjuncts(A & B)
  [(A & B)]
  """
  if isinstance(s, Expr) and s.op == '|':
    return s.args
  else:
    return [s]


def unify(x, y, s):
  """Unify expressions x,y with substitution s; return a substitution that
  would make x,y equal, or None if x,y can not unify. x and y can be
  variables (e.g. Expr('x')), constants, lists, or Exprs. [Fig. 9.1]
  >>> unify(x + y, y + C, {})
  {y: C, x: y}
  """
  if s is None:
    return None
  elif x == y:
    return s
  elif is_variable(x):
    return unify_var(x, y, s)
  elif is_variable(y):
    return unify_var(y, x, s)
  elif isinstance(x, Expr) and isinstance(y, Expr):
    return unify(x.args, y.args, unify(x.op, y.op, s))
  elif isterm(x) or isterm(y) or not x or not y:
    return utils.if_(x == y, s, None)
  elif utils.is_sequence(x) and utils.is_sequence(y) and len(x) == len(y):
    return unify(x[1:], y[1:], unify(x[0], y[0], s))
  else:
    return None


def is_variable(obj):
  """A variable is an Expr with no args and a lowercase symbol as the op."""
  return isinstance(obj, Expr) and not obj.args and is_var_symbol(obj.op)


def unify_var(var, obj, s):
  if var in s:
    return unify(s[var], obj, s)
  elif occur_check(var, obj):
    return None
  else:
    return extend(s, var, obj)


def occur_check(var, obj):
  """Return true if var occurs anywhere in obj."""
  if var == obj:
    return True
  elif isinstance(obj, Expr):
    return var.op == obj.op or occur_check(var, obj.args)
  elif not isterm(obj) and utils.is_sequence(obj):
    for xi in obj:
      if occur_check(var, xi):
        return True
  return False


def isterm(item):
  return isinstance(item, str) or isinstance(item, Description)


def extend(s, var, val):
  """Copy the substitution s and extend it by setting var to val; return copy.
  >>> extend({x: 1}, y, 2)
  {y: 2, x: 1}
  """

  if not isinstance(var, Expr):
    raise Error('%s must be an Expr.' % (var,))
  if not isinstance(val, Expr):
    raise Error('%s must be an Expr.' % (val,))
  s2 = s.copy()
  s2[var] = val
  return s2


def subst(s, expression):
  """Substitute the substitution s into the expression x.
  >>> subst({x: 42, y:0}, F(x) + y)
  (F(42) + 0)
  """
  if isinstance(expression, list):
    return [subst(s, xi) for xi in expression]
  elif isinstance(expression, tuple):
    return tuple([subst(s, expression) for xi in expression])
  elif not isinstance(expression, Expr):
    return expression
  elif is_var_symbol(expression.op):
    return s.get(expression, expression)
  else:
    return Expr(expression.op, *[subst(s, arg) for arg in expression.args])


def normalize_description(expression):
  if not isinstance(expression, Expr):
    raise Error('%s is not an Expr' % (expression,))
  if isinstance(expression.op, Description):
    newexpr = subst({}, expression)
    newexpr.op = expression.op.base.op
    return newexpr
  else:
    return expression


def deconstruct_query(query):
  return ([query.op] + query.args[1:-1], query.args[0], query.args[-1])
