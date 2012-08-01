# Copyright 2012 Energid Technologies

from energid_nlp import logic


GENERATION_PROPOSITION = '$generate'


class Error(Exception):
  pass


class Generator:
  def __init__(self, kb):
    self.kb = kb

  def generate_prim(self, concept):
    if isinstance(concept, logic.Description):
      return self.generate_prim(concept.base)
    elif isinstance(concept, logic.Expr):
      return self.generate_prim(concept.op)
    else:
      result = '%s' % (concept,)
      return str(result)

  def generate(self, concept):
    if (isinstance(concept, str) or
        isinstance(concept, logic.Description) or
        isinstance(concept, logic.Expr)):
      template = self.kb.slot_value(concept, GENERATION_PROPOSITION)
      if template is None:
        return self.generate_prim(concept)
      else:
        return self.generate_template(concept, template.op)
    else:
      return self.generate_prim(concept)

  def generate_template(self, concept, template):
    result_string = ''
    start = 0
    while start < len(template):
      slot_start = template.find('{', start)
      if slot_start == -1:
        # No more slot refs
        result_string = result_string + template[start:]
        break
      result_string = result_string + template[start:slot_start]

      slot_end = template.find('}', slot_start + 1)
      if slot_end == -1:
        raise Error("Generation template %r for %s has an unclosed '{'" % (
            template, concept))
      slot_name = template[slot_start + 1:slot_end]
      slot_value = self.kb.slot_value(concept, slot_name)
      if slot_value is not None:
        result_string = result_string + self.generate(slot_value)
      start = slot_end + 1
    # Strip whitepace out of result, in case slots in template
    # couldn't be filled.
    return result_string.strip()
