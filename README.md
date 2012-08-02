# energid-nlp

This is an open source version of Energid's natural language
processing (NLP) Python code.  It provides the following components:

* A DMAP-style parser, `energid_nlp.parser.ConceptualParser`.  Parses
  text into concepts based on a fixed grammar.
* An ICP-style parser, `energid_nlp.parser.IndexedConceptParser`.
  Parses text into best-matching concepts with more grammatical
  flexibility.
* A simple conceptual memory implemented as a propositional knowledge
  base, `energid_nlp.logic.PropKB`, similar to a classic [frame
  system](http://en.wikipedia.org/wiki/Frame_language). This is where
  the concepts that can be returned by the parsers are defined.


## Installation and usage

To run unit tests:

    python setup.py test

To install:

    python setup.py install

To try the interactive parser with a test grammar and knowledge base:

    $ python -m energid_nlp.parser energid_nlp/tests/test.fdl
    ? hog let us restart the talk on
    CP:
    [{'base': 'c-action-request',
      'slots': {'action': {'base': 'c-restart'},
                'addressee': {'base': 'c-hog'},
                'object': {'base': 'c-talkon'}}}]

To run the grammar and KB tests:

    $ PYTHONPATH=. python -m energid_nlp.parser -t energid_nlp/tests/test.fdl
    Class: c-action-request
      hog let us restart the talk on                                         ==> ok
      hog, let's start over with the talkon                                  ==> ok
    Class: c-please-action-request
      please, hog, restart talk on                                           ==> ok
    Ran 3 parse tests, 0 failed.


## Glossary

### Description

Descriptions (`energid_nlp.logic.Description`) are returned as the
results of parsing.  You can think of them as being like "frame
literals".  They can represent a single frame/concept without actually
existing in memory.  Descriptions consist of a class and a set of
slots and slot values.

One thing you can do with Descriptions is find matching concepts in
memory.  Here's an example where the memory contains a concept
representing Petunia, who is a small, gray cat and we find her by
creating a description of small cats:

    kb = logic.PropKB()
    kb.tell(logic.expr('ISA(Petunia, Cat)'))
    kb.tell(logic.expr('Size(Petunia, Small)'))
    kb.tell(logic.expr('Color(Petunia, Gray)'))

    d = logic.Description('Cat', {'Size': 'Small'})
    d.find_all(kb)
    ==> [<Expr: 'Petunia'>]

### Direct Memory Access Parsing (DMAP)

An approach to parsing in which phrasal patterns are attached to
memory structures representing concepts. Text is parsed by a recursive
process of recognizing phrasal patterns in the text and constructing
descriptions of the corresponding concepts.  See Charles Martin's 1990
Ph.D. dissertation, _Direct Memory Access Parsing_.

### Frame

A data structure used to represent concepts in memory.  Frames have a
class (type), and a set of slots and slot values.  Slot values can be
any data type, including other frames.

IS-A relationships are supported, and slots can be defined to be
inheritable.

### Indexed Concept Parsing (ICP)

An approach to parsing that looks for references to to concepts in the
input text and returns concepts that match the text best.
Practically, this means that you can get get successful parses even if
not every word you expect to be present is in the input text or if the
input text contains unexpected words.

See Will Fitzgerald's 1994 Ph.D. dissertation, [_Building Embedded
Conceptual Parsers_](http://www.entish.org/becp/).

### Frame Description Language (FDL)

A way of defining frames and attaching language to them in an XML
format.  Here's a trivial example of defining a frame that represents
a cat named Petunia, with an attached phrasal pattern that lets the
input text "petunia" parse into the concept:

    <frame id="i-petunia">
      <parent id="c-cat" />
      <slot name="name" value="petunia" />
      <phrase>petunia</phrase>
    </frame>

See `energid_nlp/fdl.xsd` for the full XML schema.
`energid_nlp/tests/test.fdl` has a more detailed example.


## License

This software is Copyright 2012, Energid Technologies, and is licensed
under the BSD license (see the accompanying LICENSE.md).

Written by John Wiseman and Michael Hannemann.
