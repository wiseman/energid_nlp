#!/bin/bash

rm .coverage
for package in utils discriminationtree logic parser fdl
do
  PYTHONPATH=. python-coverage run --append --source=energid_nlp.${package} energid_nlp/tests/${package}_test.py
done
rm -rf htmlcov && python-coverage html
