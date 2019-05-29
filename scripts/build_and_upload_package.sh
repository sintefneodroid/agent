#!/usr/bin/env bash
scripts/clean_generated_files.sh
#scripts/clean_package.sh
#python2 setup.py bdist
#python2 setup.py bdist_wheel
#python3 setup.py bdist
python3 setup.py bdist_wheel
twine upload dist/*
#scripts/clean_package.sh
