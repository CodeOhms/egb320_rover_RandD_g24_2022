#!/usr/bin/bash

echo "python -m cProfile -o ../profiles/$2.prof $1$2.py"
python -m cProfile -o ../profiles/$2.prof $1$2.py
