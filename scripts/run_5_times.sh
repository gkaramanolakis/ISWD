#!/bin/bash
cd ../iswd/
python main.py "$@" --version 0
python main.py "$@" --version 1
python main.py "$@" --version 2
python main.py "$@" --version 3
python main.py "$@" --version 4
cd - 
