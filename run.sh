#!/bin/bash
PYTHONPATH=. python3 run.py --dataset base0-1-merge --algorithm faiss-ivf --definitions algosDeep.yaml --runs 1 --local 2>&1 | tee out_new.txt