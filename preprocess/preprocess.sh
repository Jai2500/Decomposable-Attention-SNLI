#!/bin/bash

python preprocess-entail.py --srcfile processed_data/src-train.txt --targetfile processed_data/targ-train.txt --labelfile processed_data/label-train.txt --srcvalfile processed_data/src-dev.txt --targetvalfile processed_data/targ-dev.txt --labelvalfile processed_data/label-dev.txt --srctestfile processed_data/src-test.txt --targettestfile processed_data/targ-test.txt --labeltestfile processed_data/label-test.txt --outputfile data/entail --glove /scratch/a9/glove/glove.6B.300d.txt
