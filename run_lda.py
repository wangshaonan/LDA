#!/usr/bin/env python
# -*- coding: utf-8 -*-

import lda
import sys
import os
from lda import lda_learning
import re

print '\n\n usage: python run_lda.py dir_sense dir_corpusread dir_corpuswrite  \n\n'

corpuslist = os.listdir(sys.argv[1])
for corpusfile in corpuslist:
    print corpusfile
    corpus = open(sys.argv[2]+'/'+corpusfile, 'r').readlines()
    corpuswrite1 = open(sys.argv[3]+'/'+corpusfile+'.phi', 'w')
    corpuswrite2 = open(sys.argv[3]+'/'+corpusfile+'.theta', 'w')
    docs = []
    for line in corpus:
        line = line.strip()
        words = line.split()
        tmp_docs = []
        if len(words) == 0:
            docs.append([])
        for tmp in words:
            tmp_docs.append(int(tmp))
        docs.append(tmp_docs)

    vv = open(sys.argv[2]+'/'+corpusfile+'.vocab', 'r').readlines()
    voca = []
    for line in vv:
        line = line.strip()
        words = line.split()
        voca.append(words[2])

    K = 2
    alpha = 0.05   # bigger alpha = smoother theta
    beta = 0.05
    iteration = 2
    print 'alpha:' + str(alpha) + 'beta:' + str(beta) + 'iteration:' + str(iteration)
    lda0 = lda.LDA(K, alpha, beta, docs, len(voca))
    lda_learning(lda0, iteration, voca, corpuswrite1, corpuswrite2)
