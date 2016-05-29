#!/usr/bin/env python
# -*- coding: utf-8 -*-
import hdplda
import sys
import os
import numpy
from hdplda import hdplda_learning

print '\n\n usage: python run_hdplda.py dir_sense dir_corpusread dir_corpuswrite \n\n'

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
    
    alpha = numpy.random.gamma(1, 1)  #new table  bigger alpha = fewer topics
    gamma = numpy.random.gamma(1, 1)  #new k   bigger gamma = more topics

    base = 0.5   # bigger base = smoother old K
    iteration = 5000
    print 'alpha: %f  gamma: %f  base: %f  iteration: %d' %(alpha, gamma, base, iteration)
#    print 'alpha:' + str(alpha) + 'gamma' + str(gamma) + 'base' + str(base) + 'iteration' + str(iteration)
    hdplda0 = hdplda.HDPLDA(alpha, gamma, base, docs, len(voca))
    hdplda_learning(hdplda0, iteration, voca, corpuswrite1, corpuswrite2)
