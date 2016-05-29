#!/usr/bin/env python
# -*- coding: utf-8 -*-
import dual_hdplda
import sys
import os
import numpy
from dual_hdplda import hdplda_learning

'\n\n usage: run_dual_hdplda dir_sense dir_corpustopic dir_corpuslocal dir_corpusresult \n\n'

corpuslist = os.listdir(sys.argv[1])
for corpusfile in corpuslist:
    print corpusfile
    corpus1 = open(sys.argv[2]+'/'+corpusfile, 'r').readlines()
    corpus2 = open(sys.argv[3]+'/'+corpusfile, 'r')
    corpuswrite1 = open(sys.argv[4]+'/'+corpusfile+'.phi', 'w')
    corpuswrite2 = open(sys.argv[4]+'/'+corpusfile+'.theta', 'w')
    docs1 = []
    for line in corpus1:
        line = line.strip()
        words = line.split()
        tmp_docs = []
        if len(words) == 0:
            docs1.append([])
        for tmp in words:
            tmp_docs.append(int(tmp))
        docs1.append(tmp_docs)

    docs2 = []
    done = 0
    while not done:
        line = corpus2.readline()
        words = line.split()
        tmp_docs = []
        for tmp in words:
            tmp_docs.append(int(tmp))
        docs2.append(tmp_docs)
        if line == '':
            done = 1
    vv1 = open(sys.argv[2]+'/'+corpusfile+'.vocab', 'r').readlines()
    voca1 = []
    for line in vv1:
        line = line.strip()
        words = line.split()
        voca1.append(words[2])
    vv2 = open(sys.argv[3]+'/'+corpusfile+'.vocab', 'r').readlines()
    voca2 = []
    for line in vv2:
        line = line.strip()
        words = line.split()
        voca2.append(words[2])

    if not len(docs1) == len(docs2):
        print 'length of corpus1 is not equal to length of corpus2'

    alpha = numpy.random.gamma(1, 1)  #new table  bigger alpha = fewer topics
    gamma = numpy.random.gamma(1, 1)  #new k   bigger gamma = more topics
    base = 0.5   # bigger base = smoother old K
    weight = [1, 1]
    iteration = 5000
    print 'alpha: %f  gamma: %f  base: %f  weight:[ %f %f ]  iteration: %d ' %(alpha, gamma, base, weight[0], weight[1], iteration)
#    print 'alpha:' + str(alpha) + 'gamma' + str(gamma) + 'base' + str(base) + 'weight' + str(weight) + 'iteration:' + iteration
    hdplda = dual_hdplda.HDPLDA(weight, alpha, gamma, base, docs1, docs2, len(voca1), len(voca2))
    hdplda_learning(hdplda, iteration, voca1, voca2, corpuswrite1, corpuswrite2)
