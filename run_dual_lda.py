#!/usr/bin/env python
# -*- coding: utf-8 -*-
import dual_lda
import sys
import os
from dual_lda import lda_learning
import re

print 'usage: python run_dual_lda.py dir_sense dir_corpusread1(topic) dir_corpusread2(local) dir_result'

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

    K = 2
    weight = [1, 1]
    alpha = 0.05   # bigger alpha = smoother theta
    beta1 = 0.05
    beta2 = 0.01   # bigger beta = smoother phi
    iteration = 2000
    print('alpha: %s  beta1: %s  beta2: %s  weigth: [%f %f]  iteration: %d' % (alpha, beta1, beta2, weight[0], weight[1], iteration))
#    print 'alpha:' + str(alpha) + 'beta1:' + str(beta1) + 'beta2' + str(beta2) + + 'weight' + str(weight) + 'iteration:' + str(iteration)
    lda = dual_lda.LDA(K, weight, alpha, beta1, beta2, docs1, docs2, len(voca1), len(voca2))
    lda_learning(lda, iteration, voca1, voca2, corpuswrite1, corpuswrite2)
