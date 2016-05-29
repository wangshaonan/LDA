#!/usr/bin/env python
# -*- coding: utf-8 -*-


import numpy

class LDA:
    def __init__(self, K, alpha, beta, docs, V, smartinit=True):
        self.K = K
        self.alpha = alpha # parameter of topics prior
        self.beta = beta   # parameter of words prior
        self.docs = docs
        self.V = V

        self.z_m_n = [] # topics of words of documents
        self.n_m_z = numpy.zeros((len(self.docs), K)) + alpha     # word count of each document and topic
        self.n_z_t = numpy.zeros((K, V)) + beta # word count of each topic and vocabulary
        self.n_z = numpy.zeros(K) + V * beta    # word count of each topic

        self.N = 0
        for m, doc in enumerate(docs):
            self.N += len(doc)
            z_n = []
            for t in doc:
                if smartinit:
                    p_z = self.n_z_t[:, t] * self.n_m_z[m] / self.n_z
                    z = numpy.random.multinomial(1, p_z / p_z.sum()).argmax()
                else:
                    z = numpy.random.randint(0, K)
                z_n.append(z)
                self.n_m_z[m, z] += 1
                self.n_z_t[z, t] += 1
                self.n_z[z] += 1
            self.z_m_n.append(numpy.array(z_n))

    def inference(self):
        """learning once iteration"""
        for m, doc in enumerate(self.docs):
            z_n = self.z_m_n[m]
            n_m_z = self.n_m_z[m]
            for n, t in enumerate(doc):
                # discount for n-th word t with topic z
                z = z_n[n]
                n_m_z[z] -= 1
                self.n_z_t[z, t] -= 1
                self.n_z[z] -= 1

                # sampling topic new_z for t
                p_z = self.n_z_t[:, t] * n_m_z / self.n_z
                new_z = numpy.random.multinomial(1, p_z / p_z.sum()).argmax()

                # set z the new topic and increment counters
                z_n[n] = new_z
                n_m_z[new_z] += 1
                self.n_z_t[new_z, t] += 1
                self.n_z[new_z] += 1

    def worddist(self):
        """get topic-word distribution"""
        return self.n_z_t / self.n_z[:, numpy.newaxis]

    def perplexity(self, docs=None):
        if docs == None: docs = self.docs
        phi = self.worddist()
        log_per = 0
        N = 0
        Kalpha = self.K * self.alpha
        for m, doc in enumerate(docs):
            theta = self.n_m_z[m] / (len(self.docs[m]) + Kalpha)
            for w in doc:
                log_per -= numpy.log(numpy.inner(phi[:,w], theta))
            N += len(doc)
        return numpy.exp(log_per / N)

def lda_learning(lda, iteration, voca, corpuswrite1, corpuswrite2):
    pre_perp = lda.perplexity()
    print "initial perplexity=%f" % pre_perp
    for i in range(iteration):
        lda.inference()
        perp = lda.perplexity()
        print "-%d p=%f" % (i + 1, perp)
        if pre_perp:
            if pre_perp < perp:
#                output_word_topic_dist(lda, voca, corpuswrite1)
#                output_topic_doc_dist(lda, corpuswrite2)
                pre_perp = None
            else:
                pre_perp = perp
    output_word_topic_dist(lda, voca, corpuswrite1)
    output_topic_doc_dist(lda, corpuswrite2)

def output_word_topic_dist(lda, voca, corpuswrite1):
    phi = lda.worddist()
    for k in xrange(lda.K):
        corpuswrite1.write('topic'+ str(k) + '\n')
        for w in numpy.argsort(-phi[k])[:20]:
            corpuswrite1.write(str(voca[w])+' : '+str(phi[k,w]) +' ')
        corpuswrite1.write('\n')

def output_topic_doc_dist(lda, corpuswrite2):
    theta = lda.n_m_z
#    print type(theta)
#    print theta
    for doc in xrange(0, len(theta)):
#        corpuswrite2.write('##'+str(doc)+'## ')
        for ind in xrange(len(theta[doc])):
            corpuswrite2.write(str(theta[doc][ind]) + ' ')
        corpuswrite2.write('\n')
