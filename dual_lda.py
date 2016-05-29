#!/usr/bin/env python
# -*- coding: utf-8 -*-


import numpy

class LDA:
    def __init__(self, K, weight, alpha, beta1, beta2, docs1, docs2, V1, V2):
        self.K = K
        self.weight = weight
        self.alpha = alpha # parameter of topics prior
        self.beta1 = beta1   # parameter of words prior
        self.beta2 = beta2
        self.docs1 = docs1
        self.docs2 = docs2
        self.V1 = V1
        self.V2 = V2

        self.z_m_n1 = [] # topics of words of documents
        self.z_m_n2 = []
        self.n_m_z = numpy.zeros((len(self.docs1), K)) + alpha     # word count of each document and topic
        self.n_z_t1 = numpy.zeros((K, V1)) + beta1 # word count of each topic and vocabulary
        self.n_z1 = numpy.zeros(K) + V1 * beta1    # word count of each topic
        self.n_z_t2 = numpy.zeros((K, V2)) + beta2
        self.n_z2 = numpy.zeros(K) + V2 * beta2

        for m in range(len(self.docs1)):
            z_n1 = []
            z_n2 = []
            doc1 = self.docs1[m]
            doc2 = self.docs2[m]
            for n in range(len(doc1)+len(doc2)):
                if n < len(doc1):
                    t = doc1[n]
                    p_z = self.n_z_t1[:, t] * self.n_m_z[m] / self.n_z1
                    z = numpy.random.multinomial(1, p_z / p_z.sum()).argmax()
                    z_n1.append(z)
                    self.n_m_z[m, z] += self.weight[0]
                    self.n_z_t1[z, t] += 1
                    self.n_z1[z] += 1

                else:
                    nn = n - len(doc1)
                    t = doc2[nn]
                    p_z = self.n_z_t2[:, t] * self.n_m_z[m] / self.n_z2
                    z = numpy.random.multinomial(1, p_z / p_z.sum()).argmax()
                    z_n2.append(z)
                    self.n_m_z[m, z] += self.weight[1]
                    self.n_z_t2[z, t] += 1
                    self.n_z2[z] += 1
            self.z_m_n1.append(numpy.array(z_n1))
            self.z_m_n2.append(numpy.array(z_n2))

    def easymultinominal(self, p_z):
        psum = sum(p_z)
        u = numpy.random.rand()  #return random samples from a uniform distribution over [0, 1)
        u *= psum
        psum = 0
        for kk in range(0, self.K):
            psum += p_z[kk]
            if u <= psum:
                break
        return kk

    def inference(self):
        """learning once iteration"""
        for m in range(len(self.docs1)):
            doc1 = self.docs1[m]
            doc2 = self.docs2[m]
            z_n1 = self.z_m_n1[m]
            z_n2 = self.z_m_n2[m]
            n_m_z = self.n_m_z[m]
            for n in range(len(doc1)+len(doc2)):
                # discount for n-th word t with topic z
                if n < len(doc1):
                    z = z_n1[n]
                    t = doc1[n]
                    n_m_z[z] -= self.weight[0]
                    self.n_z_t1[z, t] -= 1
                    self.n_z1[z] -= 1
                    # sampling topic new_z for t
                    p_z = self.n_z_t1[:, t] * n_m_z / self.n_z1
#                    new_z = numpy.random.multinomial(1, p_z / p_z.sum()).argmax()
                    #likelihood of new component
                    new_z = self.easymultinominal(p_z)

                    n_m_z[new_z] += self.weight[0]
                    self.n_z_t1[new_z, t] += 1
                    self.n_z1[new_z] += 1
                    # set z the new topic and increment counters
                    z_n1[n] = new_z
                else:
                    nn = n - len(doc1)
                    z = z_n2[nn]
                    t = doc2[nn]
                    n_m_z[z] -= self.weight[1]
                    self.n_z_t2[z, t] -= 1
                    self.n_z2[z] -= 1
                    # sampling topic new_z for t
                    p_z = self.n_z_t2[:, t] * n_m_z / self.n_z2
#                    new_z = numpy.random.multinomial(1, p_z / p_z.sum()).argmax()
                    new_z = self.easymultinominal(p_z)

                    n_m_z[new_z] += self.weight[1]
                    self.n_z_t2[new_z, t] += 1
                    self.n_z2[new_z] += 1
                    # set z the new topic and increment counters
                    z_n2[nn] = new_z


    def worddist1(self):
        """get topic-word distribution"""
        return self.n_z_t1 / self.n_z1[:, numpy.newaxis]

    def worddist2(self):
        return self.n_z_t2/self.n_z2[:, numpy.newaxis]

    def perplexity(self):
#        if docs == None: docs = self.docs
        phi1 = self.worddist1()
        phi2 = self.worddist2()
        log_per = 0
        N = 0
        Kalpha = self.K * self.alpha
        for m in range(len(self.docs1)):
            doc1 = self.docs1[m]
            doc2 = self.docs2[m]
            theta = self.n_m_z[m] / (len(self.docs1[m]) + Kalpha)
            for n in range(len(doc1)+len(doc2)):
                if n < len(doc1):
                    t = doc1[n]
                    log_per -= numpy.log(numpy.inner(phi1[:,t], theta))
                    N += len(doc1)
                else:
                    nn = n - len(doc1)
                    t = doc2[nn]
                    log_per -= numpy.log(numpy.inner(phi2[:,t], theta))
                    N += len(doc2)
        return numpy.exp(log_per / N)

def lda_learning(lda, iteration, voca1, voca2, corpuswrite1, corpuswrite2):
    pre_perp = lda.perplexity()
    print "initial perplexity=%f" % pre_perp
    for i in range(iteration):
        lda.inference()
        perp = lda.perplexity()
        print "-%d p=%f" % (i + 1, perp)
        if pre_perp:
            if pre_perp < perp:
#                output_word_topic_dist(lda, voca1, voca2)
#                output_topic_doc_dist(lda)
                pre_perp = None
            else:
                pre_perp = perp
    output_word_topic_dist(lda, voca1, voca2, corpuswrite1)
    output_topic_doc_dist(lda, corpuswrite2)

def output_word_topic_dist(lda, voca1, voca2, corpuswrite1):
    phi1 = lda.worddist1()
    phi2 = lda.worddist2()
    for k in xrange(lda.K):
        corpuswrite1.write('topic'+ str(k) + '\n')
        for w in numpy.argsort(-phi1[k])[:20]:
            corpuswrite1.write(str(voca1[w])+' : '+str(phi1[k,w]) +' ')
        corpuswrite1.write('\n')
    for k in xrange(lda.K):
        corpuswrite1.write('topic'+ str(k) + '\n')
        for w in numpy.argsort(-phi2[k])[:20]:
            corpuswrite1.write(str(voca2[w])+' : '+str(phi2[k,w]) +' ')
        corpuswrite1.write('\n')

def output_topic_doc_dist(lda, corpuswrite2):
    theta = lda.n_m_z
    for doc in xrange(0, len(theta)):
#        corpuswrite2.write('##'+str(doc)+'## ')
        for ind in xrange(len(theta[doc])):
            corpuswrite2.write(str(theta[doc][ind])+' ')
        corpuswrite2.write('\n')
