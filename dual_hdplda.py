#!/usr/bin/env python
# -*- coding: utf-8 -*-


import numpy

class HDPLDA:
    def __init__(self, weight, alpha, gamma, base, docs1, docs2, V1, V2):
        self.weight = weight
        self.alpha = alpha
        self.base = base
        self.gamma = gamma
        self.V1 = V1
        self.V2 = V2

        self.x_ji1 = docs1 # vocabulary for each document and term
        self.t_ji1 = [numpy.zeros(len(x_i), dtype=int) - 1 for x_i in docs1] # table for each document and term (without assignment)
        self.k_jt1 = [[] for x_i in docs1] # topic for each document and table
        self.n_jt1 = [numpy.ndarray(0,dtype=int) for x_i in docs1] # number of terms for each document and table

        self.tables1 = [[] for x_i in docs1] # available id of tables for each document
        self.n_tables1 = 0

        self.x_ji2 = docs2 # vocabulary for each document and term
        self.t_ji2 = [numpy.zeros(len(x_i), dtype=int) - 1 for x_i in docs2] # table for each document and term (without assignment)
        self.k_jt2 = [[] for x_i in docs2] # topic for each document and table
        self.n_jt2 = [numpy.ndarray(0,dtype=int) for x_i in docs2] # number of terms for each document and table

        self.tables2 = [[] for x_i in docs2] # available id of tables for each document
        self.n_tables2 = 0

        # shared varibles m_k ,n_k, topics
        #self.m_k = numpy.ndarray(0,dtype=int)  # number of tables for each topic
        self.m_k1 = numpy.ndarray(0,dtype=int)  # number of tables for each topic
        self.m_k2 = numpy.ndarray(0,dtype=int)  # number of tables for each topic
        self.m_k = numpy.ndarray(0,dtype=float)

        self.n_kv1 = numpy.ndarray((0, V1),dtype=int) # number of terms for each topic and vocabulary
        self.n_k1 = numpy.ndarray(0,dtype=int)  # number of terms for each topic / sum of n_kv[k][:]
        self.n_kv2 = numpy.ndarray((0, V2),dtype=int) # number of terms for each topic and vocabulary
        self.n_k2 = numpy.ndarray(0,dtype=int)  # number of terms for each topic / sum of n_kv[k][:]

#        self.n_k = numpy.ndarray(0,dtype=int)

        self.topics = [] # available id of topics

        # memoization
        self.updated_n_tables1()
        self.Vbase1 = V1 * base
        self.gamma_f_k_new_x_ji1 = gamma / V1
        self.cur_log_base_cache1 = [0]
        self.cur_log_V_base_cache1 = [0]

        self.updated_n_tables2()
        self.Vbase2 = V2 * base
        self.gamma_f_k_new_x_ji2 = gamma / V2
        self.cur_log_base_cache2 = [0]
        self.cur_log_V_base_cache2 = [0]

    def inference(self):
        for j in range(len(self.x_ji1)):
            x_i1 = self.x_ji1[j]
            x_i2 = self.x_ji2[j]
            for i in range(len(x_i1)):
                self.sampling_table1(j, i)
            for i in range(len(x_i2)):
                self.sampling_table2(j, i)
            for t in self.tables1[j]:
                self.sampling_k1(j, t)
            for t in self.tables2[j]:
                self.sampling_k2(j, t)

    def worddist1(self):
        return [(self.n_kv1[k] + self.base) / (self.n_k1[k] + self.Vbase1) for k in self.topics]
    def worddist2(self):
        return [(self.n_kv2[k] + self.base) / (self.n_k2[k] + self.Vbase2) for k in self.topics]

    def perplexity(self):
        phi1 = self.worddist1()
        phi2 = self.worddist2()
        phi1.append(numpy.zeros(self.V1) + 1.0 / self.V1)
        phi2.append(numpy.zeros(self.V2) + 1.0 / self.V2)
        log_per = 0
        N = 0
        gamma_over_T_gamma1 = self.gamma / (self.n_tables1 + self.gamma)
        gamma_over_T_gamma2 = self.gamma / (self.n_tables2 + self.gamma)
        for j in range(len(self.x_ji1)):
            x_i1 = self.x_ji1[j]
            x_i2 = self.x_ji2[j]
            p_k = numpy.zeros(self.m_k.size)    # topic dist for document 
            for t in self.tables1[j]:
                k = self.k_jt1[j][t]
                p_k[k] += self.n_jt1[j][t]       # n_jk
            len_x_alpha = len(x_i1) + self.alpha
            p_k /= len_x_alpha
            
            p_k_parent = self.alpha / len_x_alpha
            p_k += p_k_parent * (self.m_k1 / (self.n_tables1 + self.gamma))
            
            theta = [p_k[k] for k in self.topics]
            theta.append(p_k_parent * gamma_over_T_gamma1)

            for v in x_i1:
                log_per -= numpy.log(numpy.inner([p[v] for p in phi1], theta))
            N += len(x_i1)

            for t in self.tables2[j]:
                k = self.k_jt2[j][t]
                p_k[k] += self.n_jt2[j][t]       # n_jk
            len_x_alpha = len(x_i2) + self.alpha
            p_k /= len_x_alpha

            p_k_parent = self.alpha / len_x_alpha
            p_k += p_k_parent * (self.m_k2 / (self.n_tables2 + self.gamma))

            theta = [p_k[k] for k in self.topics]
            theta.append(p_k_parent * gamma_over_T_gamma2)

            for v in x_i2:
                log_per -= numpy.log(numpy.inner([p[v] for p in phi2], theta))
            N += len(x_i2)

        return numpy.exp(log_per / N)

    def dump(self, disp_x=False):
        if disp_x: print "x_ji:", self.x_ji1
        print "t_ji1:", self.t_ji1
        print "k_jt1:", self.k_jt1
        print "n_kv1:", self.n_kv1
        print "n_kv2:", self.n_kv2
        print "n_jt1:", self.n_jt1
        print "tables1:", self.tables1
        print "t_ji2:", self.t_ji2
        print "k_jt2:", self.k_jt2
        print "n_jt2:", self.n_jt2
        print "tables2:", self.tables2
        print "n_k1:", self.n_k1
        print "n_k2:", self.n_k2
        print "m_k1:", self.m_k1
        print "m_k2:", self.m_k2
        print "m_k:", self.m_k
        print "topics:", self.topics


    # internal methods from here

    # cache for faster calcuration
    def updated_n_tables1(self):
        self.alpha_over_T_gamma1 = self.alpha / (self.n_tables1 + self.gamma)

    def updated_n_tables2(self):
        self.alpha_over_T_gamma2 = self.alpha / (self.n_tables2 + self.gamma)

    def cur_log_base1(self, n):
        """cache of \sum_{i=0}^{n-1} numpy.log(i + self.base)"""
        N = len(self.cur_log_base_cache1)
        if n < N: return self.cur_log_base_cache1[n]
        s = self.cur_log_base_cache1[-1]
        while N <= n:
            s += numpy.log(N + self.base - 1)
            self.cur_log_base_cache1.append(s)
            N += 1
        return s

    def cur_log_base2(self, n):
        """cache of \sum_{i=0}^{n-1} numpy.log(i + self.base)"""
        N = len(self.cur_log_base_cache2)
        if n < N: return self.cur_log_base_cache2[n]
        s = self.cur_log_base_cache2[-1]
        while N <= n:
            s += numpy.log(N + self.base - 1)
            self.cur_log_base_cache2.append(s)
            N += 1
        return s

    def cur_log_V_base1(self, n):
        """cache of \sum_{i=0}^{n-1} numpy.log(i + self.base * self.V)"""
        N = len(self.cur_log_V_base_cache1)
        if n < N: return self.cur_log_V_base_cache1[n]
        s = self.cur_log_V_base_cache1[-1]
        while N <= n:
            s += numpy.log(N + self.Vbase1 - 1)
            self.cur_log_V_base_cache1.append(s)
            N += 1
        return s

    def cur_log_V_base2(self, n):
        """cache of \sum_{i=0}^{n-1} numpy.log(i + self.base * self.V)"""
        N = len(self.cur_log_V_base_cache2)
        if n < N: return self.cur_log_V_base_cache2[n]
        s = self.cur_log_V_base_cache2[-1]
        while N <= n:
            s += numpy.log(N + self.Vbase2 - 1)
            self.cur_log_V_base_cache2.append(s)
            N += 1
        return s

    def log_f_k_new_x_jt1(self, n_jt, n_tv, n_kv = None, n_k = 0):
        p = self.cur_log_V_base1(n_k) - self.cur_log_V_base1(n_k + n_jt)
        for (v_l, n_l) in n_tv:
            n0 = n_kv[v_l] if n_kv != None else 0
            p += self.cur_log_base1(n0 + n_l) - self.cur_log_base1(n0)
        return p

    def log_f_k_new_x_jt2(self, n_jt, n_tv, n_kv = None, n_k = 0):
        p = self.cur_log_V_base2(n_k) - self.cur_log_V_base2(n_k + n_jt)
        for (v_l, n_l) in n_tv:
            n0 = n_kv[v_l] if n_kv != None else 0
            p += self.cur_log_base2(n0 + n_l) - self.cur_log_base2(n0)
        return p

    def count_n_jtv1(self, j, t, k_old):
        """count n_jtv and decrease n_kv for k_old"""
        x_i = self.x_ji1[j]
        t_i = self.t_ji1[j]
        n_jtv = dict()
        for i, t1 in enumerate(t_i):
            if t1 == t:
                v = x_i[i]
                self.n_kv1[k_old, v] -= 1
                if v in n_jtv:
                    n_jtv[v] += 1
                else:
                    n_jtv[v] = 1
        return n_jtv.items()

    def count_n_jtv2(self, j, t, k_old):
        """count n_jtv and decrease n_kv for k_old"""
        x_i = self.x_ji2[j]
        t_i = self.t_ji2[j]
        n_jtv = dict()
        for i, t1 in enumerate(t_i):
            if t1 == t:
                v = x_i[i]
                self.n_kv2[k_old, v] -= 1
                if v in n_jtv:
                    n_jtv[v] += 1
                else:
                    n_jtv[v] = 1
        return n_jtv.items()

    # sampling t (table) from posterior
    def sampling_table1(self, j, i):
        v = self.x_ji1[j][i]
        tables = self.tables1[j]
        t_old = self.t_ji1[j][i]
        if t_old >=0:
            k_old = self.k_jt1[j][t_old]

            # decrease counters
            self.n_kv1[k_old, v] -= 1
            self.n_k1[k_old] -= 1
            self.n_jt1[j][t_old] -= 1

            if self.n_jt1[j][t_old]==0:
                # table that all guests are gone
                tables.remove(t_old)
                self.m_k1[k_old] -= 1
                self.m_k[k_old] -= self.weight[0]
                if self.m_k[k_old] < 0.00001:
                    self.m_k[k_old] = 0
                self.n_tables1 -= 1
                self.updated_n_tables1()

                if self.m_k[k_old] == 0:
                    # topic (dish) that all guests are gone
                    self.topics.remove(k_old)
        # sampling from posterior p(t_ji=t)
        t_new = self.sampling_t1(j, i, v, tables)

        # increase counters
        self.t_ji1[j][i] = t_new
        self.n_jt1[j][t_new] += 1

        k_new = self.k_jt1[j][t_new]
        self.n_k1[k_new] += 1
        self.n_kv1[k_new, v] += 1

    def sampling_table2(self,j,i):
        v = self.x_ji2[j][i]
        tables = self.tables2[j]
        t_old = self.t_ji2[j][i]
        if t_old >=0:
            k_old = self.k_jt2[j][t_old]

            # decrease counters
            self.n_kv2[k_old, v] -= 1
            self.n_k2[k_old] -= 1
            self.n_jt2[j][t_old] -= 1

            if self.n_jt2[j][t_old]==0:
                # table that all guests are gone
                tables.remove(t_old)
                self.m_k2[k_old] -= 1
                self.m_k[k_old] -= self.weight[1]
                if self.m_k[k_old] < 0.00001:
                    self.m_k[k_old] =0
                self.n_tables2 -= 1
                self.updated_n_tables2()

                if self.m_k[k_old] == 0:
                    # topic (dish) that all guests are gone
                    self.topics.remove(k_old)

        # sampling from posterior p(t_ji=t)
        t_new = self.sampling_t2(j, i, v, tables)

        # increase counters
        self.t_ji2[j][i] = t_new
        self.n_jt2[j][t_new] += 1

        k_new = self.k_jt2[j][t_new]
        self.n_k2[k_new] += 1
        self.n_kv2[k_new, v] += 1

    def sampling_t1(self, j, i, v, tables):
        f_k = (self.n_kv1[:, v] + self.base) / (self.n_k1 + self.Vbase1)
        p_t = [self.n_jt1[j][t] * f_k[self.k_jt1[j][t]] for t in tables]
        p_x_ji = numpy.inner(self.m_k1, f_k) + self.gamma_f_k_new_x_ji1
        p_t.append(p_x_ji * self.alpha_over_T_gamma1)

        p_t = numpy.array(p_t, copy=False)
        p_t /= p_t.sum()
#        drawing = numpy.random.multinomial(1, p_t).argmax()
        drawing = self.easymultinominal(p_t)
        if drawing < len(tables):
            return tables[drawing]
        else:
            return self.new_table1(j, i, f_k)

    def sampling_t2(self, j, i, v, tables):
        f_k = (self.n_kv2[:, v] + self.base) / (self.n_k2 + self.Vbase2)
        p_t = [self.n_jt2[j][t] * f_k[self.k_jt2[j][t]] for t in tables]
        p_x_ji = numpy.inner(self.m_k2, f_k) + self.gamma_f_k_new_x_ji2
        p_t.append(p_x_ji * self.alpha_over_T_gamma2)

        p_t = numpy.array(p_t, copy=False)
        p_t /= p_t.sum()
#        drawing = numpy.random.multinomial(1, p_t).argmax()
        drawing = self.easymultinominal(p_t)
        if drawing < len(tables):
            return tables[drawing]
        else:
            return self.new_table2(j, i, f_k)

    # Assign guest x_ji to a new table and draw topic (dish) of the table
    def new_table1(self, j, i, f_k):
        # search a spare table ID
        T_j = self.n_jt1[j].size
        for t_new in range(T_j):
            if t_new not in self.tables1[j]: break
        else:
            # new table ID (no spare)
            t_new = T_j
            self.n_jt1[j].resize(t_new+1)
            self.n_jt1[j][t_new] = 0
            self.k_jt1[j].append(0)
        self.tables1[j].append(t_new)
        self.n_tables1 += 1
        self.updated_n_tables1()

        # sampling of k for new topic(= dish of new table)
        p_k = [self.m_k[k] * f_k[k] for k in self.topics]
        p_k.append(self.gamma_f_k_new_x_ji1)
        k_new = self.sampling_topic1(numpy.array(p_k, copy=False))

        self.k_jt1[j][t_new] = k_new
        self.m_k1[k_new] += 1
        self.m_k[k_new] += self.weight[0]
        return t_new

    def new_table2(self, j, i, f_k):
        # search a spare table ID
        T_j = self.n_jt2[j].size
        for t_new in range(T_j):
            if t_new not in self.tables2[j]: break
        else:
            # new table ID (no spare)
            t_new = T_j
            self.n_jt2[j].resize(t_new+1)
            self.n_jt2[j][t_new] = 0
            self.k_jt2[j].append(0)
        self.tables2[j].append(t_new)
        self.n_tables2 += 1
        self.updated_n_tables2()

        # sampling of k for new topic(= dish of new table)
        p_k = [self.m_k[k] * f_k[k] for k in self.topics]
        p_k.append(self.gamma_f_k_new_x_ji2)
        k_new = self.sampling_topic2(numpy.array(p_k, copy=False))

        self.k_jt2[j][t_new] = k_new
        self.m_k2[k_new] += 1
        self.m_k[k_new] += self.weight[1]
        return t_new

    def easymultinominal(self, p_z):
        psum = sum(p_z)
        u = numpy.random.rand()  #return random samples from a uniform distribution over [0, 1)
        u *= psum
        psum = 0
        for kk in range(0, len(p_z)):
            psum += p_z[kk]
            if u <= psum:
                break
        return kk

    # sampling topic
    # In the case of new topic, allocate resource for parameters
    def sampling_topic1(self, p_k):
#        drawing = numpy.random.multinomial(1, p_k / p_k.sum()).argmax()
        drawing = self.easymultinominal(p_k)
        if drawing < len(self.topics):
            # existing topic
            k_new = self.topics[drawing]
        else:
            # new topic
            K = self.m_k.size
            for k_new in range(K):
                # recycle table ID, if a spare ID exists
                if k_new not in self.topics: break
            else:
                # new table ID, if otherwise
                k_new = K
                self.n_k1 = numpy.resize(self.n_k1, k_new + 1)
                self.n_k2 = numpy.resize(self.n_k2, k_new + 1)  #
                self.n_k1[k_new] = 0
                self.n_k2[k_new] = 0
                self.m_k = numpy.resize(self.m_k, k_new + 1)
                self.m_k1 = numpy.resize(self.m_k1, k_new + 1)
                self.m_k2 = numpy.resize(self.m_k2, k_new + 1)
                self.m_k[k_new] = 0
                self.m_k1[k_new] = 0
                self.m_k2[k_new] = 0
                self.n_kv1 = numpy.resize(self.n_kv1, (k_new+1, self.V1))
                self.n_kv2 = numpy.resize(self.n_kv2, (k_new+1, self.V2))  #
                self.n_kv1[k_new, :] = numpy.zeros(self.V1, dtype=int)
                self.n_kv2[k_new, :] = numpy.zeros(self.V2, dtype=int)
            self.topics.append(k_new)
        return k_new

    def sampling_topic2(self, p_k):
#        drawing = numpy.random.multinomial(1, p_k / p_k.sum()).argmax()
        drawing = self.easymultinominal(p_k)
        if drawing < len(self.topics):
            # existing topic
            k_new = self.topics[drawing]
        else:
            # new topic
            K = self.m_k.size
            for k_new in range(K):
                # recycle table ID, if a spare ID exists
                if k_new not in self.topics: break
            else:
                # new table ID, if otherwise
                k_new = K
                self.n_k2 = numpy.resize(self.n_k2, k_new + 1)
                self.n_k1 = numpy.resize(self.n_k1, k_new + 1)   #
                self.n_k2[k_new] = 0
                self.n_k1[k_new] = 0
                self.m_k = numpy.resize(self.m_k, k_new + 1)
                self.m_k[k_new] = 0
                self.m_k1 = numpy.resize(self.m_k1, k_new + 1)
                self.m_k1[k_new] = 0
                self.m_k2 = numpy.resize(self.m_k2, k_new + 1)
                self.m_k2[k_new] = 0
                self.n_kv2 = numpy.resize(self.n_kv2, (k_new+1, self.V2))
                self.n_kv1 = numpy.resize(self.n_kv1, (k_new+1, self.V1))  #
                self.n_kv2[k_new, :] = numpy.zeros(self.V2, dtype=int)
                self.n_kv1[k_new, :] = numpy.zeros(self.V1, dtype=int)
            self.topics.append(k_new)
        return k_new

    def sampling_k1(self, j, t):
        """sampling k (dish=topic) from posterior"""
        k_old = self.k_jt1[j][t]
        n_jt = self.n_jt1[j][t]
        self.m_k1[k_old] -= 1
        self.m_k[k_old] -= self.weight[0]
        if self.m_k[k_old] < 0.00001:
            self.m_k[k_old] = 0
        self.n_k1[k_old] -= n_jt
        if self.m_k[k_old] == 0:
            self.topics.remove(k_old)

        # sampling of k
        n_jtv = self.count_n_jtv1(j, t, k_old) # decrement n_kv also in this method
        K = len(self.topics)
        log_p_k = numpy.zeros(K+1)
        for i, k in enumerate(self.topics):
            log_p_k[i] = self.log_f_k_new_x_jt1(n_jt, n_jtv, self.n_kv1[k, :], self.n_k1[k]) + numpy.log(self.m_k[k])
        log_p_k[K] = self.log_f_k_new_x_jt1(n_jt, n_jtv) + numpy.log(self.gamma)
        k_new = self.sampling_topic1(numpy.exp(log_p_k - log_p_k.max())) # for too small

        # update counters
        self.k_jt1[j][t] = k_new
        self.m_k1[k_new] += 1
        self.m_k[k_new] += self.weight[0]
        self.n_k1[k_new] += self.n_jt1[j][t]
        for v, t1 in zip(self.x_ji1[j], self.t_ji1[j]):
            if t1 != t: continue
            self.n_kv1[k_new, v] += 1

    def sampling_k2(self, j, t):
        """sampling k (dish=topic) from posterior"""
        k_old = self.k_jt2[j][t]
        n_jt = self.n_jt2[j][t]
        self.m_k2[k_old] -= 1
        self.m_k[k_old] -= self.weight[1]
        if self.m_k[k_old] < 0.00001:
            self.m_k[k_old] = 0
        self.n_k2[k_old] -= n_jt
        if self.m_k[k_old] == 0:
            self.topics.remove(k_old)

        # sampling of k
        n_jtv = self.count_n_jtv2(j, t, k_old) # decrement n_kv also in this method
        K = len(self.topics)
        log_p_k = numpy.zeros(K+1)
        for i, k in enumerate(self.topics):
            log_p_k[i] = self.log_f_k_new_x_jt2(n_jt, n_jtv, self.n_kv2[k, :], self.n_k2[k]) + numpy.log(self.m_k[k])
        log_p_k[K] = self.log_f_k_new_x_jt2(n_jt, n_jtv) + numpy.log(self.gamma)
        k_new = self.sampling_topic2(numpy.exp(log_p_k - log_p_k.max())) # for too small

        # update counters
        self.k_jt2[j][t] = k_new
        self.m_k2[k_new] += 1
        self.m_k[k_new] += self.weight[1]
        self.n_k2[k_new] += self.n_jt2[j][t]
        for v, t1 in zip(self.x_ji2[j], self.t_ji2[j]):
            if t1 != t: continue
            self.n_kv2[k_new, v] += 1


def hdplda_learning(hdplda, iteration, voca1, voca2, corpuswrite1, corpuswrite2):
    for i in range(iteration):
        hdplda.inference()
        print "-%d K=%d p=%f" % (i + 1, len(hdplda.topics), hdplda.perplexity())
    output_word_topic_dist(hdplda, voca1, voca2, corpuswrite1)
    output_topic_doc_dist(hdplda, corpuswrite2)
    return hdplda

def output_word_topic_dist(hdplda, voca1, voca2, corpuswrite1):
    phi1 = hdplda.worddist1()
    phi2 = hdplda.worddist2()
    for k,v in enumerate(hdplda.topics):
    	corpuswrite1.write('topic'+ str(v) + '\n')
        for w in numpy.argsort(-phi1[k])[:20]:
            corpuswrite1.write(str(voca1[w])+' : '+str(phi1[k][w]) +' ')
        corpuswrite1.write('\n')
    for k,v in enumerate(hdplda.topics):
        corpuswrite1.write('topic'+ str(v) + '\n')
        for w in numpy.argsort(-phi2[k])[:20]:
            corpuswrite1.write(str(voca2[w])+' : '+str(phi2[k][w]) +' ')
        corpuswrite1.write('\n')

def output_topic_doc_dist(hdplda, corpuswrite2):
    theta = []
    for j in range(len(hdplda.x_ji1)):
        p_k = numpy.zeros(max(hdplda.topics)+1)
        for t in hdplda.tables1[j]:
            k = hdplda.k_jt1[j][t]
            p_k[k] += hdplda.weight[0]*hdplda.n_jt1[j][t]
        for t in hdplda.tables2[j]:
            k = hdplda.k_jt2[j][t]
            p_k[k] += hdplda.weight[1]*hdplda.n_jt2[j][t]
        theta.append(p_k)
    for doc in xrange(0, len(theta)):
#        corpuswrite2.write('##'+str(doc)+'## ')
        for ind in xrange(len(theta[doc])):
            corpuswrite2.write(str(theta[doc][ind]) + ' ')
        corpuswrite2.write('\n')
