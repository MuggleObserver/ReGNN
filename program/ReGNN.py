# -*- coding: utf-8 -*-

import math
import os

import tensorflow as tf

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


class Model(object):
    def __init__(self, hidden_size=100, out_size=100, batch_size=100, nonhybrid=True):
        self.hidden_size = hidden_size
        self.out_size = out_size
        self.batch_size = batch_size
        self.mask = tf.placeholder(dtype=tf.float32)
        self.mask_r = tf.placeholder(dtype=tf.float32)
        self.mask_e = tf.placeholder(dtype=tf.float32)
        self.alias = tf.placeholder(dtype=tf.int32)
        self.item = tf.placeholder(dtype=tf.int32)
        self.tar = tf.placeholder(dtype=tf.int32)
        self.his_list = tf.placeholder(dtype=tf.int32)
        self.nonhybrid = nonhybrid
        self.stdv = 1.0 / math.sqrt(self.hidden_size)

        self.nasr_w1 = tf.get_variable('nasr_w1', [self.out_size, self.out_size], dtype=tf.float32,
                                       initializer=tf.random_uniform_initializer(-self.stdv, self.stdv))
        self.nasr_w2 = tf.get_variable('nasr_w2', [self.out_size, self.out_size], dtype=tf.float32,
                                       initializer=tf.random_uniform_initializer(-self.stdv, self.stdv))

        self.nasr_w1_r = tf.get_variable('nasr_w1_r', [self.out_size, self.out_size], dtype=tf.float32,
                                         initializer=tf.random_uniform_initializer(-self.stdv, self.stdv))
        self.nasr_w2_r = tf.get_variable('nasr_w2_r', [self.out_size, self.out_size], dtype=tf.float32,
                                         initializer=tf.random_uniform_initializer(-self.stdv, self.stdv))

        self.nasr_w1_e = tf.get_variable('nasr_w1_e', [self.out_size, self.out_size], dtype=tf.float32,
                                         initializer=tf.random_uniform_initializer(-self.stdv, self.stdv))
        self.nasr_w2_e = tf.get_variable('nasr_w2_e', [self.out_size, self.out_size], dtype=tf.float32,
                                         initializer=tf.random_uniform_initializer(-self.stdv, self.stdv))

        self.nasr_v = tf.get_variable('nasrv', [1, self.out_size], dtype=tf.float32,
                                      initializer=tf.random_uniform_initializer(-self.stdv, self.stdv))
        self.nasr_v_r = tf.get_variable('nasrv_r', [1, self.out_size], dtype=tf.float32,
                                        initializer=tf.random_uniform_initializer(-self.stdv, self.stdv))
        self.nasr_v_e = tf.get_variable('nasrv_e', [1, self.out_size], dtype=tf.float32,
                                        initializer=tf.random_uniform_initializer(-self.stdv, self.stdv))

        self.nasr_b = tf.get_variable('nasr_b', [self.out_size], dtype=tf.float32, initializer=tf.zeros_initializer())
        self.nasr_b_r = tf.get_variable('nasr_b_r', [self.out_size], dtype=tf.float32,
                                        initializer=tf.zeros_initializer())
        self.nasr_b_e = tf.get_variable('nasr_b_e', [self.out_size], dtype=tf.float32,
                                        initializer=tf.zeros_initializer())

        self.w_re = tf.get_variable('w_re', [self.out_size, 2], dtype=tf.float32,
                                    initializer=tf.random_uniform_initializer(-self.stdv, self.stdv))
        self.output_r = tf.zeros([self.batch_size, 37483], dtype=tf.float32, name='out_r')

    def forward(self, re_embedding, train=True):
        rm = tf.reduce_sum(self.mask, 1)
        last_id = tf.gather_nd(self.alias, tf.stack([tf.range(self.batch_size), tf.to_int32(rm) - 1],
                                                    axis=1))
        last_h = tf.gather_nd(re_embedding, tf.stack([tf.range(self.batch_size), last_id],
                                                     axis=1))
        seq_h = tf.stack([tf.nn.embedding_lookup(re_embedding[i], self.alias[i]) for i in range(self.batch_size)],
                         axis=0)  # batch_size*T*d
        last = tf.matmul(last_h, self.nasr_w1)
        seq = tf.matmul(tf.reshape(seq_h, [-1, self.out_size]), self.nasr_w2)
        last = tf.reshape(last, [self.batch_size, 1, -1])
        m = tf.nn.sigmoid(last + tf.reshape(seq, [self.batch_size, -1, self.out_size]) + self.nasr_b)
        coef = tf.matmul(tf.reshape(m, [-1, self.out_size]), self.nasr_v, transpose_b=True) * tf.reshape(
            self.mask, [-1, 1])
        b = self.embedding[1:]
        # ======================================= add repeat mode ================================================
        last_r = tf.matmul(last_h, self.nasr_w1_r)
        seq_r = tf.matmul(tf.reshape(seq_h, [-1, self.out_size]), self.nasr_w2_r)
        last_r = tf.reshape(last_r, [self.batch_size, 1, -1])
        m_r = tf.nn.sigmoid(last_r + tf.reshape(seq_r, [self.batch_size, -1, self.out_size]) + self.nasr_b_r)
        coef_r = tf.matmul(tf.reshape(m_r, [-1, self.out_size]), self.nasr_v_r, transpose_b=True) * tf.reshape(
            self.mask, [-1, 1])
        ma_r = tf.concat([tf.reduce_sum(tf.reshape(coef_r, [self.batch_size, -1, 1]) * seq_h, 1),
                          tf.reshape(last, [-1, self.out_size])], -1)
        self.B_r = tf.get_variable('B_r', [2 * self.out_size, self.out_size],
                                   initializer=tf.random_uniform_initializer(-self.stdv, self.stdv))
        y1_r = tf.matmul(ma_r, self.B_r)

        logits_r = tf.matmul(y1_r, b, transpose_b=True)

        logits_r = logits_r * self.mask_r

        # ========================================== explore mode======================================================
        last_e = tf.matmul(last_h, self.nasr_w1_e)
        seq_e = tf.matmul(tf.reshape(seq_h, [-1, self.out_size]), self.nasr_w2_e)
        last_e = tf.reshape(last_e, [self.batch_size, 1, -1])
        m_e = tf.nn.sigmoid(last_e + tf.reshape(seq_e, [self.batch_size, -1, self.out_size]) + self.nasr_b_e)
        coef_e = tf.matmul(tf.reshape(m_e, [-1, self.out_size]), self.nasr_v_e, transpose_b=True) * tf.reshape(
            self.mask, [-1, 1])
        coef_e = tf.reshape(coef_e, [self.batch_size, -1])

        if not self.nonhybrid:
            ma = tf.concat([tf.reduce_sum(tf.reshape(coef, [self.batch_size, -1, 1]) * seq_h, 1),
                            tf.reshape(last, [-1, self.out_size])], -1)
            self.B = tf.get_variable('B', [2 * self.out_size, self.out_size],
                                     initializer=tf.random_uniform_initializer(-self.stdv, self.stdv))
            y1 = tf.matmul(ma, self.B)
            # =======================================add repeat================================================

            p_re = tf.nn.softmax(tf.matmul(y1, self.w_re))
            p_re = tf.transpose(p_re)
            pr = p_re[0]
            pe = p_re[1]

            pr = tf.reshape(pr, [-1, 1])
            pe = tf.reshape(pe, [-1, 1])

            ma_e = tf.concat([tf.reduce_sum(tf.reshape(coef_e, [self.batch_size, -1, 1]) * seq_h, 1),
                              tf.reshape(last, [-1, self.out_size])], -1)
            self.B_e = tf.get_variable('B_e', [2 * self.out_size, self.out_size],
                                       initializer=tf.random_uniform_initializer(-self.stdv, self.stdv))
            y1_e = tf.matmul(ma_e, self.B_e)
            logits_e = tf.matmul(y1_e, b, transpose_b=True)
            logits_e = logits_e * self.mask_e

            pr_logits = pr * logits_r
            pe_logits = pe * logits_e
            logits = pr_logits + pe_logits

        loss = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.tar - 1, logits=logits))
        self.vars = tf.trainable_variables()
        if train:
            lossL2 = tf.add_n([tf.nn.l2_loss(v) for v in self.vars if v.name not
                               in ['bias', 'gamma', 'b', 'g', 'beta']]) * self.L2
            loss = loss + lossL2
        return loss, logits



    def run(self, fetches, tar, item, adj_in, adj_out, alias, mask, mask_r, mask_e):
        return self.sess.run(fetches, feed_dict={self.tar: tar, self.item: item, self.adj_in: adj_in,
                                                 self.adj_out: adj_out, self.alias: alias, self.mask: mask,
                                                 self.mask_r: mask_r, self.mask_e: mask_e})

    def get_p_i_r(self, att_list, output_r,
                  his_list, ):
        for i in range(len(his_list) - 1):
            for j in range(1, len(his_list[0]) - 1):
                if his_list[i][j] != 0:
                    output_r[i][his_list[i][j] - 1] = att_list[i][
                        j - 1]
        return output_r

    def get_p_i_e(self, score_list, his_list):
        for i in range(len(his_list) - 1):
            for j in range(1, len(his_list[0]) - 1):
                if his_list[i][j] != 0:
                    score_list[i][his_list[i][j] - 1] = 0
        return score_list

    def get_paramters(self, tar, item, adj_in, adj_out, alias, mask):
        return tar, item, adj_in, adj_out, alias, mask


class ReGNN(Model):
    def __init__(self, hidden_size=100, out_size=100, batch_size=300, n_node=None,
                 lr=None, l2=None, step=1, decay=None, lr_dc=0.1, nonhybrid=False):
        super(ReGNN, self).__init__(hidden_size, out_size, batch_size, nonhybrid)
        self.embedding = tf.get_variable(shape=[n_node, hidden_size], name='embedding', dtype=tf.float32,
                                         initializer=tf.random_uniform_initializer(-self.stdv,
                                                                                   self.stdv))
        self.adj_in = tf.placeholder(dtype=tf.float32, shape=[self.batch_size, None, None])
        self.adj_out = tf.placeholder(dtype=tf.float32, shape=[self.batch_size, None, None])
        self.n_node = n_node
        self.L2 = l2
        self.step = step
        self.nonhybrid = nonhybrid
        self.W_in = tf.get_variable('W_in', shape=[self.out_size, self.out_size], dtype=tf.float32,
                                    initializer=tf.random_uniform_initializer(-self.stdv, self.stdv))
        self.b_in = tf.get_variable('b_in', [self.out_size], dtype=tf.float32,
                                    initializer=tf.random_uniform_initializer(-self.stdv, self.stdv))
        self.W_out = tf.get_variable('W_out', [self.out_size, self.out_size], dtype=tf.float32,
                                     initializer=tf.random_uniform_initializer(-self.stdv, self.stdv))
        self.b_out = tf.get_variable('b_out', [self.out_size], dtype=tf.float32,
                                     initializer=tf.random_uniform_initializer(-self.stdv, self.stdv))
        with tf.variable_scope('regnn_model', reuse=None):
            self.loss_train, _ = self.forward(self.ggnn())
        with tf.variable_scope('regnn_model', reuse=True):
            self.loss_test, self.score_test = self.forward(self.ggnn(), train=False)
        self.global_step = tf.Variable(0)
        self.learning_rate = tf.train.exponential_decay(lr, global_step=self.global_step, decay_steps=decay,
                                                        decay_rate=lr_dc, staircase=True)
        self.opt = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss_train, global_step=self.global_step)

        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.99)
        config = tf.ConfigProto(gpu_options=gpu_options)
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)
        self.sess.run(tf.global_variables_initializer())

    def ggnn(self):
        fin_state = tf.nn.embedding_lookup(self.embedding, self.item)
        cell = tf.nn.rnn_cell.GRUCell(self.out_size)
        with tf.variable_scope('gru'):
            for i in range(self.step):
                fin_state = tf.reshape(fin_state, [self.batch_size, -1,
                                                   self.out_size])
                fin_state_in = tf.reshape(tf.matmul(tf.reshape(fin_state, [-1, self.out_size]),
                                                    self.W_in) + self.b_in, [self.batch_size, -1, self.out_size])
                fin_state_out = tf.reshape(tf.matmul(tf.reshape(fin_state, [-1, self.out_size]),
                                                     self.W_out) + self.b_out,
                                           [self.batch_size, -1, self.out_size])
                av = tf.concat([tf.matmul(self.adj_in, fin_state_in),
                                tf.matmul(self.adj_out, fin_state_out)], axis=-1)
                state_output, fin_state = \
                    tf.nn.dynamic_rnn(cell, tf.expand_dims(tf.reshape(av, [-1, 2 * self.out_size]), axis=1),
                                      initial_state=tf.reshape(fin_state, [-1, self.out_size]))
        return tf.reshape(fin_state, [self.batch_size, -1, self.out_size])
