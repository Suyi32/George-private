# other name: PolicyGradient_PCPO_PPPO_me_mini_sim.py
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from testbed.util.commons import *
import numpy as np
# np.set_printoptions(threshold=sys.maxsize)

import tensorflow as tf
import time


np.random.seed(1)
tf.set_random_seed(1)
desired_kl = 0.0001


class PolicyGradient:

    def get_fisher_product_op(self):
        directional_gradients = tf.reduce_sum(self.kl_flat_gradients_op*self.vec)
        return get_flat_gradients(directional_gradients, self.trainable_variables)

    def get_fisher_product(self, vec, damping = 1e-3):
        self.feed_dict[self.vec] = vec
        return self.sess.run(self.fisher_product_op, self.feed_dict) + damping*vec

    def __init__(self,
                 n_actions,
                 n_features,
                 learning_rate=0.001,
                 suffix="",
                 safety_requirement=0.1,
                 params=None):

        self.n_actions = n_actions
        self.n_features = n_features
        self.lr = learning_rate
        print("learning rate: {}".format(self.lr))
        self.pg_lr = 0.01
        self.suffix = suffix

        self.safety_requirement = safety_requirement

        """
        self.ep_obs, self.ep_as, self.ep_rs: observation, action, reward, recorded for each batch
        """
        self.ep_obs, self.ep_as, self.ep_rs, self.ep_ss = [], [], [], []

        """
        self.tput_batch: record throughput for each batch, used to show progress while training
        self.tput_persisit, self.episode: persist to record throughput, used to be stored and plot later
        """
        self.tput_batch, self.tput_persisit, self.safe_batch, self.safe_persisit = [], [], [], []
        self.coex_persisit, self.sum_persisit = [], []
        self.node_used_persisit = []
        self.episode = []
        self.node_used = []
        self.ss_perapp_persisit = []
        self.ss_coex_persisit = []
        self.ss_sum_persisit = []
        self.start_cpo =False
        self.count = 0
        # TODO self.vio = []: violation
        ##### PPO CHANGE #####
        self.ppo_sample_inter = 3
        self.ppo_sample_counter = 0
        self.grad_squared = 0
        self.params = params 
        self.clip_eps = params['clip_eps']
        ##### PPO CHANGE #####

        self._build_net()
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        restore = ['Actor' + self.suffix  + '/fc1' + self.suffix + '/kernel:0', 'Actor' + self.suffix  + '/fc1' + self.suffix + '/bias:0', 'Actor' + self.suffix  + '/fc2' + self.suffix + '/kernel:0', 'Actor' + self.suffix  + '/fc2' + self.suffix + '/bias:0']
        restore_var = [v for v in tf.all_variables() if v.name in restore]
        self.saver = tf.train.Saver(var_list=restore_var)
        # self.saver = tf.train.Saver()

    def _build_net(self):

        with tf.variable_scope("Actor"+self.suffix):

            with tf.name_scope('inputs'+self.suffix):
                self.tf_obs = tf.placeholder(tf.float32, [None, self.n_features], name='observation'+self.suffix)
                self.tf_acts = tf.placeholder(tf.int32, [None, ], name='actions_num'+self.suffix)
                self.tf_vt = tf.placeholder(tf.float32, [None, ], name='actions_value'+self.suffix)
                self.tf_safe = tf.placeholder(tf.float32, [None, ], name='safety_value'+self.suffix)
                self.entropy_weight = tf.placeholder(tf.float32, shape=(), name='entropy_weight_clustering'+self.suffix)

                ##### PPO change #####
                self.ppo_ratio = tf.placeholder(tf.float32, [None, ], name='ppo_ratio'+self.suffix)
                ##### PPO change #####

            layer = tf.layers.dense(
                inputs=self.tf_obs,
                units=128,
                activation=tf.nn.tanh,
                # kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.3),
                kernel_initializer=tf.orthogonal_initializer(gain=np.sqrt(2.)), # ppo default initialization
                bias_initializer=tf.constant_initializer(0.1),
                name='fc1'+self.suffix
            )

            all_act = tf.layers.dense(
                inputs=layer,
                units=self.n_actions,
                activation=None,
                # kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.3),
                kernel_initializer=tf.orthogonal_initializer(gain=np.sqrt(2.)), # ppo default initialization
                bias_initializer=tf.constant_initializer(0.1),
                name='fc2'+self.suffix
            )

            self.trainable_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Actor'+self.suffix)
            self.trainable_variables_shapes = [var.get_shape().as_list() for var in self.trainable_variables]

            # sampling
            self.all_act_prob = tf.nn.softmax(all_act, name='act_prob'+self.suffix)
            self.all_act_prob = tf.clip_by_value(self.all_act_prob, 1e-20, 1.0)

            with tf.name_scope('loss'+self.suffix):
                neg_log_prob = tf.reduce_sum(-tf.log(tf.clip_by_value(self.all_act_prob, 1e-30, 1.0)) * tf.one_hot(indices=self.tf_acts, depth=self.n_actions), axis=1)
                loss = tf.reduce_mean(neg_log_prob * self.tf_vt)
                loss += self.entropy_weight * tf.reduce_mean(tf.reduce_sum(tf.log(tf.clip_by_value(self.all_act_prob, 1e-30, 1.0)) * self.all_act_prob, axis=1))
                self.entro = self.entropy_weight * tf.reduce_mean(tf.reduce_sum(tf.log(tf.clip_by_value(self.all_act_prob, 1e-30, 1.0)) * self.all_act_prob, axis=1))
                self.loss = loss
            with tf.name_scope('train' + self.suffix):
                self.train_op = tf.train.AdamOptimizer(self.pg_lr).minimize(loss)

            # safety loss
            """
            * -1?
            """
            self.chosen_action_log_probs = tf.reduce_sum(tf.log(tf.clip_by_value(self.all_act_prob, 1e-30, 1.0)) * tf.one_hot(indices=self.tf_acts, depth=self.n_actions), axis=1)
            ##### PPO CHANGE #####
            self.ppo_old_chosen_action_log_probs = tf.placeholder(tf.float32, [None])
            ##### PPO CHANGE #####
            self.old_chosen_action_log_probs = tf.stop_gradient(tf.placeholder(tf.float32, [None]))
            # self.each_safety_loss = tf.exp(self.chosen_action_log_probs - self.old_chosen_action_log_probs) * self.tf_safe
            self.each_safety_loss = (tf.exp(self.chosen_action_log_probs) - tf.exp(self.old_chosen_action_log_probs)) * self.tf_safe
            self.average_safety_loss = tf.reduce_mean(self.each_safety_loss)  #/ self.n_episodes tf.reduce_sum
            # self.average_safety_loss +=self.entro

            # KL D
            self.old_all_act_prob = tf.stop_gradient(tf.placeholder(tf.float32, [None, self.n_actions]))

            def kl(x, y):
                EPS = 1e-10
                x = tf.where(tf.abs(x) < EPS, EPS * tf.ones_like(x), x)
                y = tf.where(tf.abs(y) < EPS, EPS * tf.ones_like(y), y)
                X = tf.distributions.Categorical(probs=x + EPS)
                Y = tf.distributions.Categorical(probs=y + EPS)
                return tf.distributions.kl_divergence(X, Y, allow_nan_stats=False)


            self.each_kl_divergence = kl(self.all_act_prob, self.old_all_act_prob)  # tf.reduce_sum(kl(self.all_act_prob, self.old_all_act_prob), axis=1)
            self.average_kl_divergence = tf.reduce_mean(self.each_kl_divergence)
            # self.kl_gradients = tf.gradients(self.average_kl_divergence, self.trainable_variables)  # useless

            self.desired_kl = desired_kl
            # self.metrics = [self.loss, self.average_kl_divergence, self.average_safety_loss, self.entro] # Luping
            self.metrics = [self.loss, self.loss, self.average_safety_loss, self.entro] # Luping

            # FLat
            self.flat_params_op = get_flat_params(self.trainable_variables)
            """not use tensorflow default function, here we calculate the gradient by self:
            (1) loss: g
            (2) kl: directional_gradients (math, fisher)
            (3) safe: b 
            """
            ##### PPO change #####
            #### PPO Suyi's Change ####
            with tf.name_scope('ppoloss' + self.suffix):
                self.ppo_ratio = tf.exp(self.chosen_action_log_probs - self.ppo_old_chosen_action_log_probs)
                # self.ppo_ratio = tf.Print(self.ppo_ratio, [self.ppo_ratio], "self.ppo_ratio: ")

                surr = self.ppo_ratio * self.tf_vt
                self.ppoloss = -tf.reduce_mean(tf.minimum(
                    surr,
                    tf.clip_by_value(self.ppo_ratio, 1.- self.clip_eps, 1.+ self.clip_eps) * self.tf_vt))
                
                self.ppoloss += self.entropy_weight * tf.reduce_mean(tf.reduce_sum(tf.log(tf.clip_by_value(self.all_act_prob, 1e-30, 1.0)) * self.all_act_prob, axis=1))
                # self.ppoloss += 0.01 * tf.reduce_mean(tf.reduce_sum(tf.log(tf.clip_by_value(self.all_act_prob, 1e-30, 1.0)) * self.all_act_prob, axis=1))

            with tf.variable_scope('ppotrain'):
                # self.atrain_op = tf.train.AdamOptimizer(self.lr).minimize(self.ppoloss)
                self.atrain_op = tf.train.AdamOptimizer(self.lr).minimize(self.ppoloss)
            #### PPO Suyi's Change ####

            self.ppoloss_flat_gradients_op = get_flat_gradients(self.ppoloss, self.trainable_variables)
            ##### PPO change #####

            self.loss_flat_gradients_op = get_flat_gradients(self.loss, self.trainable_variables)
            self.kl_flat_gradients_op = get_flat_gradients(self.average_kl_divergence, self.trainable_variables)
            self.constraint_flat_gradients_op = get_flat_gradients(self.average_safety_loss, self.trainable_variables)

            self.vec = tf.placeholder(tf.float32, [None])
            self.fisher_product_op = self.get_fisher_product_op()

            self.new_params = tf.placeholder(tf.float32, [None])
            self.params_assign_op = assign_network_params_op(self.new_params, self.trainable_variables, self.trainable_variables_shapes)

            # with tf.name_scope('train'):
            #     self.train_op = tf.train.AdamOptimizer(self.lr).minimize(loss)
                # decay_rate =0.99999 # 0.999
                # learning_rate = 1e-1
                # self.train_op = tf.train.RMSPropOptimizer(learning_rate, decay_rate).minimize(loss)

    def choose_action(self, observation):
        prob_weights = self.sess.run(self.all_act_prob, feed_dict={self.tf_obs: observation})  # (4,) ->(1,4)
        action = np.random.choice(range(prob_weights.shape[1]), p=prob_weights.ravel())
        return action, prob_weights

    def choose_action_determine(self, observation):
        prob_weights = self.sess.run(self.all_act_prob, feed_dict={self.tf_obs: observation})  # (4,) ->(1,4)
        action = np.argmax(prob_weights.ravel())
        return action, prob_weights

    def store_training_samples_per_episode(self, s, a, r, ss):
        self.ep_obs.extend(s)
        self.ep_as.extend(a)
        self.ep_rs.extend(r)
        self.ep_ss.extend(ss)

    def store_tput_per_episode(self, tput, episode, list_check, list_check_per_app, list_check_coex, list_check_sum):
        self.tput_batch.append(tput)
        self.tput_persisit.append(tput)
        self.episode.append(episode)
        self.safe_batch.append(list_check)
        self.ss_perapp_persisit.append(list_check_per_app)
        self.ss_coex_persisit.append(list_check_coex)
        self.ss_sum_persisit.append(list_check_sum)

    def learn_ppo (self, epoch_i, entropy_weight, IfPrint=False):
        discounted_ep_rs_norm = self._discount_and_norm_rewards()

        if self.ppo_sample_counter == 0:
            self.ppo_old_params = self.sess.run(self.flat_params_op)
            
        cur_params = self.sess.run(self.flat_params_op)
        self.sess.run(self.params_assign_op, feed_dict={self.new_params: self.ppo_old_params})
        ppo_past_chosen_action_log_probs = self.sess.run(self.chosen_action_log_probs, feed_dict={
                            self.tf_obs: np.vstack(self.ep_obs), 
                            self.tf_acts: np.array(self.ep_as),
                            self.tf_vt: discounted_ep_rs_norm, 
                            self.entropy_weight: entropy_weight,
                            # self.tf_safe: np.array(self.ep_ss)
                            self.tf_safe: self._discount_and_norm_safety()})  # used in safe_loss  
        self.sess.run(self.params_assign_op, feed_dict={self.new_params: cur_params})                  
        
        self.ppo_old_params = self.sess.run(self.flat_params_op)
        self.ppo_sample_counter += 1          

        for _ in range(1):
            _, loss, all_act_prob, entropy = self.sess.run([self.atrain_op, self.ppoloss, self.all_act_prob, self.entro], feed_dict={
                self.tf_obs: np.vstack(self.ep_obs),
                self.tf_acts: np.array(self.ep_as),
                self.tf_vt: discounted_ep_rs_norm,
                self.entropy_weight: entropy_weight,
                self.tf_safe: self._discount_and_norm_safety(),
                self.ppo_old_chosen_action_log_probs: ppo_past_chosen_action_log_probs
        })

        # print("{} learn_ppo: epoch: {}, all_act_prob: {}".format(self.suffix, epoch_i, all_act_prob.tolist() ))
        if IfPrint:
            print("PPPO(learn_ppo): epoch: %d, tput: %f, self.ep_ss: %f, safe_mean: %f, entro: %f, loss: %f" % (
                epoch_i, np.mean(self.tput_batch), np.mean(self.ep_ss), np.mean(self.safe_batch), np.mean(entropy), np.mean(loss)))

        self.ep_obs, self.ep_as, self.ep_rs, self.ep_ss = [], [], [], []
        self.tput_batch, self.safe_batch = [], []

        return 5

    def learn_vio (self, epoch_i, entropy_weight, IfPrint=False):
        discounted_ep_rs_norm = -self._discount_and_norm_safety()

        for _ in range(1):
            _, loss, all_act_prob, entropy = self.sess.run([self.train_op, self.loss, self.all_act_prob, self.entro], feed_dict={
                self.tf_obs: np.vstack(self.ep_obs),
                self.tf_acts: np.array(self.ep_as),
                self.tf_vt: discounted_ep_rs_norm,
                self.entropy_weight: entropy_weight
        })

        # print("{} learn_vio: epoch: {}, all_act_prob: {}".format(self.suffix, epoch_i, all_act_prob.tolist() ))
        if IfPrint:
            print("PPPO(learn_vio): epoch: %d, tput: %f, self.ep_ss: %f, safe_mean: %f, entro: %f, loss: %f" % (
                epoch_i, np.mean(self.tput_batch), np.mean(self.ep_ss), np.mean(self.safe_batch), np.mean(entropy), np.mean(loss)))

        self.ep_obs, self.ep_as, self.ep_rs, self.ep_ss = [], [], [], []
        self.tput_batch, self.safe_batch = [], []

        return 0

    def learn(self, epoch_i, entropy_weight, Ifprint=False):

        if np.mean(self.ep_ss) < self.safety_requirement:
            self.count += 1
        else:
            self.count = 0
        if self.count > 10:
            self.start_cpo = True
        # self.start_cpo = True
        if not self.start_cpo:
            print("self.count",self.count, "np.mean(self.ep_ss)", np.mean(self.ep_ss), "self.start_cpo", self.start_cpo)
            return self.learn_vio(epoch_i, entropy_weight, Ifprint)
        discounted_ep_rs_norm = self._discount_and_norm_rewards()
        self.feed_dict = {self.tf_obs: np.vstack(self.ep_obs), self.tf_acts: np.array(self.ep_as),
                          self.tf_vt: discounted_ep_rs_norm, self.entropy_weight: entropy_weight,
                          # self.tf_safe: np.array(self.ep_ss)
                          self.tf_safe: self._discount_and_norm_safety()}

        chosen_action_log_probs = self.sess.run(self.chosen_action_log_probs, self.feed_dict)  # used in safe_loss
        self.feed_dict[self.old_chosen_action_log_probs] = chosen_action_log_probs  # same value, but stop gradient

        # g, b, old_all_act_prob, old_params, old_safety_loss = self.sess.run(
        #     [self.loss_flat_gradients_op,
        #      self.constraint_flat_gradients_op,
        #      self.all_act_prob,
        #      self.flat_params_op,
        #      self.average_safety_loss],
        #     self.feed_dict)

        b, old_all_act_prob, old_safety_loss = self.sess.run(
            [self.constraint_flat_gradients_op,
             self.all_act_prob,
             self.average_safety_loss],
            self.feed_dict)

        # kl diveregnce
        self.feed_dict[self.old_all_act_prob] = old_all_act_prob

        # math
        # v = do_conjugate_gradient(self.get_fisher_product, g)  # x = A-1g
        # H_b = doConjugateGradient(self.getFisherProduct, b)
        # approx_g = self.get_fisher_product(v)  # g = Ax = AA-1g
        # b = self.getFisherProduct(H_b)
        safety_constraint = self.safety_requirement - np.mean(self.ep_ss)
        linear_constraint_threshold = np.maximum(0, safety_constraint) + old_safety_loss
        eps = 1e-8
        delta = 2 * self.desired_kl
        c = -safety_constraint
        # q = np.dot(approx_g, v)

        if (np.dot(b, b) < eps):
            # lam = np.sqrt(q / delta)
            # nu = 0
            w = 0
            r, s, A, B = 0, 0, 0, 0
            optim_case = 4
        else:
            norm_b = np.sqrt(np.dot(b, b))
            unit_b = b / norm_b
            w = norm_b * do_conjugate_gradient(self.get_fisher_product, unit_b)
            # r = np.dot(w, approx_g)
            s = np.dot(w, self.get_fisher_product(w))
            # A = q - (r ** 2 / s)
            B = delta - (c ** 2 / s)
            if (c < 0 and B < 0):
                optim_case = 3
            elif (c < 0 and B > 0):
                optim_case = 2
            elif (c > 0 and B > 0):
                optim_case = 1
            else:
                optim_case = 0
                # return self.learn_vio(epoch_i, entropy_weight, Ifprint)
            # lam = np.sqrt(q / delta)
            # nu = 0

            # if (optim_case == 2 or optim_case == 1):
            #     lam_mid = r / c
            #     L_mid = - 0.5 * (q / lam_mid + lam_mid * delta)

            #     lam_a = np.sqrt(A / (B + eps))
            #     L_a = -np.sqrt(A * B) - r * c / (s + eps)

            #     lam_b = np.sqrt(q / delta)
            #     L_b = -np.sqrt(q * delta)

            #     if lam_mid > 0:
            #         if c < 0:
            #             if lam_a > lam_mid:
            #                 lam_a = lam_mid
            #                 L_a = L_mid
            #             if lam_b < lam_mid:
            #                 lam_b = lam_mid
            #                 L_b = L_mid
            #         else:
            #             if lam_a < lam_mid:
            #                 lam_a = lam_mid
            #                 L_a = L_mid
            #             if lam_b > lam_mid:
            #                 lam_b = lam_mid
            #                 L_b = L_mid

            #         if L_a >= L_b:
            #             lam = lam_a
            #         else:
            #             lam = lam_b

            #     else:
            #         if c < 0:
            #             lam = lam_b
            #         else:
            #             lam = lam_a

            #     nu = max(0, lam * c - r) / (s + eps)


        '''
        ### CPO policy update ###
        if optim_case > 0:
            full_step = (1. / (lam + eps)) * (v + nu * w)

        else:
            full_step = np.sqrt(delta / (s + eps)) * w
        # print("optim_case: %f" %(optim_case))

        if (optim_case == 0 or optim_case == 1):
            new_params, status, new_kl_divergence, new_safety_loss, new_loss, entro = do_line_search_CPO(self.get_metrics, old_params, full_step, self.desired_kl, linear_constraint_threshold, check_loss=False)
        else:
            new_params, status, new_kl_divergence, new_safety_loss, new_loss, entro = do_line_search_CPO(self.get_metrics, old_params, full_step, self.desired_kl, linear_constraint_threshold)
        '''

        ####################### PCPO modified the policy update direction BEGIN #######################
        c_scale = 0.0
        #beta_soft = 0.5 # 1.0 default/0.5 used/0.1 used/0.01 used
        if optim_case == 0: #constraint does not intersect
            beta_soft = 1.0
        else:
            beta_soft = 1.0

        if optim_case > 0:
            old_params = self.sess.run(self.flat_params_op)
            self.learn_ppo(epoch_i, entropy_weight, Ifprint)
            cur_params = self.sess.run(self.flat_params_op)


            # print("[KEY] b.T @ (cur_params - old_params) = {}".format(b.T @ (cur_params - old_params)))
            # print("[KEY] np.sqrt(delta / (q + eps)) * r = {}".format(np.sqrt(delta / (q + eps)) * r))
            # flat_descent_step_tr = np.sqrt(delta / (q + eps)) * v # sqrt(2 * delta / g^t H^{-1} g) H^{-1} g

            # check the Lagrangian multipiler of the projected gradient descent
            #lam_pg = max(0, np.sqrt(delta / (q + eps)) * r + c + c_scale) / (flat_b.dot(flat_b) + eps) 
            # projected gradient descent
            #flat_descent_step_pg = lam_pg * flat_b

            # This Lagrangian multipiler is derived from using KL projection
            # lam_pg = max(0, np.sqrt(delta / (q + eps)) * r + c + c_scale) / (s + eps)
            lam_pg = max(0, b.T @ (cur_params - old_params) + c + c_scale) / (s + eps)

            # KL projected gradient descent
            flat_descent_step_pg_KL = lam_pg * w

            # combine these two directions
            # flat_descent_step = flat_descent_step_tr + beta_soft * flat_descent_step_pg_KL
            flat_descent_step = beta_soft * flat_descent_step_pg_KL

            # This is used to remove the line search procedure
            optim_case = -1 # Suyi' change
        else:
            cur_params = self.sess.run(self.flat_params_op)
            # when infeasible, take the step that purly decreases the constrained value
            flat_descent_step = np.sqrt(delta / (s + eps)) * w
            optim_case = -2 # Suyi' change

        # This is used to remove the line search procedure
        # optim_case = -1   Suyi' change

        ## Replacing line search
        def pcpo_update(f, old_params, full_step, desired_kl_divergence, linear_constraint_threshold):
            old_loss, old_kl_divergence, old_safety_loss, entro_old = f(old_params)

            new_params = old_params - full_step
            new_loss, new_kl_divergence, new_safety_loss, entro = f(new_params)

            if np.isnan(full_step).any():
                return  old_params, True, new_kl_divergence, new_safety_loss, new_loss, entro

            if np.isnan(new_params).any():
                return old_params, False, new_kl_divergence, new_safety_loss, new_loss, entro
            else:
                return new_params, True, new_kl_divergence, new_safety_loss, new_loss, entro

            return old_params, False, new_kl_divergence, new_safety_loss, new_loss, entro

        if optim_case == -1 or optim_case == -2:
            # new_params, status, new_kl_divergence, new_safety_loss, new_loss, entro = pcpo_update(self.get_metrics, old_params, flat_descent_step, self.desired_kl, linear_constraint_threshold)
            new_params, status, new_kl_divergence, new_safety_loss, new_loss, entro = pcpo_update(self.get_metrics, cur_params, flat_descent_step, self.desired_kl, linear_constraint_threshold)
        ####################### PCPO modified the policy update direction END #######################

        print('Success: ', status, "optim_case:", optim_case)

        if (status == False):
            self.sess.run(self.params_assign_op, feed_dict={self.new_params: new_params})


        # _, loss, all_act_prob, entro = self.sess.run([self.train_op, self.loss, self.all_act_prob, self.entro], feed_dict=self.feed_dict)
        # if Ifprint:
        #     print("PCPO(learn): time: %f, epoch: %d, tput: %f, self.ep_ss: %f, safe_mean: %f, new_kl_divergence: %f, new_safety_loss: %f, new_loss: %f, entro: %f" % (
        #         time.time(), epoch_i, np.mean(self.tput_batch), np.mean(self.ep_ss), np.mean(self.safe_batch), new_kl_divergence, new_safety_loss, new_loss, entro))

        self.ep_obs, self.ep_as, self.ep_rs, self.ep_ss = [], [], [], []
        self.tput_batch, self.safe_batch = [], []

        return optim_case

    def get_metrics(self, new_params):
        self.sess.run(self.params_assign_op, feed_dict = {self.new_params: new_params})
        return self.sess.run(self.metrics, self.feed_dict)

    def _discount_and_norm_rewards(self):
        """
        Normalize reward per batch
        :return:
        """
        discounted_ep_rs = np.array(self.ep_rs)
        discounted_ep_rs -= np.mean(discounted_ep_rs)
        if np.std(discounted_ep_rs) != 0:
            discounted_ep_rs /= np.std(discounted_ep_rs)
        return discounted_ep_rs

    def save_session(self, ckpt_path):
        self.saver.save(self.sess, ckpt_path)

    def restore_session(self, ckpt_path):
        # restore = ['Actor' + self.suffix + '/fc1' + self.suffix + '/kernel:0', 'Actor' + self.suffix + '/fc1' + self.suffix + '/bias:0', 'Actor' + self.suffix + '/fc0' + self.suffix + '/kernel:0',
        #            'Actor' + self.suffix + '/fc0' + self.suffix + '/bias:0']
        # restore_var = [v for v in tf.all_variables() if v.name in restore]
        # self.saver = tf.train.Saver(var_list=restore_var)
        self.saver.restore(self.sess, ckpt_path)
        # self.saver = tf.train.Saver()

    def _discount_and_norm_safety(self):
        """
                Normalize safety violation per batch
                :return:
                """
        discounted_ep_ss = np.array(self.ep_ss)
        discounted_ep_ss -= np.mean(discounted_ep_ss)
        if np.std(discounted_ep_ss) != 0:
            discounted_ep_ss /= np.std(discounted_ep_ss)
        return discounted_ep_ss