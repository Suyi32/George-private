import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from testbed.util.commons import *
import numpy as np
import tensorflow as tf
import time
from scheduler import LinearScheduler

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
                 safety_requirement=0.1):

        self.n_actions = n_actions
        self.n_features = n_features
        self.lr = learning_rate
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
        final_step = 50000 
        self.clip_eps = 0.2
        self.lr_scheduler = LinearScheduler(self.lr, final_step, 'lr')
        self.epsilon_scheduler = LinearScheduler(self.clip_eps, final_step, 'epsilon')
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
                self.train_op = tf.train.AdamOptimizer(self.lr).minimize(loss)

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
        discounted_ep_rs_norm = -self._discount_and_norm_safety()

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
        if IfPrint:

            print("PPPO(learn_ppo): epoch: %d, tput: %f, self.ep_ss: %f, safe_mean: %f, entro: %f, loss: %f" % (
                epoch_i, np.mean(self.tput_batch), np.mean(self.ep_ss), np.mean(self.safe_batch), np.mean(entropy), np.mean(loss)))

        self.ep_obs, self.ep_as, self.ep_rs, self.ep_ss = [], [], [], []
        self.tput_batch, self.safe_batch = [], []

        self.lr_scheduler.decay(epoch_i)
        self.epsilon_scheduler.decay(epoch_i)
        self.clip_eps = self.epsilon_scheduler.get_variable()
        self.lr = self.lr_scheduler.get_variable()
        
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
        if IfPrint:

            print("PPPO(learn_vio): epoch: %d, tput: %f, self.ep_ss: %f, safe_mean: %f, entro: %f, loss: %f" % (
                epoch_i, np.mean(self.tput_batch), np.mean(self.ep_ss), np.mean(self.safe_batch), np.mean(entropy), np.mean(loss)))

        self.ep_obs, self.ep_as, self.ep_rs, self.ep_ss = [], [], [], []
        self.tput_batch, self.safe_batch = [], []

        return 0

    def learn(self, epoch_i, entropy_weight, Ifprint=False):

        if np.mean(self.safe_batch) < 0.5*self.safety_requirement:
            self.count += 1
        else:
            self.count = 0
        # if self.count > 5:
        #     self.start_cpo = True   
        if self.count > 20:
            self.start_cpo = True
        # self.start_cpo = True
        if not self.start_cpo:
            return self.learn_vio(epoch_i, entropy_weight, Ifprint)
            # print("self.count",self.count, "np.mean(self.safe_batch)", np.mean(self.safe_batch), "self.start_cpo", self.start_cpo)
            # return self.learn_ppo(epoch_i, entropy_weight, Ifprint)


        discounted_ep_rs_norm = self._discount_and_norm_rewards()

        self.feed_dict = {self.tf_obs: np.vstack(self.ep_obs), self.tf_acts: np.array(self.ep_as),
                          self.tf_vt: discounted_ep_rs_norm, self.entropy_weight: entropy_weight,
                          # self.tf_safe: np.array(self.ep_ss)
                          self.tf_safe: self._discount_and_norm_safety()}

        chosen_action_log_probs = self.sess.run(self.chosen_action_log_probs, self.feed_dict)  # used in safe_loss
        self.feed_dict[self.old_chosen_action_log_probs] = chosen_action_log_probs  # same value, but stop gradient
      

        ##### Suyi's Change PPO #####
        b, old_all_act_prob, old_params  = self.sess.run(
            [self.constraint_flat_gradients_op,
             self.all_act_prob,
             self.flat_params_op],
            self.feed_dict)
        ##### Suyi's Change PPO #####

        # kl diveregnce
        self.feed_dict[self.old_all_act_prob] = old_all_act_prob

        eps = 1e-8


        ##### Suyi's Change  PPPO #####
        safety_constraint = self.safety_requirement - np.mean(self.safe_batch)
        c = -safety_constraint
        if (np.dot(b, b) < eps):
            w = 0.0
            s = 0.0
        else:
            norm_b = np.sqrt(np.dot(b, b))
            unit_b = b / norm_b
            w = norm_b * do_conjugate_gradient(self.get_fisher_product, unit_b)
            s = np.dot(w, self.get_fisher_product(w))
        ##### Suyi's Change  PPPO #####
        
        ####################### PPPO modified the policy update direction BEGIN #######################
        c_scale = 0.0
        #beta_soft = 0.5 # 1.0 default/0.5 used/0.1 used/0.01 used
        optim_case = 0.0 # to be deleted
        if optim_case == 0: #constraint does not intersect
            beta_soft = 1.0
        else:
            beta_soft = 1.0


        # if optim_case >= 0:
        if optim_case >= 0: # Suyi's change
            ##### PPO change #####
            self.learn_ppo(epoch_i, entropy_weight, Ifprint)
            cur_params = self.sess.run(self.flat_params_op) 
            flat_descent_step = max(0, (b.T @ (cur_params - old_params) + c)/(s + eps)  ) * w  # Projection update

            ##### Normalization #####
            flat_descent_step_lr = 1.0
            norm_flat_descent_step = np.sqrt(np.dot(flat_descent_step, flat_descent_step))
            unit_flat_descent_step = flat_descent_step / (norm_flat_descent_step + eps)
            print("normalized flat_descent_step", unit_flat_descent_step, np.sum(unit_flat_descent_step))
            ##### Normalization #####

            ##### RMSProp #####
            # flat_descent_step_lr = 0.01
            # self.grad_squared = 0.9 * self.grad_squared + 0.1 * flat_descent_step * flat_descent_step
            # print("RMSProp", (flat_descent_step_lr/(np.sqrt(self.grad_squared)+eps)) * flat_descent_step, np.sum((flat_descent_step_lr/(np.sqrt(self.grad_squared)+eps)) * flat_descent_step))
            ##### RMSProp #####


            new_params = cur_params - flat_descent_step_lr * unit_flat_descent_step
            # new_params = cur_params - (flat_descent_step_lr/(np.sqrt(self.grad_squared)+eps)) * flat_descent_step
            self.sess.run(self.params_assign_op, feed_dict={self.new_params: new_params})

            # This is used to remove the line search procedure
            optim_case = -1 # Suyi' change
        else:
            # when infeasible, take the step that purly decreases the constrained value
            flat_descent_step = np.sqrt(delta / (s + eps)) * w
            optim_case = -2 # Suyi' change

        # _, loss, all_act_prob, entro = self.sess.run([self.train_op, self.loss, self.all_act_prob, self.entro], feed_dict=self.feed_dict)
        # if Ifprint:
            # print("self.tput_batch:", self.tput_batch, "self.ep_ss:", self.ep_ss)
            # print("PPPO(learn): time: %f, epoch: %d, tput: %f, self.ep_ss: %f, safe_mean: %f, new_kl_divergence: %f, new_safety_loss: %f, new_loss: %f, entro: %f" % (
            #     time.time(), epoch_i, np.mean(self.tput_batch), np.mean(self.ep_ss), np.mean(self.safe_batch), new_kl_divergence, new_safety_loss, new_loss, entro))

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