import numpy as np
import tensorflow as tf
from ActorCritic import AC
# import random as rd
import gym
import gym_ple
import gym_pull
from ppaquette_gym_doom.wrappers.action_space import ToDiscrete

from gym.wrappers import SkipWrapper
from ActorCritic import AC
from Image import PreprocessImage
import time
class Agent:
    def __init__(self, name, sess, env='ppaquette/DoomDefendCenter-v0', eps=100, eps_save=0, time=400, learning=1e-6, gamma=.96, tau=.94, seed=None, render=False, rec=False):
        self.name = 'agent_' + str(name)
        self.eps = eps
        self.eps_ran = 0
        self.time = time
        self.render = render
        self.learning = learning
        self.gamma = gamma
        self.tau = tau
        self.sess = sess
        self.eps_save = eps_save

        self.env = gym.make(env)
        self.env = ToDiscrete("minimal")(self.env)
        self.env = SkipWrapper(2)(self.env)
        self.env = PreprocessImage(self.env, height=80, width=80, grayscale=True)
        if rec:
            self.env = gym.wrappers.Monitor(self.env, 'videos', force=True)
        if seed is not None:
            self.env.seed(seed)
            # rd.seed(seed)
            # tf.set_random_seed(seed)

        self.output = self.env.action_space.n
        self.input = np.concatenate(([None], list(self.env.observation_space.shape)))

        with tf.variable_scope('global', reuse=True):
            self.global_step = tf.Variable(0, name='global_step', trainable=False)
            global_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'global')
        
        if self.eps_save > 0:
            vars_save = {v.op.name: v for v in global_vars}
            self.saver = tf.train.Saver(vars_save)
            self.check_save = self.save
        else:
            self.check_save = lambda: None

        with tf.variable_scope(self.name):
            self.ac = AC(self.input, self.output, self.sess)
            local_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.name)
            with tf.variable_scope('loss_critic'):
                self.target_critic = tf.placeholder(tf.float32, [None], name='target_critic')
                self.batch = tf.placeholder(tf.int32, name='batch')
                self.loss_critic = tf.reduce_sum(tf.square(self.target_critic - tf.reshape(self.ac.value_critic, [self.batch]), name='square_loss_critic') * .5, name='reduce_sum_critic')

            with tf.variable_scope('loss_actor'):
                self.td_error = tf.placeholder(tf.float32, [None], name='td_error')
                self.action = tf.placeholder(tf.int32, [None], name='action')
                self.loss_actor = tf.reduce_sum(self.td_error * tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.ac.action_logits, labels=self.action))
                entropy = -tf.reduce_sum(tf.nn.softmax(self.ac.action_logits) * tf.nn.log_softmax(self.ac.action_logits))
                self.loss_actor -= entropy * .02

            with tf.variable_scope('loss'):
                self.loss_total = self.loss_actor + .5 * self.loss_critic

        gradients = tf.gradients(self.loss_total, local_vars)
        gradients, _ = tf.clip_by_global_norm(gradients, 10.)

        with tf.variable_scope('global', reuse=tf.AUTO_REUSE):
            opt = tf.train.AdagradOptimizer(self.learning)
            self.train_op = [opt.apply_gradients(zip(gradients, global_vars)), self.global_step.assign_add(self.batch)]

        self.reset_model = [l.assign(g) for l, g in zip(local_vars, global_vars)]
            
    def work(self):
        while self.eps_ran < self.eps:
            self.sess.run(self.reset_model)
            self.eps_ran += 1
            self.memory = []
            self.lstm_state = self.ac.initial_state()
            self.run()
            self.learn()
            self.check_save()

    def run(self):
        self.R = 0
        self.Rr = 0
        s = self.env.reset()
        for _ in range(self.time):
            if self.render:
                self.env.render()
            a, self.lstm_state = self.ac.act([s], self.lstm_state, rand=False)
            a = a[0]
            s_, r, d, i = self.env.step(a)
            self.R = self.R + r
            r = r if d else r - .0001
            self.Rr = self.Rr + r
            self.push(s, a, r, s_, d)
            s = s_
            if d:
                break

    def push(self, s, a, r, s_, d):
        self.memory.append((s, a, r, d))
        self.s_ = s_

    def save(self):
        if self.eps_ran % self.eps_save == 0:
            st = self.saver.save(self.sess, 'save/a3c', global_step=self.global_step)
            print('Saved!\n{0}'.format(st))

    def learn(self):
        Rc_target = []
        Rc = .0 if self.memory[-1][-1] else self.sess.run(tf.reduce_max(self.ac.action_logits, axis=1, name='reduce_max'), feed_dict={self.ac.x: [self.memory[-1][0]]})[0]
        for i in reversed(range(len(self.memory))):
            Rc = self.memory[i][2] + self.gamma * Rc
            Rc_target.append(Rc)

        Rc_target.reverse()
        state = np.array([self.memory[i][0] for i in range(len(self.memory))])
        action_actor = [self.memory[i][1] for i in range(len(self.memory))]
        state_tmp = np.concatenate((state, np.array([self.s_])), axis=0)
        feed_dict = {
            self.ac.x: state_tmp,
            self.ac.lstm_state: self.ac.initial_state(),
            self.ac.keep_prob: 1.
        }
        action_critic = np.reshape(self.sess.run(self.ac.value_critic, feed_dict=feed_dict), (len(state_tmp)))
        td_error = np.array([self.memory[i][2] + self.gamma * action_critic[i + 1] - action_critic[i] for i in range(len(state))])
        td_error = (td_error - td_error.mean()) / td_error.std()

        gae = []
        g = 0
        for i in reversed(range(len(td_error))):
            g = g * self.gamma * self.tau + td_error[i]
            gae.append(g)

        gae.reverse()
        gae = np.array(gae)
        gae = (gae - gae.mean()) / gae.std()

        feed_dict = {
            self.ac.x: state, 
            self.target_critic: Rc_target,
            self.td_error: gae,
            self.action: action_actor,
            self.ac.lstm_state: self.ac.initial_state(),
            self.batch: len(self.memory),
            self.ac.keep_prob: .5
        }

        _, lp, lv = self.sess.run([self.train_op, self.loss_actor, self.loss_critic], feed_dict=feed_dict)
        # global_step = self.sess.run(self.global_step)
        print('************************************************')
        print('Agent {2} {5}\nLoss Actor {0}\nLoss Critic {4}\nReward {1}\nReward cumulated {3}'.format(lp, self.R, self.name, self.Rr, lv, self.eps_ran))
        print('************************************************\n')

    def sample_in_out(self):
        return (self.input, self.output)
