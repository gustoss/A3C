import numpy as np
import tensorflow as tf
import multiprocessing
import threading
import time

from ActorCritic import AC
from Agent import Agent

seed = 13
load = None#'save/a3c-185829'
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

with tf.Session(config=config) as sess:
    tf.set_random_seed(seed)
    coord = tf.train.Coordinator()
    ags = []
    with tf.variable_scope('global', reuse=tf.AUTO_REUSE):
        ac = AC([None, 80, 80, 1], 4, sess)

    ags.append(Agent('a', sess, env='ppaquette/DoomDefendCenter-v0', eps=2000, render=True))
    num_ag = multiprocessing.cpu_count()
    for i in range(num_ag):
        ags.append(Agent(i, sess, env='ppaquette/DoomDefendCenter-v0', eps=5000))

    sess.run(tf.global_variables_initializer())
    if load is not None:
        ac.load(sess, load)
    
    ag_threads = []
    for ag in ags:
        fn = lambda: ag.work()
        t = threading.Thread(target=fn)
        t.start()
        time.sleep(2.)
        ag_threads.append(t)

    coord.join(ag_threads)
