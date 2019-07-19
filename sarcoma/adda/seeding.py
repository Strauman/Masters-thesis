import numpy as np
seed_state=tuple(np.random.randint(1,100000,4))
from sys import argv
if "--seed" in argv:
    seedstate_arg=argv[argv.index("--seed")+1]
    seed_state=eval(seedstate_arg)
    print(f"SETTING SEED FROM ARGS: {seed_state}")
else:
    # seed_state=(77089, 23378, 35128, 76532)
    # seed_state=(78446, 84890, 41731, 81688)
    # seed_state=(78446, 84890, 41731, 81688)
    # seed_state=(39650, 96297, 61125, 53982)
    # seed_state=(56758, 8115, 66619, 24503)
    seed_state=(33723, 77330, 73167, 13166)
    pass
import os
os.environ['PYTHONHASHSEED']=str(seed_state[0])
import random
random.seed(seed_state[1])
np.random.seed(seed_state[2])
import tensorflow as tf
tf.random.set_random_seed(seed_state[3])

generate = tf.random_uniform((10,), 0, 10)

def set_seeds():
    os.environ['PYTHONHASHSEED']=str(seed_state[0])
    random.seed(seed_state[1])
    np.random.seed(seed_state[2])
    tf.random.set_random_seed(seed_state[3])
def set_seed_state(state):
    os.environ['PYTHONHASHSEED']=str(state[0])
    random.seed(state[1])
    np.random.seed(state[2])
    tf.random.set_random_seed(state[3])
