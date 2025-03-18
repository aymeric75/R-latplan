# global setting

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

import warnings
warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('Agg')

import numpy as np
np.set_printoptions(formatter={'float_kind':lambda x: "%.5f" % x})

# https://stackoverflow.com/questions/48979426/keras-model-accuracy-differs-after-loading-the-same-saved-model
from numpy.random import seed
seed(42) # keras seed fixing

import tensorflow as tf
tf.compat.v1.set_random_seed(42)
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

import keras.backend.tensorflow_backend as K
# K.set_floatx('float16')
print("Default float: {}".format(K.floatx()))
import keras.backend
keras.backend.clear_session()

K.set_session(
    tf.compat.v1.Session(
        config=tf.compat.v1.ConfigProto(
            allow_soft_placement=True, 
            intra_op_parallelism_threads=16,
            inter_op_parallelism_threads=8,
            device_count = {'CPU': 32, 'GPU': 1},
            gpu_options =
            tf.compat.v1.GPUOptions(
                per_process_gpu_memory_fraction=0.8,
                allow_growth=True,))))

# # Check if GPUs are available
# gpus = tf.config.experimental.list_physical_devices('GPU')
# if gpus:
#     print("GPUs are available:")
#     for gpu in gpus:
#         print(f"  {gpu}")
# else:
#     print("No GPUs are available.")


# # Enable logging of device placement
# tf.debugging.set_log_device_placement(True)

# # Example computation to observe the placement
# # Example tensor
# with tf.device('/GPU:0'):
#     tensor = tf.constant([[1.0, 2.0, 3.0]])
#     print(f"Device: {tensor.device}")

# exit()


from keras_radam      import RAdam
import keras.optimizers
setattr(keras.optimizers,"radam", RAdam)



from . import util
from . import puzzles
from . import model
#from . import modelVanilla
from . import ama2model
from . import main

