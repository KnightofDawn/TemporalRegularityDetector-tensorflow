# tensorflow
import tensorflow as tf
tf.set_random_seed(2016)
sess = tf.Session()

# numpy
import numpy as np
np.random.seed(2016)

from model import TemporalRegularityDetector:

# CHECK : data.raw
height     = 227
width      = 227
batch_size = 128

inputs_ = np.fromfile("data.raw", dtype=np.float32).reshape([-1, height, width, 10])
input_shape = [None, height, width, 1]
detector = TemporalRegularityDetector(sess, input_shape)

batch_idx = 0
for i in range(epoch_num):
  while batch_idx < inputs_.shape[0]:
    detector.fit(input_)
    
    batch_idx += batch_size
  detector.save("ckpt/model.ckpt")
