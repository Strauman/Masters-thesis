import tensorflow as tf
import numpy as np
import pout
class BatchPersistent(object):
    pass
# class Concatenator(BatchPersistent):
#     """docstring for MeanAccumulator."""
#     def __init__(self, tensor, name=None, collections=None):
#         self.name = name
#         self.tensor = tensor
#         # self.conc_arr = tensor
#         self.batch_nums = batch_nums = tf.Variable(initial_value=0, dtype=tensor.dtype)
#         self.step_batch = step_batch = tf.assign_add(batch_nums, 1)
#         ###
#         self.conc_arr = tf.TensorArray(tensor.dtype, size=tf.shape(tensor)[0], dynamic_size=True, clear_after_read=None)
#         self.concat_op=self.conc_arr.write(self.batch_nums, tensor)
#         ###
#         self.update_op = [self.step_batch, self.concat_op]
#         self.flush_op=tf.no_op()
#         # self.output_tensor=tf.reduce_mean(self.conc_arr.stack(), axis=(1,2))
#         self.output_tensor=self.conc_arr.stack()
#         super(Concatenator, self).__init__()
class MeanAccumulator(BatchPersistent):
    """docstring for MeanAccumulator."""

    def __init__(self, tensor, name=None, collections=None):
        self.name = name
        self.tensor = tensor
        self.collections = collections
        self.accumulator = accumulator = tf.Variable(initial_value=tf.zeros_like(tensor), dtype=tensor.dtype)
        self.batch_nums = batch_nums = tf.Variable(initial_value=tf.zeros_like(tensor), dtype=tensor.dtype)
        self.accumulate_op = accumulate_op = tf.assign_add(accumulator, tensor)
        self.step_batch = step_batch = tf.assign_add(batch_nums, 1)
        self.update_op = update_op = tf.group([step_batch, accumulate_op])
        eps = 1e-5
        self.output_tensor = output_tensor = accumulator / (tf.nn.relu(batch_nums - eps) + eps)
        self.flush_op = flush_op = tf.group([tf.assign(accumulator, 0), tf.assign(batch_nums, 0)])
        super(MeanAccumulator, self).__init__()

    @staticmethod
    def make_accumulator(tensor):
        a = MeanAccumulator(tensor)
        return a.output_tensor, a.update_op, a.flush_op


b_size=19

A=tf.placeholder(tf.float32, (None,), name="A")
conced=None
summed=np.zeros(b_size)
meaned=MeanAccumulator(tf.reduce_mean(A))
with tf.Session() as sess:
    sess.run([tf.global_variables_initializer(),tf.local_variables_initializer()])
    for i in np.arange(11):
        cctensor=np.arange(i*b_size,(i+1)*b_size)
        feed={A:cctensor}
        summed+=cctensor
        if conced is None:
            conced=cctensor
        else:
            conced=np.concatenate([conced, cctensor])
        sess.run(meaned.update_op, feed)
    print("True summed:", summed)
    print("pers. summed:", sess.run(meaned.accumulator))
    print("Concatenated tensor:", conced)
    print("Acc mean:", sess.run(meaned.output_tensor))
    print("True mean:", np.mean(conced))
