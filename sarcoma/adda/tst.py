    # import tensorflow as tf
    # import numpy as np
    #
    #
    # def batch_persistent_mean(tensor):
    #     # Make a variable that keeps track of the sum
    #     accumulator = tf.Variable(initial_value=tf.zeros_like(tensor), dtype=tf.float32)
    #     # Keep count of batches in accumulator (needed to estimate mean)
    #     batch_nums = tf.Variable(initial_value=tf.zeros_like(tensor), dtype=tf.float32)
    #     # Make an operation for accumulating, increasing batch count
    #     accumulate_op = tf.assign_add(accumulator, tensor)
    #     step_batch = tf.assign_add(batch_nums, 1)
    #     update_op = tf.group([step_batch, accumulate_op])
    #     eps = 1e-5
    #     output_tensor = accumulator / (tf.nn.relu(batch_nums - eps) + eps)
    #     # In regards to the tf.nn.relu, it's a hacky zero_guard:
    #     # if batch_nums are zero then return eps, else it'll be batch_nums
    #     # Make an operation to reset
    #     flush_op = tf.group([tf.assign(accumulator, 0), tf.assign(batch_nums, 0)])
    #     return output_tensor, update_op, flush_op
    #
    # # Make a variable that we want to accumulate
    # X = tf.Variable(0., dtype=tf.float32)
    # # Make our persistant mean operations
    # Xbar, upd, flush = batch_persistent_mean(X)
    #
    # sess = tf.InteractiveSession()
    # with tf.Session() as sess:
    #     sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
    #     # Calculate the mean of 1+2+...+20
    #     for i in range(20):
    #         sess.run(upd, {X: i})
    #     print(sess.run(Xbar), "=", np.mean(np.arange(20)))
    #     for i in range(40):
    #         sess.run(upd, {X: i})
    #     # Now Xbar is the mean of (1+2+...+20+1+2+...+40):
    #     print(sess.run(Xbar), "=", np.mean(np.concatenate([np.arange(20), np.arange(40)])))
    #     # Now flush it
    #     sess.run(flush)
    #     print("flushed. Xbar=", sess.run(Xbar))
    #     for i in range(40):
    #         sess.run(upd, {X: i})
    #     print(sess.run(Xbar), "=", np.mean(np.arange(40)))
# import tensorflow as tf
# import numpy as np
# y=np.concat([np.ones(10), np.zeros(10)])
# yhat=np.ones(20)
# # Y=tf.placeholder(tf.int, [20])
# # Yhat=tf.placeholder(tf.int, [20])
# logits = tf.placeholder(tf.int64, [2,3])
# labels = tf.Variable([[0, 1, 0], [1, 0, 1]])
# acc_sum=tf.Variable("acc_sum")
#
# acc, acc_op = tf.metrics.accuracy(labels=tf.argmax(labels, 1), predictions=tf.argmax(logits,1))
# import asyncio
# async def repeat(interval, func, *args, **kwargs):
#     """Run func every interval seconds.
#
#     If func has not finished before *interval*, will run again
#     immediately when the previous iteration finished.
#
#     *args and **kwargs are passed as the arguments to func.
#     """
#     while True:
#         await asyncio.gather(
#             func(*args, **kwargs),
#             asyncio.sleep(interval),
#         )
# async def f():
#     await asyncio.sleep(1)
#     print('Hello')
#
#
# async def g():
#     await asyncio.sleep(0.5)
#     print('Goodbye')
#
#
# async def main():
#     t1 = asyncio.ensure_future(repeat(3, f))
#     t2 = asyncio.ensure_future(repeat(2, g))
#     await t1
#     await t2
#
#
# loop = asyncio.get_event_loop()
# loop.run_until_complete(main())
# print("Hello there!")
#
# class _Filter(object):
#     """docstring for _Filter."""
#     def __init__(self, f):
#         self.f=f
#     def __call__(self, arrs, idx=0, n_out=-1, pool_size=None, **kwargs):
#         self.pool_size=pool_size
#         return self.f(self, arrs=arrs, idx=idx, n_out=n_out, pool_size=pool_size, **kwargs)
#
# @_Filter
# def my_filter_func(self, arrs, idx, *args, **kwargs):
#     print(f"{arrs} {idx} ps:{self.pool_size}")
# # my_filter=_Filter(my_filter_func)
# # my_filter("hello", "world")
# my_filter_func("hello","world", pool_size="fifteen")
# from .configs import method_defs
# print(method_defs.DS)
from .helpers import util
def column_to_rows(columns):
    n_rows=len(columns[0])
    rows=[[] for _ in range(n_rows)]
    for i,c in enumerate(columns):
        for j,value in enumerate(c):
            rows[j].append(value)
    return rows
def DA_ROWS():
    # vals=set_state_values()
    cfg=util.Dottify(source_ds_name="SRC", target_ds_name="DST")
    vals=util.Dottify(
        initial=util.Dottify(f1="1", accuracy="2", auc="3"),
        latest=util.Dottify(f1="10", accuracy="20", auc="30")
    )
    # columns={'initial', 'latest'}
    columns=[]
    columns.append(["\\F1", "Accuracy", "AUC"])
    for state,v in vals.items():
        columns.append([v.f1, v.accuracy, v.auc])
    rows=column_to_rows(columns)
    rows=["&".join(r) for r in rows]
    rows=[r+"&" for r in rows]
    # Set the multicol
    multirow=f"\multirow{{{len(rows)}}}{{*}}{{{cfg.source_ds_name} $\\to$ {cfg.target_ds_name}}}"
    rows[0]=multirow+" & "+rows[0]+"\\TstrutM"
    rows[-1]=rows[-1]+"\\BstrutM"
    rows="\\\\\n".join(rows)
    rows+="\\\\\\sline"
    return rows

print(DA_ROWS())
