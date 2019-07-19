from ..trainables import _pretrain
from .. import metrics
import tensorflow as tf
from ..helpers.trainhelp import TF_Bestie
from sys import exit as xit
class Pretrain_LBL(_pretrain._Pretrain): # pylint: disable=W0212
    def build_graph(self):
        #pylint: disable=W0201
        self.average_losses=[]
        self.labels=tf.cast(self.Ys, tf.int32)
        pred=tf.argmax(self.Yhat,axis=1)
        print(self.Ys.shape, self.Yhat.shape)
        # self.labels=tf.Print(self.labels,[self.labels, self.Yhat], summarize=self.Yhat.shape[1])
        self.loss = tf.losses.sparse_softmax_cross_entropy(labels=self.labels, logits=self.Yhat)
        # xit()
        # labels=tf.Print(labels,[labels,self.Yhat])

        self.acc = tf.reduce_mean(tf.cast(tf.math.equal(self.labels, tf.cast(pred, tf.int32)), tf.float32))
        self.performance_measure=self.acc
        self.besties=[TF_Bestie(trainable=self, tf_op=self.acc, comparer=">", param_state="best", name="Best_accuracy", verbose=True)]

        self.optimizer=self.get_optimizer()
        self.minimizer=self.optimizer.minimize(self.loss+self.model_losses, global_step=self.global_step, var_list=self.train_vars)
        self.add_summary(self.performance_measure, "accuracy")
        self.add_summary(self.loss, "loss")
        # self.ignore_use(self.update_ops)
        # pp(self.update_ops)
        self.train_op = tf.group([self.minimizer, self.update_ops])
        # self.train_op = tf.group([self.minimizer])
