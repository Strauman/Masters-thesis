
from .. import trainhelp as hlp

from .. import color_print
from .. import util
import tensorflow as tf
from . import Trainable
from ..configs import COMBO_CFG
import time
from .trainable import StopTraining, EarlyStopping, FinishedEpoch
from ..trainhelp import ensure_list
import inspect

def combo_apply(f):
    def wrapper(self,adversarials, *args,**kwargs):
        adversarials=ensure_list(adversarials)
        rv=[]
        for adv in adversarials:
            rv.append(f(self, *args, p=adv, **kwargs))
        return rv
    return wrapper
capl=combo_apply
class SequentialAlias():#pylint: disable=R0903
    def __init__(self, arg_func=None):
        """
        arg_func: function that will generate args or kwargs to be sent to the function,
                  e.g.
                  some_fn=SequentialAlias(lambda trable: trable.info)
                will call trable.some_fn(trable.info) on all adv.
        """
        self.arg_func=None

    def __call__(self,*args,**kwargs):#pylint: disable=W0235
        return super(SequentialAlias,self).__call__(*args,**kwargs)

class SequentialHandler:#pylint: disable=R0903
    def __init__(self, *args, **kwargs):
        super(SequentialHandler,self).__init__(*args, **kwargs)
        for fun_name, attr_inst in inspect.getmembers(self):
            if isinstance(attr_inst, SequentialAlias):
                try:
                    arg_func=attr_inst.arg_func
                    setattr(self, fun_name, lambda adversarials, *args, sh=self, fun_name=fun_name,arg_func=arg_func,**kwargs: sh._str_combo_apply(fun_name, arg_func,adversarials, *args,**kwargs))#pylint: disable=E0602
                except Exception as e:
                    print(f"Can't set {fun_name}?")
                    raise

    def _str_combo_apply(self, attr, arg_func,adversarials, *args, **kwargs):
        rv=[]
        adversarials=ensure_list(adversarials)
        # adversarials=[args[0]]
        # if not isinstance(adversarials, (list, tuple)):
        #     adversarials=[adversarials]
        for adv in adversarials:
            send_kwargs=kwargs
            send_args=list(args)+[]
            if callable(arg_func):
                gen_args=arg_func(adv)
                if isinstance(gen_args, dict):
                    send_kwargs=merge_dicts(kwargs, gen_args)
                elif isinstance(gen_args, (tuple, list)):
                    send_args.append(gen_args)
            rv.append(getattr(adv, attr)(*send_args,**send_kwargs))
        return rv
from typing import Callable
def merge_dicts(template,update):
    c={}
    c.update(template)
    c.update(update)
    return c
def dummy():
    pass
class ComboTrainer(SequentialHandler):
    c_opts=None # type: COMBO_CFG
    def __init__(self, discriminator:Trainable,target_map:Trainable,combo_settings: COMBO_CFG, sess):
        super(ComboTrainer, self).__init__()
        options=dict(
            step=0,
            printerval=100
        )
        # c_opts={}
        # c_opts.update(options)
        # c_opts.update(combo_settings)
        # self.c_opts=

        c_opts=util.Dottify(merge_dicts(options,combo_settings.__dict__))
        # For keeping type hinting alive
        setattr(self, "c_opts", c_opts)
        self.c_opts #type: COMBO_SETTINGS
        self.prev_print=0 if self.c_opts.printerval_seconds else -2*self.c_opts.printerval
        self.prev_write=0 if self.c_opts.printerval_seconds else -2*self.c_opts.write_out_interval
        self.discriminator=discriminator
        self.target_map=target_map
        self.adversarials=[self.discriminator,self.target_map]
        self._curr_trains=self.adversarials
        self.info = util.Dottify(step=1, epoch=1, local_epoch=1, local_step=0)
        self._sess=sess
    # @property
    # def info(self):
    # @capl
    # def cb_init(self,adv, *args, **kwargs):
    #
    #     self.discriminator.cb_init(self.discriminator.info,*args,**kwargs)
    #     self.target_map.cb_init(self.target_map.info,*args,**kwargs)
    cb_init=SequentialAlias()
    cb_iter=SequentialAlias()
    cb_epoch=SequentialAlias()
    cb_printer=SequentialAlias()
    update_state=SequentialAlias()
    #pylint: disable=R0201
    @combo_apply
    def train_init(self,p: Trainable=None):
        p.cb_epoch()
        p.info.local_step = 1
        p.info.local_epoch = 1
        p.local_init_loss = None
    @combo_apply
    def step_iter(self,p: Trainable=None):
        pass
        # p.info.step += 1
        # p.info.local_step+=1
    @combo_apply
    def offer_optimizer_reset(self,p: Trainable=None):
        if p.opts.reset_optimizers:
            color_print(f"Resetting optimizer variables for {p.name}",style="warning")
            p._sess.run([tf.variables_initializer(p._optimizer.variables())])# pylint: disable=W0212

    def check_optimizer_reset(self):
        # print(f"DStep:{self.discriminator.info.step} -- {self.discriminator.info.local_step}")
        if self.c_opts.reset_tm_optimizer_steps > 0 and (self.target_map.info.step-self.last_tm_reset) >= self.c_opts.reset_tm_optimizer_steps:
            self.last_tm_reset=self.target_map.info.step
            color_print("Resetting target_map optimizer", style="danger")
            self.target_map.reset_optimizer()
            if self.c_opts.tm_steps_after_reset is not None and self.c_opts.tm_steps_after_reset > 0:
                self.last_tm_reset+=self.c_opts.tm_steps_after_reset
                self.__train_steps(self.target_map, self.c_opts.tm_steps_after_reset)
        if self.c_opts.reset_disc_optimizer_steps > 0 and (self.discriminator.info.step-self.last_disc_reset) >= self.c_opts.reset_disc_optimizer_steps:
            self.last_disc_reset=self.discriminator.info.step
            color_print("Resetting discriminator optimizer", style="danger")
            self.discriminator.reset_optimizer()
            if self.c_opts.disc_steps_after_reset is not None and self.c_opts.disc_steps_after_reset > 0:
                self.last_disc_reset+=self.c_opts.disc_steps_after_reset
                self.__train_steps(self.discriminator, self.c_opts.disc_steps_after_reset)

    def maybe_write(self,adversarials):
        # Printerval time or steps?
        self.check_optimizer_reset()
        cond_name="dt" if self.c_opts.printerval_seconds else "dstep"
        if self.c_opts.printerval_seconds:
            delta_print=time.time() - self.prev_print
            delta_write=time.time() - self.prev_write
        else:
            delta_print=self.info.step - self.prev_print
            delta_write=self.info.step - self.prev_write
        do_print=(delta_print > self.c_opts.printerval)
        do_write=(delta_write > self.c_opts.write_out_interval)

        if do_print or do_write:
            now= time.time() if self.c_opts.printerval_seconds else self.info.step
            self.prev_write=now
            if do_print:
                print(f"{cond_name} print={delta_print}, {cond_name} write={delta_write}")
                print("-"*10)
                self.prev_print=now

            self.cb_printer(adversarials, do_print=do_print)

    def iter_ad(self,adversarials,printer=True):
        ad=ensure_list(adversarials)
        self.cb_iter(ad)
        # self.step_iter(ad)
        # if printer:
            # self.maybe_write(ad)
        self.info.step+=1
    def simul_iter(self):
        disc_iter_ops=self.discriminator.iter_op
        tm_iter_ops=self.target_map.iter_op
        ops_ordered=[disc_iter_ops[0],tm_iter_ops[0],disc_iter_ops[1:],tm_iter_ops[1:]]
        try:
            _tr1,_tr2,out_ops_d,out_ops_tm = self._sess.run(ops_ordered, self.target_map.feeds_tr())
            self.discriminator.register_cb_iter(*out_ops_d)
            self.target_map.register_cb_iter(*out_ops_tm)
            # self.discriminator.register_cb_iter(out_ops[2], out_ops[3])
            # self.target_map.register_cb_iter(out_ops[4], out_ops[5])
        except tf.errors.OutOfRangeError:
            raise FinishedEpoch("No one dealt with finished epoch, so just quitting", trainable=None)




    def do_iter(self, trainable, printer=True):
        trainable.cb_iter()
        # trainable.info.step += 1
        # trainable.info.local_step+=1
        if printer:
            self.maybe_write(trainable)
        self.info.step+=1

    def __train_steps(self, adversarial, steps, printer=True):
        self.cb_epoch(adversarial)
        for _ in range(steps):
            try:
                self.iter_ad(adversarial,printer=printer)
            except FinishedEpoch as fe:
                # adversarial.init_its_tr()
                # fe.trainable.init_its_tr()
                fe.trainable.cb_epoch()
                # self.cb_epoch(adversarial)

                pass
        self.cb_printer(adversarial)
        return adversarial

    def __train(self,adversarials):
        """ Trains adversarials for one epoch """
        did_print_err = False
        ad=ensure_list(adversarials)
        finished_epoch_trainables=[]
        differ_step_training=self.c_opts.disc_tm_train_epoch_count==(1,1)
        first_adv=self.target_map
        last_adv=self.discriminator
        if self.c_opts.disc_train_first:
            first_adv,last_adv=last_adv,first_adv
        def on_finish_epoch(fe):
            pass
            # print(f"Finished {fe.trainable.name}")
            # print(f"Got {self.target_map.name} and {self.discriminator.name}")
            # if fe.trainable not in finished_epoch_trainables:
                # finished_epoch_trainables.append(fe.trainable)
            # fe.trainable.cb_epoch()

        def try_iter(ad):
            try:
                ad.cb_iter()
                # ad.step_iter()
                # if printer:
                    # self.maybe_write(ad)
                self.info.step+=1
            except FinishedEpoch as fe:
                # on_finish_epoch(fe)
                # self.cb_epoch()
                ad.cb_epoch()
                # self.cb_epoch(adversarial)
                # return len(finished_epoch_trainables)<2
                return True
            return True

        def try_iter_steps(ad,steps):
            for s in range(steps):
                keep_on=try_iter(ad)
                self.info.step+=1
                if not keep_on:
                    return False
            return True
        if differ_step_training and len(adversarials)==1:
            color_print("Cannot differ step training when only one adversarial given", style="warning")
        elif differ_step_training:
            disc_steps,tm_steps=self.c_opts.disc_tm_train_step_count
            class Stepper(util.Dottify):
                trainable=None
                steps=None
                cluster_reset=None
                cluster_reset_after=None
            disc_stepper=Stepper(trainable=self.discriminator, steps=disc_steps, cluster_reset=self.c_opts.reset_disc_optimizer_after_cluster_step, cluster_reset_after=self.c_opts.cluster_reset_after) # type: Stepper
            target_stepper=Stepper(trainable=self.target_map,steps=tm_steps,cluster_reset=self.c_opts.reset_tm_optimizer_after_cluster_step, cluster_reset_after=self.c_opts.cluster_reset_after)

            if (disc_steps,tm_steps)==(0,0): raise ValueError("Can't train when no steps on disc or tm")
            first_stepper, last_stepper=(target_stepper, disc_stepper)
            if self.c_opts.disc_train_first:
                first_stepper, last_stepper=last_stepper,first_stepper
            if disc_steps==tm_steps==1:
                differ_step_training=False
            # print(first_stepper[1], last_stepper[1])
            while True:
                if differ_step_training:
                    # with tf.device('/gpu:0'):
                    # with tf.device('/gpu:1'):
                    steppers=[first_stepper,last_stepper]
                    for stepper in steppers:
                        if stepper.steps>0:
                            if stepper.cluster_reset and not stepper.cluster_reset_after:
                                tf.get_default_session().run([tf.variables_initializer(stepper.trainable._optimizer.variables())])
                            keep_on=try_iter_steps(stepper.trainable,stepper.steps)
                            if stepper.cluster_reset and stepper.cluster_reset_after:
                                tf.get_default_session().run([tf.variables_initializer(stepper.trainable._optimizer.variables())])
                            # if not keep_on:
                                # break
                            # stepper.trainable.cb_epoch()
                    if self.c_opts.cluster_reset_both_after:
                        tf.get_default_session().run([
                            tf.variables_initializer(first_stepper.trainable._optimizer.variables()),
                            tf.variables_initializer(last_stepper.trainable._optimizer.variables())
                        ])

                    self.maybe_write([first_stepper.trainable, last_stepper.trainable])
                else:
                    if self.c_opts.simultaneous_iteration:
                        try:
                            self.simul_iter()
                        except FinishedEpoch as fe:
                            self.discriminator.cb_epoch()
                            self.target_map.cb_epoch()
                            break
                    else:
                        keep_on=try_iter(first_stepper.trainable)
                        if not keep_on:
                            break
                        keep_on=try_iter(last_stepper.trainable)
                        if not keep_on:
                            break
                    self.maybe_write([first_stepper.trainable, last_stepper.trainable])
            self.maybe_write([first_stepper.trainable, last_stepper.trainable])
                    # else:
                        # fe.trainable.init_its_tr()





        # self.step_epoch(ad)

    def _train(self):
        self.info.step=0
        self.last_tm_reset=0
        self.last_disc_reset=0
        # Init all adversaries. Once.
        self.cb_init(self.adversarials)
        disc_count,tm_count=self.c_opts.disc_tm_train_epoch_count
        disc_train_initial=self.c_opts.initial_disc_train_steps
        initial_tm_train_steps=self.c_opts.initial_tm_train_steps
        disc_train_first=self.c_opts.disc_train_first
        tm_init_stepper=(initial_tm_train_steps,self.target_map, initial_tm_train_steps)
        disc_init_stepper=(disc_train_initial, self.discriminator, disc_train_initial)
        first_init_stepper,last_init_stepper=tm_init_stepper,disc_init_stepper
        if self.c_opts.disc_train_first:
            first_init_stepper,last_init_stepper=last_init_stepper,first_init_stepper
        if first_init_stepper[0]>0:
            self.__train_steps(*first_init_stepper[1:])
        if last_init_stepper[0]>0:
            self.__train_steps(*last_init_stepper[1:])
        while True:
            if disc_count==tm_count==1:
                self.__train(self.adversarials)
            else:
                self.train_init(self.adversarials)
                ad=self.discriminator
                for d in range(disc_count):
                    print(f"Disc {d}")
                    try:
                        self.__train(ad)
                    except EarlyStopping:
                        self.update_state(ad)
                        break
                    self.update_state(ad)
                ad=self.target_map
                for tm in range(tm_count):
                    print(f"tm {tm}")
                    try:
                        self.__train(ad)
                    except EarlyStopping:
                        self.update_state(ad)
                        break
                self.update_state(ad)
            adsteps=[a.info.step for a in self.adversarials]

            if self.c_opts.max_steps is not None and self.c_opts.max_steps > 0 and sum(adsteps) >= self.c_opts.max_steps:
                adplus='+'.join([str(ast) for ast in adsteps])
                raise EarlyStopping(f"Reached max number of combo steps: {adplus}={sum(adsteps)} > {self.c_opts.max_steps}")
    def train(self):
        try:
            self._train()
        except KeyboardInterrupt as e:
            print("KeyboardInterrupt")
            self.update_state(self.adversarials)
            time.sleep(0.7)
            return
        except StopTraining as e:
            print("Requested to stop training")
            print(str(e))
            self.update_state(self.adversarials)
            return
