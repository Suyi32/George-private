import tensorflow as tf


class LinearScheduler:
    def __init__(self, initial_value, final_step, name):
        self.final_step = final_step
        self.initial_value = initial_value
        # self.variable = tf.Variable(initial_value, name=name)
        # self.decayed_ph = tf.placeholder(tf.float32)
        # self.decay_op = self.variable.assign(self.decayed_ph)
        self.name = name
        self.value = initial_value

    def decay(self, step):
        # decay = 1.0 - (float(step) / self.final_step)
        # if decay < 0.0:
        #     decay = 0.0
        # feed_dict = {self.decayed_ph: decay * self.initial_value}
        # tf.get_default_session().run(self.decay_op, feed_dict=feed_dict)
        self.value = max(1.0 - float(step) / self.final_step, 0.0) * self.initial_value

    def get_variable(self):
        # self.variable = tf.Print(self.variable, [self.variable], "{}: ".format(self.name))
        # return self.variable
        # print("{}: {}".format(self.name, self.value))
        return self.value