import os
import re

import tensorflow as tf
import tensorflow_addons as tfa

from libs.lr_scheduler import WarmUpSchedule


def set_cuda_visible_device(ngpus):
    empty = []
    for i in range(4):
        os.system('nvidia-smi -i ' + str(i) + ' | grep "No running" | wc -l > empty_gpu_check')
        f = open('empty_gpu_check')
        out = int(f.read())
        if int(out) == 1:
            empty.append(i)
    if len(empty) < ngpus:
        print('avaliable gpus are less than required')
    cmd = ''
    for i in range(ngpus):
        cmd += str(empty[i]) + ','
    return cmd


def get_learning_rate_scheduler(lr_schedule='stair',
                                graph_dim=256,
                                warmup_steps=1000,
                                init_lr=1e-3,
                                decay_steps=500,
                                decay_rate=0.1,
                                staircase=True):
    scheduler = None
    if lr_schedule == 'warmup':
        scheduler = WarmUpSchedule(
            d_model=graph_dim,
            warmup_steps=warmup_steps
        )

    elif lr_schedule == 'stair':

        scheduler = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=init_lr,
            decay_steps=decay_steps,
            decay_rate=decay_rate,
            staircase=staircase
        )

    return scheduler


def get_loss_function(loss_type):
    loss_fn = None
    if loss_type == 'bce':
        loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=False)
    elif loss_type == 'mse':
        loss_fn = tf.keras.losses.MeanSquaredError()
    else:
        if 'focal' in loss_type:
            alpha = float(loss_type.split('_')[1])
            gamma = float(loss_type.split('_')[2])
            loss_fn = tfa.losses.SigmoidFocalCrossEntropy(
                from_logits=False, alpha=alpha, gamma=gamma
            )
    return loss_fn


class MultitaskLoss(tf.keras.losses.Loss):
    def __init__(self,
                 prop_list,
                 loss_dict):
        super(MultitaskLoss, self).__init__()
        self.num_props = len(prop_list)
        self.loss_fn_list = []
        for prop in prop_list:
            self.loss_fn_list.append(get_loss_function(loss_dict[prop]))

    def call(self, y_true, y_pred):
        loss = 0
        for i in range(self.num_props):
            loss += self.loss_fn_list[i](y_true[:, i], y_pred[:, i])
        return loss


def get_metric_list(loss_type):

    if loss_type in ['mse']:
        metrics = [
            tf.keras.metrics.MeanAbsoluteError(name='MAE'),
            tf.keras.metrics.RootMeanSquaredError(name='RMSE'),
        ]

    else:
        metrics = [
            tf.keras.metrics.BinaryAccuracy(name='Accuracy'),
            tf.keras.metrics.AUC(curve='ROC', name='AUROC'),
            tf.keras.metrics.AUC(curve='PR', name='AUPRC'),
            tf.keras.metrics.Precision(name='Precision'),
            tf.keras.metrics.Recall(name='Recall'),
        ]

    return metrics


def get_task_options(benchmark_task_type):
    if benchmark_task_type == 'reg':
        # last_activation, loss, metrics
        return None, tf.keras.losses.MeanSquaredError(), [tf.keras.metrics.MeanSquaredError()]
    else:
        return tf.keras.activations.sigmoid, tf.keras.losses.BinaryCrossentropy(), [tf.keras.metrics.BinaryAccuracy(),
                                                                              tf.keras.metrics.AUC()]


class CustomAdamW(tf.keras.optimizers.Optimizer):
  """A basic Adam optimizer that includes "correct" L2 weight decay."""

  def __init__(self,
               learning_rate,
               weight_decay_rate=0.0,
               beta_1=0.9,
               beta_2=0.999,
               epsilon=1e-6,
               name="CustomAdamW",
               exclude_from_weight_decay=None):
    """Constructs a AdamWeightDecayOptimizer."""
    super(CustomAdamW, self).__init__(name)

    self._name = name
    self.learning_rate = learning_rate
    self.weight_decay_rate = weight_decay_rate
    self.beta_1 = beta_1
    self.beta_2 = beta_2
    self.epsilon = epsilon
    self.exclude_from_weight_decay = exclude_from_weight_decay

  def get_config(self):
      """Returns the config of the optimimizer.
      An optimizer config is a Python dictionary (serializable)
      containing the configuration of an optimizer.
      The same optimizer can be reinstantiated later
      (without any saved state) from this configuration.
      Returns:
          Python dictionary.
      """
      config = {"name": self._name}
      if hasattr(self, "learning_rate"):
          config["learning_rate"] = self.learning_rate
      if hasattr(self, "weight_decay_rate"):
          config["weight_decay_rate"] = self.weight_decay_rate
      if hasattr(self, "beta_1"):
          config["beta_1"] = self.beta_1
      if hasattr(self, "beta_2"):
          config["beta_2"] = self.beta_2
      if hasattr(self, "epsilon"):
          config["epsilon"] = self.epsilon
      return config

  def apply_gradients(self, grads_and_vars, global_step=None, name=None):
    """See base class."""
    assignments = []
    for (grad, param) in grads_and_vars:
      if grad is None or param is None:
        continue

      param_name = self._get_variable_name(param.name)

      m = tf.compat.v1.get_variable(
          name=param_name + "/adam_m",
          shape=param.shape.as_list(),
          dtype=tf.float32,
          trainable=False,
          initializer=tf.zeros_initializer())
      v = tf.compat.v1.get_variable(
          name=param_name + "/adam_v",
          shape=param.shape.as_list(),
          dtype=tf.float32,
          trainable=False,
          initializer=tf.zeros_initializer())

      # Standard Adam update.
      next_m = (
          tf.multiply(self.beta_1, m) + tf.multiply(1.0 - self.beta_1, grad))
      next_v = (
          tf.multiply(self.beta_2, v) + tf.multiply(1.0 - self.beta_2,
                                                    tf.square(grad)))

      update = next_m / (tf.sqrt(next_v) + self.epsilon)

      # Just adding the square of the weights to the loss function is *not*
      # the correct way of using L2 regularization/weight decay with Adam,
      # since that will interact with the m and v parameters in strange ways.
      #
      # Instead we want ot decay the weights in a manner that doesn't interact
      # with the m/v parameters. This is equivalent to adding the square
      # of the weights to the loss with plain (non-momentum) SGD.
      if self._do_use_weight_decay(param_name):
        update += self.weight_decay_rate * param

      update_with_lr = self.learning_rate * update

      next_param = param - update_with_lr

      assignments.extend(
          [param.assign(next_param),
           m.assign(next_m),
           v.assign(next_v)])
    return tf.group(*assignments, name=name)

  def _do_use_weight_decay(self, param_name):
    """Whether to use L2 weight decay for `param_name`."""
    if not self.weight_decay_rate:
      return False
    if self.exclude_from_weight_decay:
      for r in self.exclude_from_weight_decay:
        if re.search(r, param_name) is not None:
          return False
    return True

  def _get_variable_name(self, param_name):
    """Get the variable name from the tensor name."""
    m = re.match("^(.*):\\d+$", param_name)
    if m is not None:
      param_name = m.group(1)
    return param_name