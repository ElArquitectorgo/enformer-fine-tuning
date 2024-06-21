import tensorflow as tf
from tqdm import tqdm
import matplotlib.pyplot as plt
import os

def get_last_epoch(file):
    return int(file[file.find('_') + 1 : file.find('-')])

def _reduced_shape(shape, axis):
  if axis is None:
    return tf.TensorShape([])
  return tf.TensorShape([d for i, d in enumerate(shape) if i not in axis])


class CorrelationStats(tf.keras.metrics.Metric):
  """Contains shared code for PearsonR and R2."""

  def __init__(self, reduce_axis=None, name='pearsonr'):
    """Pearson correlation coefficient.

    Args:
      reduce_axis: Specifies over which axis to compute the correlation (say
        (0, 1). If not specified, it will compute the correlation across the
        whole tensor.
      name: Metric name.
    """
    super(CorrelationStats, self).__init__(name=name)
    self._reduce_axis = reduce_axis
    self._shape = None  # Specified in _initialize.

  def _initialize(self, input_shape):
    # Remaining dimensions after reducing over self._reduce_axis.
    self._shape = _reduced_shape(input_shape, self._reduce_axis)

    weight_kwargs = dict(shape=self._shape, initializer='zeros')
    self._count = self.add_weight(name='count', **weight_kwargs)
    self._product_sum = self.add_weight(name='product_sum', **weight_kwargs)
    self._true_sum = self.add_weight(name='true_sum', **weight_kwargs)
    self._true_squared_sum = self.add_weight(name='true_squared_sum',
                                             **weight_kwargs)
    self._pred_sum = self.add_weight(name='pred_sum', **weight_kwargs)
    self._pred_squared_sum = self.add_weight(name='pred_squared_sum',
                                             **weight_kwargs)

  def update_state(self, y_true, y_pred, sample_weight=None):
    """Update the metric state.

    Args:
      y_true: Multi-dimensional float tensor [batch, ...] containing the ground
        truth values.
      y_pred: float tensor with the same shape as y_true containing predicted
        values.
      sample_weight: 1D tensor aligned with y_true batch dimension specifying
        the weight of individual observations.
    """
    if self._shape is None:
      # Explicit initialization check.
      self._initialize(y_true.shape)
    y_true.shape.assert_is_compatible_with(y_pred.shape)
    y_true = tf.cast(y_true, 'float32')
    y_pred = tf.cast(y_pred, 'float32')

    self._product_sum.assign_add(
        tf.reduce_sum(y_true * y_pred, axis=self._reduce_axis))

    self._true_sum.assign_add(
        tf.reduce_sum(y_true, axis=self._reduce_axis))

    self._true_squared_sum.assign_add(
        tf.reduce_sum(tf.math.square(y_true), axis=self._reduce_axis))

    self._pred_sum.assign_add(
        tf.reduce_sum(y_pred, axis=self._reduce_axis))

    self._pred_squared_sum.assign_add(
        tf.reduce_sum(tf.math.square(y_pred), axis=self._reduce_axis))

    self._count.assign_add(
        tf.reduce_sum(tf.ones_like(y_true), axis=self._reduce_axis))

  def result(self):
    raise NotImplementedError('Must be implemented in subclasses.')

  def reset_states(self):
    if self._shape is not None:
      tf.keras.backend.batch_set_value([(v, np.zeros(self._shape))
                                        for v in self.variables])


class PearsonR(CorrelationStats):
  """Pearson correlation coefficient.

  Computed as:
  ((x - x_avg) * (y - y_avg) / sqrt(Var[x] * Var[y])
  """

  def __init__(self, reduce_axis=(0,), name='pearsonr'):
    """Pearson correlation coefficient.

    Args:
      reduce_axis: Specifies over which axis to compute the correlation.
      name: Metric name.
    """
    super(PearsonR, self).__init__(reduce_axis=reduce_axis,
                                   name=name)

  def result(self):
    true_mean = self._true_sum / self._count
    pred_mean = self._pred_sum / self._count

    covariance = (self._product_sum
                  - true_mean * self._pred_sum
                  - pred_mean * self._true_sum
                  + self._count * true_mean * pred_mean)

    true_var = self._true_squared_sum - self._count * tf.math.square(true_mean)
    pred_var = self._pred_squared_sum - self._count * tf.math.square(pred_mean)
    tp_var = tf.math.sqrt(true_var) * tf.math.sqrt(pred_var)
    correlation = covariance / tp_var

    return correlation


class R2(CorrelationStats):
  """R-squared  (fraction of explained variance)."""

  def __init__(self, reduce_axis=None, name='R2'):
    """R-squared metric.

    Args:
      reduce_axis: Specifies over which axis to compute the correlation.
      name: Metric name.
    """
    super(R2, self).__init__(reduce_axis=reduce_axis,
                             name=name)

  def result(self):
    true_mean = self._true_sum / self._count
    total = self._true_squared_sum - self._count * tf.math.square(true_mean)
    residuals = (self._pred_squared_sum - 2 * self._product_sum
                 + self._true_squared_sum)

    return tf.ones_like(residuals) - residuals / total


class MetricDict:
  def __init__(self, metrics):
    self._metrics = metrics

  def update_state(self, y_true, y_pred):
    for k, metric in self._metrics.items():
      metric.update_state(y_true, y_pred)

  def result(self):
    return {k: metric.result() for k, metric in self._metrics.items()}
  
  
def evaluate_model_R2(model, dataset, max_steps=None):
  metric = MetricDict({'R2': R2(reduce_axis=(0,1))})
  @tf.function
  def predict(x):
    return model(x, is_training=False)

  for i, batch in tqdm(enumerate(dataset)):
    if max_steps is not None and i > max_steps:
      break
    metric.update_state(batch['target'], predict(batch['sequence']))

  return metric.result()

def evaluate_model_PearsonR(model, dataset, max_steps=None):
  metric = MetricDict({'PearsonR': PearsonR(reduce_axis=(0,1))})
  @tf.function
  def predict(x):
    return model(x, is_training=False)

  for i, batch in tqdm(enumerate(dataset)):
    if max_steps is not None and i > max_steps:
      break
    metric.update_state(batch['target'], predict(batch['sequence']))

  return metric.result()

def plot_losses(train_losses, val_metrics, output_dir, epoch, organism):
  plt.plot(train_losses, label='Training Loss')
  plt.plot(val_metrics, label='Validation Score')
  plt.xlabel('Epoch')
  plt.legend()
  
  output_file = os.path.join(output_dir, f'{organism}_epoch_{epoch}_loss.png')
  plt.savefig(output_file)
  
  plt.clf()

def plot_both_losses(h_train_losses, h_val_metrics, m_train_losses, m_val_metrics, output_dir, epoch, config):
  plt.plot(h_train_losses, label='Human Training Loss')
  plt.plot(h_val_metrics, label='Human Validation Score')
  plt.plot(m_train_losses, label='Mouse Training Loss')
  plt.plot(m_val_metrics, label='Mouse Validation Score')
  plt.xlabel('Epoch')
  plt.legend()
  
  output_file = os.path.join(output_dir, f'{config}_epoch_{epoch}_loss.png')
  plt.savefig(output_file)
  
  plt.clf()

def evaluate_extended_model(model, dataset, corr, head=None, max_steps=None):
  if corr == 'PearsonR':
    metric = MetricDict({'PearsonR': PearsonR(reduce_axis=(0,1))})
  else:
    metric = MetricDict({'R2': R2(reduce_axis=(0,1))})
  
  if head != None:
    @tf.function
    def predict(x):
      return model(x, is_training=False)[head]
  else:
    @tf.function
    def predict(x):
      return model(x, is_training=False)

  for i, batch in tqdm(enumerate(dataset)):
    if max_steps is not None and i > max_steps:
      break
    metric.update_state(batch['target'], predict(batch['sequence']))

  return metric.result()