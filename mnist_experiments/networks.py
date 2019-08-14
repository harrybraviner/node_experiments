import tensorflow as tf
import numpy as np
from abc import ABC, abstractmethod


class EWMA:
    """
    Exponential Weighted Moving Average.
    Helper class for tracking smoothed quantities.
    """
    def __init__(self, decay_lambda: float=0.1):
        self._decay_lambda = decay_lambda
        self._running_ewma = None

    def update(self, x: float):

        if self._running_ewma is None:
            self._running_ewma = x
        else:
            self._running_ewma *= 1.0 - self._decay_lambda
            self._running_ewma += self._decay_lambda * x

    def get(self):
        return self._running_ewma


class BaseNetwork:

    def __init__(self, img_h: int=28, img_w: int=28, num_classes: int=10):
        # Placeholders
        self._input_img_batch = tf.placeholder(shape=(None, img_h, img_w, 1), dtype=tf.float32)
        self._input_ground_truth_oh= tf.placeholder(shape=(None, num_classes), dtype=tf.float32)

        self._batch_size = tf.shape(self._input_img_batch)[0]

        # Delegate building the actual network to a method
        self._output_logits = self._get_logits(self._input_img_batch)

        # Define training loss and accuracy for evaluation
        self.ce_loss = tf.losses.softmax_cross_entropy(logits=self._output_logits, onehot_labels=self._input_ground_truth_oh)
        
        # Accuracy, for evaluation
        self._output_class = tf.math.argmax(self._output_logits, axis=1)
        self._gt_class = tf.math.argmax(self._input_ground_truth_oh, axis=1)
        self._correct_predictions = tf.math.equal(self._output_class, self._gt_class)
        self._acc = tf.reduce_mean(tf.cast(self._correct_predictions, dtype=tf.float32), axis=0)
        
        self._train_step = tf.train.MomentumOptimizer(learning_rate=1e-2, momentum=0.9).minimize(self.ce_loss)
    
        self._sess = tf.Session()
        self._sess.run(tf.global_variables_initializer())
        
        # Track training history
        self._running_ce = EWMA(decay_lambda=0.01)
        self._ce_history = []
        self._running_acc = EWMA(decay_lambda=0.01)
        self._acc_history = []

    @abstractmethod
    def _get_logits(self, input_img: tf.Tensor) -> tf.Tensor:
        raise NotImplementedError('Abstract method _get_logits is not implemented.')

    def train_batch(self, input_images, input_gt_oh):
        
        _, ce_this_batch, acc_this_batch = self._sess.run([self._train_step, self.ce_loss, self._acc],
                                                          feed_dict={self._input_img_batch: input_images,
                                                                     self._input_ground_truth_oh: input_gt_oh})
        
        self._running_ce.update(ce_this_batch)
        self._ce_history.append(self._running_ce.get())
        self._running_acc.update(acc_this_batch)
        self._acc_history.append(self._running_acc.get())

    def evaluate_on_batch(self, input_images, input_gt_oh):

        ce_this_batch, acc_this_batch, size_this_batch = self._sess.run([self.ce_loss, self._acc, self._batch_size],
                                                                        feed_dict={self._input_img_batch: input_images,
                                                                                   self._input_ground_truth_oh: input_gt_oh})
        return size_this_batch, ce_this_batch, acc_this_batch


def res_block(input_tensor: tf.Tensor, in_filters: int, out_filters: int, stride=1) -> tf.Tensor:
    """
    Helper method for constructing networks.
    Constructs a residual block that first applies activation, then has two paths:
        1) Downsamples only by 1x1 conv, then feeds to end.
        2) Applies 2 3x3 conv filters. Only first of these has an activation function.
    """
    # FIXME - may want to apply batch-norm here
    t = tf.nn.relu(input_tensor)

    # Downsample for the skip connection
    # FIXME - should this start close to an identity transformation?
    kernel_ds_xavier_range = np.sqrt(6) / (np.sqrt(1*1*in_filters) + np.sqrt(out_filters))
    kernel_ds = tf.Variable(tf.random.uniform(shape=(1, 1, in_filters, out_filters), dtype=tf.float32,
                                              minval=-kernel_ds_xavier_range, maxval=+kernel_ds_xavier_range))
    t_downsampled = tf.nn.conv2d(t, kernel_ds, strides=(1, stride, stride, 1), padding='SAME')

    kernel_c1_range = np.sqrt(6) / (np.sqrt(3*3*in_filters) + np.sqrt(out_filters))
    kernel_c1 = tf.Variable(tf.random.uniform(shape=(3, 3, in_filters, out_filters), dtype=tf.float32,
                                              minval=-kernel_c1_range, maxval=+kernel_c1_range))
    t = tf.nn.conv2d(t, kernel_c1, strides=(1, stride, stride, 1), padding='SAME')
    # FIXME - should also apply batch norm here to match Chen et al.
    t = tf.nn.relu(t)

    kernel_c2_range = np.sqrt(6) / (np.sqrt(3*3*out_filters) + np.sqrt(out_filters))
    kernel_c2 = tf.Variable(tf.random.uniform(shape=(3, 3, out_filters, out_filters), dtype=tf.float32,
                                             minval=-kernel_c2_range, maxval=+kernel_c2_range))
    t = tf.nn.conv2d(t, kernel_c2, strides=(1, 1, 1, 1), padding='SAME')

    # Skip connection
    # N.B. No activation function. This is the sum of the output of two convolutions (one 3x3, the other 1x1).
    return t + t_downsampled


def downsample_net(input_tensor: tf.Tensor, n_filters: int) -> tf.Tensor:
    """
    Common network to all of the methods in the experiment in Chen et al.
    Used first to downsample the image by a factor of 4 in each dimension.
    """
    # Convolution with 3x3 kernels
    # N.B. Chen et al. do use biases in this conv.
    kernel_range = np.sqrt(6) / (np.sqrt(3*3*1) + np.sqrt(n_filters))
    kernel = tf.Variable(tf.random.uniform(shape=(3, 3, 1, n_filters), dtype=tf.float32,
                                           minval=-kernel_range, maxval=+kernel_range))
    biases = tf.Variable(tf.zeros(shape=(1, 1, 1, n_filters), dtype=tf.float32))
    conv_output = tf.nn.conv2d(input_tensor, kernel, strides=(1, 1, 1, 1), padding='SAME') + biases

    return res_block(res_block(conv_output, in_filters=n_filters, out_filters=n_filters, stride=2), in_filters=n_filters, out_filters=n_filters, stride=2)


class FCNet(BaseNetwork):

    def __init__(self, *args, **kwargs):
        super(FCNet, self).__init__(*args, **kwargs)

    def _get_logits(self, input_img: tf.Tensor) -> tf.Tensor:
        img_h, img_w = input_img.shape.as_list()[1:3]   # Get image dimensions

        downsampled_img = downsample_net(self._input_img_batch, n_filters=64)
        dim_ds_img = (img_h//4)*(img_w//4)*64

        flattened_ds_img = tf.reshape(downsampled_img, shape=(-1, dim_ds_img))
        flattened_act = tf.nn.relu(flattened_ds_img)

        W_range = np.sqrt(6) / (np.sqrt(dim_ds_img) + np.sqrt(10))
        W_fc = tf.Variable(tf.random.uniform(shape=(dim_ds_img, 10), dtype=tf.float32,
                                             minval=-W_range, maxval=+W_range), name='W_fc')
        b_fc = tf.Variable(tf.zeros(shape=(10), dtype=tf.float32))
        return tf.matmul(flattened_act, W_fc)


class ResNet6(BaseNetwork):
    """
    Network used by Chen et al. in their paper.
    Uses 6 residual blocks after the downsampling layer.
    """

    def __init__(self, *args, **kwargs):
        super(ResNet6, self).__init__(*args, **kwargs)

    def _get_logits(self, input_img: tf.Tensor) -> tf.Tensor:
        img_h, img_w = input_img.shape.as_list()[1:3]   # Get image dimensions

        downsampled_img = downsample_net(self._input_img_batch, n_filters=64)
        dim_ds_img = (img_h//4)*(img_w//4)*64

        x = downsampled_img
        for _ in range(6):
            x = res_block(x, 64, 64)

        flattened_x = tf.reshape(x, shape=(-1, dim_ds_img))
        flattened_act = tf.nn.relu(flattened_x)

        W_range = np.sqrt(6) / (np.sqrt(dim_ds_img) + np.sqrt(10))
        W_fc = tf.Variable(tf.random.uniform(shape=(dim_ds_img, 10), dtype=tf.float32,
                                             minval=-W_range, maxval=+W_range), name='W_fc')
        b_fc = tf.Variable(tf.zeros(shape=(10), dtype=tf.float32))
        return tf.matmul(flattened_act, W_fc)

