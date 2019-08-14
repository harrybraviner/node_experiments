import tensorflow as tf
from abc import ABC, abstractmethod

class BaseNetwork:

    def __init__(self, img_h: int=28, img_w: int=28, num_classes: int=10):
        # Placeholders
        self._input_img_batch = tf.placeholder(shape=(None, img_h, img_w, 1), dtype=tf.float32)
        self._input_ground_truth_oh= tf.placeholder(shape=(None, num_classes), dtype=tf.float32)

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
        
        self._running_ce = EWMA(decay_lambda=0.1)
        self._ce_history = []
        self._running_acc = EWMA(decay_lambda=0.1)
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

