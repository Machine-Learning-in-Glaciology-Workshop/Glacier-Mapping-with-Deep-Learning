import tensorflow as tf


class FocalLoss(tf.keras.losses.Loss):
    def __init__(
        self, gamma=2.0, alphas=None, reduction=tf.keras.losses.Reduction.AUTO, name="focal_loss"
    ):
        super(FocalLoss, self).__init__(reduction=reduction, name=name)
        self.gamma = gamma
        self.alphas = alphas

    def call(self, y_true, y_pred):
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.keras.backend.clip(y_pred, epsilon, 1 - epsilon)
        ce = y_true * tf.math.log(y_pred)
        loss = -tf.math.pow(1 - y_pred, self.gamma) * ce
        if self.alphas is not None:
            loss = self.alphas * loss
        return tf.math.reduce_mean(tf.math.reduce_sum(loss, axis=-1))
