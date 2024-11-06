import tensorflow as tf
from keras import backend as K
from keras.layers import Layer
from keras.models import load_model
from keras import initializers

class AttentionWithContext(Layer):
    def __init__(self, **kwargs):
        super(AttentionWithContext, self).__init__(**kwargs)
        self.supports_masking = True
        self.bias = True

    def build(self, input_shape):
        self.W = self.add_weight(name='W',
                                 shape=(input_shape[-1], input_shape[-1]),
                                 initializer=initializers.get('glorot_uniform'))
        if self.bias:
            self.b = self.add_weight(name='b',
                                     shape=(input_shape[-1],),
                                     initializer='zeros')
        self.u = self.add_weight(name='u',
                                 shape=(input_shape[-1],),
                                 initializer=initializers.get('glorot_uniform'))
        super(AttentionWithContext, self).build(input_shape)

    def call(self, x, mask=None):
        uit = tf.tanh(tf.tensordot(x, self.W, axes=1))
        if self.bias:
            uit += self.b
        ait = tf.tensordot(uit, self.u, axes=1)
        a = tf.exp(ait)

        if mask is not None:
            mask = tf.cast(mask, tf.float32)
            a *= mask

        a /= tf.reduce_sum(a, axis=1, keepdims=True) + K.epsilon()
        a = tf.expand_dims(a, axis=-1)
        weighted_input = x * a
        return tf.reduce_sum(weighted_input, axis=1)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])

def f1(y_true, y_pred):
    y_pred = tf.round(y_pred)
    tp = tf.reduce_sum(tf.cast(y_true * y_pred, 'float32'), axis=1)
    precision = tp / (tf.reduce_sum(tf.cast(y_pred, 'float32'), axis=1) + K.epsilon())
    recall = tp / (tf.reduce_sum(tf.cast(y_true, 'float32'), axis=1) + K.epsilon())
    f1_val = 2 * precision * recall / (precision + recall + K.epsilon())
    return tf.reduce_mean(f1_val)

model = load_model('CPSC2018_10_fold_model_0', custom_objects={
    'AttentionWithContext': AttentionWithContext,
    'f1': f1
})
model.summary()
