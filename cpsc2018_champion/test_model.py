import tensorflow as tf
from keras import backend as K
from keras.layers import Layer
from keras.models import load_model
from keras import initializers, regularizers, constraints
import numpy as np

# 定義 AttentionWithContext 類別
class AttentionWithContext(Layer):
    def __init__(self,
                 W_regularizer=None, u_regularizer=None, b_regularizer=None,
                 W_constraint=None, u_constraint=None, b_constraint=None,
                 bias=True, **kwargs):
        self.supports_masking = True
        self.init = initializers.get('glorot_uniform')
        self.W_regularizer = regularizers.get(W_regularizer)
        self.u_regularizer = regularizers.get(u_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)
        self.W_constraint = constraints.get(W_constraint)
        self.u_constraint = constraints.get(u_constraint)
        self.b_constraint = constraints.get(b_constraint)
        self.bias = bias
        super(AttentionWithContext, self).__init__(**kwargs)

    def build(self, input_shape):
        # 定義權重 W, b 和 u
        self.W = self.add_weight(shape=(input_shape[-1], input_shape[-1]),
                                 initializer=self.init,
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint,
                                 name='{}_W'.format(self.name))
        if self.bias:
            self.b = self.add_weight(shape=(input_shape[-1],),
                                     initializer='zeros',
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint,
                                     name='{}_b'.format(self.name))
        self.u = self.add_weight(shape=(input_shape[-1],),
                                 initializer=self.init,
                                 regularizer=self.u_regularizer,
                                 constraint=self.u_constraint,
                                 name='{}_u'.format(self.name))
        super(AttentionWithContext, self).build(input_shape)

    def call(self, x, mask=None):
        # 計算注意力權重
        uit = K.tanh(K.dot(x, self.W))
        if self.bias:
            uit += self.b
        ait = tf.tensordot(uit, self.u, axes=[[2], [0]])
        a = K.exp(ait)
        if mask is not None:
            a *= K.cast(mask, K.floatx())
        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())
        a = K.expand_dims(a)
        weighted_input = x * a
        return K.sum(weighted_input, axis=1)

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[-1]

# 定義 f1 指標
def f1(y_true, y_pred):
    y_pred = K.round(y_pred)
    tp = K.sum(K.cast(y_true * y_pred, 'float'), axis=1)
    precision = tp / (K.sum(K.cast(y_pred, 'float'), axis=1) + K.epsilon())
    recall = tp / (K.sum(K.cast(y_true, 'float'), axis=1) + K.epsilon())
    f1_val = 2 * (precision * recall) / (precision + recall + K.epsilon())
    return K.mean(f1_val)

# 載入模型並指定 AttentionWithContext 和 f1 作為自定義物件
model = load_model("CPSC2018_10_fold_model_0", custom_objects={'AttentionWithContext': AttentionWithContext, 'f1': f1})

# # 顯示模型結構
# model.summary()

# 測試模型推論
test_input = np.random.rand(1, 72000, 12)  # 隨機生成一個測試輸入
test_output = model.predict(test_input)
print("測試輸出：", test_output)

# 檢查第一層卷積層的權重
conv_weights = model.get_layer("conv1d_31").get_weights()
print("第一層卷積層的權重形狀：", [w.shape for w in conv_weights])
