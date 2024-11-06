import os
import argparse
import scipy.io as sio
from keras import initializers, regularizers, constraints, backend as K
import csv
import numpy as np
import tensorflow as tf
from tensorflow.python.client import device_lib
from keras.models import Sequential, load_model
from keras.layers import LSTM, GRU, TimeDistributed, Bidirectional, LeakyReLU
from keras.layers import Dense, Dropout, Activation, Flatten, Input, Reshape, CuDNNGRU
from keras.layers import Conv1D, MaxPool1D, GlobalAveragePooling1D, concatenate, AveragePooling1D
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping
from keras.models import Model
from keras import initializers, regularizers, constraints
from keras.layers import Layer
from keras.layers import BatchNormalization

## example:
# X: input data, whose shape is (72000,12)
# Y: output data, whose shape is  = (9,)
# Y = weighted_predict_for_one_sample_only(X)

def dot_product(x, kernel):
    """計算點積，支持不同的後端"""
    if K.backend() == 'tensorflow':
        return K.squeeze(K.dot(x, K.expand_dims(kernel)), axis=-1)
    else:
        return K.dot(x, kernel)
    
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
        assert len(input_shape) == 3
        self.W = self.add_weight((input_shape[-1], input_shape[-1],),
                                  initializer=self.init,
                                  name='{}_W'.format(self.name),
                                  regularizer=self.W_regularizer,
                                  constraint=self.W_constraint)
        if self.bias:
            self.b = self.add_weight((input_shape[-1],),
                                      initializer='zero',
                                      name='{}_b'.format(self.name),
                                      regularizer=self.b_regularizer,
                                      constraint=self.b_constraint) 
            self.u = self.add_weight((input_shape[-1],),
                                      initializer=self.init,
                                      name='{}_u'.format(self.name),
                                      regularizer=self.u_regularizer,
                                      constraint=self.u_constraint) 
        super(AttentionWithContext, self).build(input_shape)
 
    def compute_mask(self, input, input_mask=None):
        return None
 
    def call(self, x, mask=None):
        uit = dot_product(x, self.W) 
        if self.bias:
            uit += self.b 
        uit = K.tanh(uit)
        
        ait = dot_product(uit, self.u) 
        a = K.exp(ait)
        
        if mask is not None:
            a *= K.cast(mask, K.floatx())
            
        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx()) 
        a = K.expand_dims(a)
        
        weighted_input = x * a
        return K.sum(weighted_input, axis=1)
 
    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[-1]


# 配置 GPU 設置
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # 設定 GPU 記憶體使用量的自動增長
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("GPU 設置成功，記憶體自動增長啟用")
    except RuntimeError as e:
        print(e)
        
batch_size = 64
num_classes = 9
epochs = 1000000000000000000000000000000000

# 加載數據
magicVector = np.load('./magicVector_test_val_strategy.npy')
leadsLabel = np.asarray(['I','II','III','aVR','aVL','aVF','V1','V2','V3','V4','V5','V6'])

def build_model(input_shape=(72000, 12), num_classes=9):
    main_input = Input(shape=input_shape, dtype='float32', name='main_input')

    # 卷積層和 LeakyReLU 的結合
    x = main_input
    for _ in range(5):  # 五組卷積組合
        x = Conv1D(12, 3, padding='same')(x)
        x = LeakyReLU(alpha=0.3)(x)
        x = Conv1D(12, 3, padding='same')(x)
        x = LeakyReLU(alpha=0.3)(x)
        x = Conv1D(12, 24, strides=2, padding='same')(x)
        x = LeakyReLU(alpha=0.3)(x)
        x = Dropout(0.2)(x)

    # Bidirectional GRU 和 Attention with Context
    x = Bidirectional(GRU(12, return_sequences=True))(x)
    x = LeakyReLU(alpha=0.3)(x)
    x = Dropout(0.2)(x)

    x = AttentionWithContext()(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.3)(x)
    x = Dropout(0.2)(x)

    # 輸出層
    main_output = Dense(num_classes, activation='sigmoid')(x)

    # 建立模型
    model = Model(inputs=main_input, outputs=main_output)
    return model



models = {}  # 用字典來管理模型

# 初始化模型並載入權重
models = {}  # 用字典來管理模型

for fold in range(10):    
    for lead in range(13):
        model = build_model(input_shape=(72000, 12), num_classes=9)
        
        model_name = f'model_{lead}_{fold}'
        models[model_name] = model
        
        # 加載對應的權重
        weight_path = f'CPSC2018_10_fold_model_{fold}' if lead == 12 else f'CPSC2018_10_fold_model_{leadsLabel[lead]}_{fold}'
        if os.path.exists(weight_path):
            models[model_name].load_weights(weight_path)


def cpsc2018(record_base_path, models, magicVector):  
    def weighted_prediction_for_one_sample_only(target):
        fold_predict = np.zeros((10, 9))
        for fold in range(10):    
            lead_predict = np.zeros((13, 9))
            for lead in range(13):
                model_key = f'model_{lead}_{fold}'
                model = models[model_key]
                
                if lead == 12:
                    lead_predict[lead, :] = model.predict(target)[0, :].copy()
                else:
                    zeroIndices = np.asarray(list(set(range(12)) - {lead}))
                    target_temp = target.copy()
                    target_temp[0, :, zeroIndices] = 0
                    lead_predict[lead, :] = model.predict(target_temp)[0, :].copy()
            lead_predict = np.mean(lead_predict, axis=0)
            fold_predict[fold, :] = lead_predict.copy()
        y_pred = np.mean(fold_predict, axis=0)
        return y_pred * magicVector

    # 創建並寫入 CSV 文件
    with open('answers.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Recording', 'Result'])  # CSV 標題
        
        for mat_item in os.listdir(record_base_path):
            if mat_item.endswith('.mat') and not mat_item.startswith('._'):
                # 加載並處理數據
                record_path = os.path.join(record_base_path, mat_item)
                ecg = np.zeros((72000, 12), dtype=np.float32)
                ecg_data = sio.loadmat(record_path)['ECG'][0][0][2].T
                ecg[-ecg_data.shape[0]:, :] = ecg_data
                X = np.expand_dims(ecg, axis=0)
                
                # 預測和結果處理
                Y = weighted_prediction_for_one_sample_only(X)
                result = np.argmax(Y) + 1  # 預測類別 +1

                # 檢查預測範圍
                result = result if 1 <= result <= 9 else 1

                # 輸出答案
                record_name, _ = os.path.splitext(mat_item)
                answer = [record_name, result]
                print(answer)
                writer.writerow(answer)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-p',
                        '--recording_path',
                        help='Path where test record files are saved',
                        required=True)
    args = parser.parse_args()

    # 檢查指定的路徑是否存在
    if not os.path.isdir(args.recording_path):
        raise ValueError(f"Recording path '{args.recording_path}' does not exist.")

    # 呼叫 `cpsc2018` 函數並執行
    cpsc2018(record_base_path=args.recording_path, models=models, magicVector=magicVector)