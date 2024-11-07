import tensorflow as tf
print("TensorFlow CUDA version:", tf.sysconfig.get_build_info()["cuda_version"])
print("TensorFlow cuDNN version:", tf.sysconfig.get_build_info()["cudnn_version"])