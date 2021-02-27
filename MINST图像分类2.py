# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import  datasets
import os

# Helper libraries
import numpy as np


# 设置后台打印日志等级 避免后台打印一些无用的信息
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# 利用Tensorflow2中的接口加载mnist数据集
(train_images, train_labels), (test_images, test_labels) = datasets.fashion_mnist.load_data()

print(train_images.shape)
print(len(train_labels))
print(train_labels)
print(test_images.shape )
print(len(test_labels))

train_images=train_images/255.0
test_images=test_images/255.0

model=keras.Sequential([
    keras.layers.Flatten(input_shape=(28,28)),  #第一层Flatten将二维数组转化为一维数组平铺在一行
    keras.layers.Dense(128,activation='relu'),  #第二层Dense为全连接层128个神经元,激活函数为relu
    keras.layers.Dense(10,activation='softmax') #第三层Dense为全连接层10个神经元，激活函数为softmax
])


# optimizer-优化器算法，更新模型参数的算法
# loss损失函数
# metrics-指标，用来监视训练和测试步数
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


#训练模型
model.fit(train_images,train_labels,epochs=10)

predictions=model.predict(test_images)
print(predictions[0])
print(np.argmax(predictions[0]))
print(test_labels[0])


