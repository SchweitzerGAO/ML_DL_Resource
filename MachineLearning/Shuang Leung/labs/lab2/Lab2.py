import os
from keras.models import Sequential
from keras.layers import Input, Dense, BatchNormalization, Activation, Conv2D, Flatten, MaxPooling2D,Dropout
from keras.models import Model
from keras import optimizers
from keras.datasets import mnist,cifar10
from keras.utils.np_utils import to_categorical
from tensorflow.keras.utils import plot_model
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from keras.losses import SparseCategoricalCrossentropy
from keras.metrics import SparseCategoricalAccuracy

os.environ["CUDA_VISIBLE_DEVICES"] = ""
# 固定随机数种子，提高结果可重复性（在CPU上测试有效）
tf.random.set_seed(233)
np.random.seed(233)

'''第一步：选择模型'''
model = Sequential()  # 采用贯序模型

'''第二步：构建网络层'''
# 在此处构建你的网络
#####################################################################################
model.add(Input((32, 32, 3)))
# model.add(Flatten())
model.add(Conv2D(filters=7, kernel_size=3, strides=1, padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Conv2D(filters=7, kernel_size=3, strides=1, padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=2))
model.add(Dense(100))
model.add(Dropout(0.4))
model.add(Activation('relu'))
# model.add(Dense(500))
# model.add(Activation('relu'))
model.add(Flatten())
model.add(Dense(10))
model.add(Activation('softmax'))

#####################################################################################

'''第三步：网络优化/编译/模型输出'''
# 在此处调整优化器
# learning_rate：大于0的浮点数，学习率
# momentum：大于0的浮点数，动量参数
# decay：大于0的浮点数，每次更新后的学习率衰减值
# nesterov：布尔值，确定是否使用Nesterov动量
sgd = optimizers.SGD(learning_rate=0.01, decay=1e-6, momentum=0.9, nesterov=True)  # 优化函数，设定学习率（lr）等参数
adam = optimizers.Adam(learning_rate=0.001, decay=1e-6, epsilon=1e-7, amsgrad=True)
# 在此处调整损失函数，并编译网络
model.compile(loss='categorical_crossentropy', optimizer=sgd,
              metrics=['accuracy'])  # 使用交叉熵作为loss函数

# 在此处输出网络的架构。此处参数可以不用调整。
# model表示自定义的模型名 to_file表示存储的文件名 show_shapes是否显示形状  rankdir表示方向T(top)B(Bottow)


plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=False, rankdir='TB')

'''第四步：训练'''

# 数据集获取 mnist 数据集的介绍可以参考 https://blog.csdn.net/simple_the_best/article/details/75267863
(X_train, y_train), (X_test, y_test) = cifar10.load_data()  # 使用Keras自带的mnist工具读取数据（第一次运行需要联网）

# 数据处理与归一化
# 注意：X_train和X_test可以直接输入卷积层，但需要先Flatten才能输入全连接层
X_train = X_train.reshape((50000, 32, 32, 3)).astype('float') / 255
X_test = X_test.reshape((10000, 32, 32, 3)).astype('float') / 255

# 生成OneHot向量
Y_train = to_categorical(y_train)
Y_test = to_categorical(y_test)

batch_size = 256
epoch = 50
validation_split = 0.3
# 在此处调整训练细节 
'''
   .fit的一些参数
   batch_size：对总的样本数进行分组，每组包含的样本数量
   epochs ：训练次数
   shuffle：是否把数据随机打乱之后再进行训练
   validation_split：拿出百分之多少用来做交叉验证
   verbose：屏显模式 0：不输出  1：输出进度  2：输出每次的训练结果
'''
history = model.fit(X_train, Y_train, batch_size=batch_size, epochs=epoch,
                    shuffle=True, verbose=2, validation_split=validation_split)

'''第五步：输出与可视化'''
print("test set")
# 误差评价 ：按batch计算在batch用到的输入数据上模型的误差，并输出测试集准确率
scores = model.evaluate(X_test, Y_test, batch_size=256, verbose=1)
print("The test loss is %f" % scores[0])
print("The accuracy of the model is %f" % scores[1])
# 在此处实现你的可视化功能
#####################################################################################
train_loss = history.history['loss']
train_acc = history.history['accuracy']
val_loss = history.history['val_loss']
val_acc = history.history['val_accuracy']

# plot the learning curve
epoch_list = [i for i in range(1, epoch + 1)]

plt.figure()

plt.subplot(121)
plt.title('Loss diagram')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.plot(epoch_list, train_loss, color='r', label='train')
plt.plot(epoch_list, val_loss, color='b', label='val')
plt.legend()

plt.subplot(122)
plt.title('Acc')
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.plot(epoch_list, train_acc, color='r', label='train')
plt.plot(epoch_list, val_acc, color='b', label='val')
plt.legend()

plt.show()

#####################################################################################
