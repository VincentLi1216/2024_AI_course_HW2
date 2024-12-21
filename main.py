import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

# 加載數據集
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# 調整數據形狀和範圍
train_images = train_images.reshape((60000, 28, 28, 1)).astype('float32') / 255
test_images = test_images.reshape((10000, 28, 28, 1)).astype('float32') / 255

# 獲取One-hot編碼標籤
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

# 自定義交叉熵損失函數
def custom_cross_entropy(y_true, y_pred):
    # 使用小的常數避免計算 log(0) 導致的數值穩定性問題
    epsilon = 1e-7
    y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
    # 計算交叉熵
    return -tf.reduce_sum(y_true * tf.math.log(y_pred))

# 建立模型
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))  # 第一層卷積層
model.add(layers.MaxPooling2D((2, 2)))  # 第一層池化層
model.add(layers.Conv2D(64, (3, 3), activation='relu'))  # 第二層卷積層
model.add(layers.MaxPooling2D((2, 2)))  # 第二層池化層
model.add(layers.Conv2D(64, (3, 3), activation='relu'))  # 第三層卷積層
model.add(layers.Flatten())  # Flatten layer
model.add(layers.Dense(64, activation='relu'))  # Dense layer
model.add(layers.Dense(10, activation='softmax'))  # Output layer

# 編譯模型，使用自定義的交叉熵損失函數
model.compile(optimizer='adam',
              loss=custom_cross_entropy,
              metrics=['accuracy'])

# 訓練模型
model.fit(train_images, train_labels, epochs=5, batch_size=64)

# 評估模型
test_loss, test_acc = model.evaluate(test_images, test_labels)
print(f'Test accuracy: {test_acc:.4f}')
