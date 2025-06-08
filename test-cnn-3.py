import numpy as np
import matplotlib.pyplot as plt

# 設置matplotlib中文字體
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

import os

# 資料夾路徑
dataset_path = "C:/Users/nancy/Documents/3B361077/flower_photos/flower_photos"  

# 儲存結果
class_counts = {}
total = 0

# 走訪每個子資料夾（每個分類）
for class_name in os.listdir(dataset_path):
    class_folder = os.path.join(dataset_path, class_name)
    if os.path.isdir(class_folder):
        image_files = [f for f in os.listdir(class_folder) if f.lower().endswith('.jpg')]
        count = len(image_files)
        class_counts[class_name] = count
        total += count

# 輸出統計表
print("flower_photos 資料集分類統計：\n")
print(f"{'類別':<15}{'數量'}")
print("-" * 25)
for class_name, count in class_counts.items():
    print(f"{class_name:<15}{count}")
print("-" * 25)
print(f"{'總數':<15}{total}")

import os
import shutil
import random

# 設定路徑
source_dir = "C:/Users/nancy/Documents/3B361077/flower_photos/flower_photos"
output_dir = "C:/Users/nancy/Documents/3B361077/flower_split"

# 設定切分比例
train_ratio = 0.7
val_ratio = 0.2
test_ratio = 0.1

# 確保資料夾自動建立
splits = ['train', 'val', 'test']
for split in splits:
    os.makedirs(os.path.join(output_dir, split), exist_ok=True)

# 遍歷每個類別資料夾
for class_name in os.listdir(source_dir):
    class_path = os.path.join(source_dir, class_name)
    if not os.path.isdir(class_path):
        continue  # 如果不是資料夾（可能是README），就跳過

    # 讀取所有圖片檔案
    image_files = [f for f in os.listdir(class_path) if f.lower().endswith('.jpg')]
    random.shuffle(image_files)  # 打亂順序

    # 計算數量
    total = len(image_files)
    train_count = int(total * train_ratio)
    val_count = int(total * val_ratio)

    # 切割
    train_files = image_files[:train_count]
    val_files = image_files[train_count:train_count + val_count]
    test_files = image_files[train_count + val_count:]

    # 搬移檔案
    for split_name, file_list in zip(['train', 'val', 'test'], [train_files, val_files, test_files]):
        split_class_dir = os.path.join(output_dir, split_name, class_name)
        os.makedirs(split_class_dir, exist_ok=True)

        for file in file_list:
            src = os.path.join(class_path, file)
            dst = os.path.join(split_class_dir, file)
            shutil.copy2(src, dst)

print("自動分割完成，接著使用 ImageDataGenerator 載入圖片。")

from tensorflow.keras.preprocessing.image import ImageDataGenerator

# 圖片尺寸與批次大小
img_size = (128, 128)
batch_size = 32

# 訓練資料使用資料增強
train_datagen = ImageDataGenerator(
    rescale=1.0/255,           # 標準化：像素值縮放到[0,1]
    rotation_range=20,         # 隨機旋轉
    width_shift_range=0.2,     # 水平平移
    height_shift_range=0.2,    # 垂直平移
    horizontal_flip=True       # 水平翻轉
)

# 驗證與測試資料不需資料增強，只需 rescale
test_val_datagen = ImageDataGenerator(rescale=1.0/255)

# 建立資料產生器
train_generator = train_datagen.flow_from_directory(
    'flower_split/train',
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical'
)

val_generator = test_val_datagen.flow_from_directory(
    'flower_split/val',
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical'
)

test_generator = test_val_datagen.flow_from_directory(
    'flower_split/test',
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False   # 測試集不打亂，方便報告分析
)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input

# 模型架構
model = Sequential()
model.add(Input(shape=(128, 128, 3)))  # 影像大小 + RGB 三通道
model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(5, activation='softmax'))

# 編譯模型
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 模型摘要
model.summary()

# 訓練模型（使用資料生成器）
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=20
)

# 顯示訓練最後一個 epoch 的準確度與損失函數
train_loss = history.history["loss"][-1]
train_acc = history.history["accuracy"][-1]
val_loss = history.history["val_loss"][-1]
val_acc = history.history["val_accuracy"][-1]

print("\n=== 訓練與驗證集最終結果 ===")
print(f"訓練集準確度：{train_acc:.3f}")
print(f"訓練集損失值：{train_loss:.3f}")
print(f"驗證集準確度：{val_acc:.3f}")
print(f"驗證集損失值：{val_loss:.3f}")
# 儲存模型
model.save("flower_model.keras")
print("模型已儲存為 flower_model.keras")

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

# 取得預測結果
Y_pred = model.predict(test_generator)
y_pred = np.argmax(Y_pred, axis=1)
y_true = test_generator.classes
class_labels = list(test_generator.class_indices.keys())

# 1. 測試集準確率
loss, acc = model.evaluate(test_generator, verbose=0)
print("測試集準確率：{:.2f}%".format(acc * 100))

# 2. 混淆矩陣
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_labels, yticklabels=class_labels)
plt.title("混淆矩陣")
plt.xlabel("預測類別")
plt.ylabel("實際類別")
plt.show()

# 3. 類別別精確率表格
report = classification_report(y_true, y_pred, target_names=class_labels, output_dict=True)
import pandas as pd
report_df = pd.DataFrame(report).transpose()
print("分類報告：")
print(report_df[["precision", "recall", "f1-score"]].round(2))

# 正確與錯誤預測的索引
correct = np.where(y_pred == y_true)[0]
incorrect = np.where(y_pred != y_true)[0]

# 顯示正確案例（5張）
plt.figure(figsize=(12, 6))
for i, idx in enumerate(correct[:5]):
    img, _ = test_generator[idx]
    plt.subplot(2, 5, i + 1)
    plt.imshow(img[0])
    plt.title(f" {class_labels[y_pred[idx]]}")
    plt.axis('off')

# 顯示錯誤案例（5張）
for i, idx in enumerate(incorrect[:5]):
    img, _ = test_generator[idx]
    plt.subplot(2, 5, i + 6)
    plt.imshow(img[0])
    plt.title(f" {class_labels[y_pred[idx]]}\n→ {class_labels[y_true[idx]]}")
    plt.axis('off')

plt.suptitle("正確與錯誤預測案例（各5張）")
plt.tight_layout()
plt.show()

from tensorflow.keras.models import Model

# 抽取一張測試圖片
test_img, _ = test_generator[0]
img = test_img[0]  # 第一張圖
input_img = np.expand_dims(img, axis=0)  # 扩維度

# 找出模型中所有 Conv2D 層的輸出
layer_outputs = [layer.output for layer in model.layers if isinstance(layer, Conv2D)]
activation_model = Model(inputs=model.input, outputs=layer_outputs)

# 執行預測，取得特徵圖
feature_maps = activation_model.predict(input_img)

# 顯示每層前 6 張特徵圖
for layer_name, feature_map in zip([layer.name for layer in model.layers if isinstance(layer, Conv2D)], feature_maps):
    n_features = feature_map.shape[-1]
    size = feature_map.shape[1]

    plt.figure(figsize=(12, 4))
    plt.suptitle(f"特徵圖：{layer_name}")
    for i in range(min(6, n_features)):
        plt.subplot(1, 6, i + 1)
        plt.imshow(feature_map[0, :, :, i], cmap='viridis')
        plt.axis('off')
    plt.show()
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input

# 以一張圖片大小建立新的輸入
new_input = Input(shape=(128, 128, 3))  
new_outputs = model(new_input)
activation_model = Model(inputs=new_input, outputs=[layer.output for layer in model.layers if 'conv' in layer.name])

import matplotlib.pyplot as plt
# 顯示訓練和驗證損失
loss = history.history["loss"]
epochs = range(1, len(loss)+1)
val_loss = history.history["val_loss"]
plt.plot(epochs, loss, "bo-", label="Training Loss")
plt.plot(epochs, val_loss, "ro--", label="Validation Loss")
plt.title("Training and Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()
# 顯示訓練和驗證準確度
acc = history.history["accuracy"]
epochs = range(1, len(acc)+1)
val_acc = history.history["val_accuracy"]
plt.plot(epochs, acc, "bo-", label="Training Acc")
plt.plot(epochs, val_acc, "ro--", label="Validation Acc")
plt.title("Training and Validation Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.show()