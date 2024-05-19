# %%
import matplotlib.pyplot as plt
import numpy as np
import cv2
import tensorflow as tf
import os

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from sklearn.metrics import confusion_matrix
import seaborn as sns
import datetime

import pathlib
from sklearn.model_selection import train_test_split

# %%
data_dir = 'bar_datasets'
data_dir = pathlib.Path(data_dir)
image_count = len(list(data_dir.glob('*/*.png')))
print(f'Image Count : {image_count}')

# %%
durian_heatmap_dict = {
    'ripe': list(data_dir.glob('ripe/*')),
    'unripe': list(data_dir.glob('unripe/*')),
}

durian_labels_dict = {
    'ripe': 0,
    'unripe': 1
}
# print(f'Some the files : {durian_heatmap_dict['ripe'][:5]}')

# %%
str(durian_heatmap_dict['ripe'][0])

# %%
X, y = [], []


output_folder = "resized_images"

for durian_ripeness, images in durian_heatmap_dict.items():
    for index, image in enumerate(images):
        img = cv2.imread(str(image))
        resized_img = cv2.resize(img, (200, 200))
        X.append(resized_img)
        y.append(durian_labels_dict[durian_ripeness])

        # Uncomment if you want to check all the resized images
        # output_name = f"{durian_ripeness}_{index}.jpg"
        # output_path = os.path.join(output_folder, output_name)
        # cv2.imwrite(output_path, resized_img)

# %%
X = np.array(X)
y = np.array(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7)
X_train = X_train / 255
X_test = X_test / 255
X_validation = X / 255

# %%
convDim = 3
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(8, (convDim, convDim),
                           activation='relu', input_shape=(200, 200, 3)),
    tf.keras.layers.MaxPool2D(2, 2),
    tf.keras.layers.Conv2D(16, (convDim, convDim),
                           activation='relu', input_shape=(200, 200, 3)),
    tf.keras.layers.MaxPool2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(8, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
model.compile(loss='binary_crossentropy',
              optimizer='adam', metrics=['accuracy'])
model.summary()

# %%
EPOCH = 25
model_fit = model.fit(X_train, y_train, epochs=EPOCH,
                      validation_data=(X_test, y_test))

# %%
# Plot training and validation history
plt.figure(figsize=(12, 6))

# Plot training & validation accuracy values
plt.subplot(1, 2, 1)
plt.plot(model_fit.history['accuracy'])
plt.plot(model_fit.history['val_accuracy'])  # Add validation accuracy
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['Train', 'Validation'], loc='upper left')

# Plot training & validation loss values
plt.subplot(1, 2, 2)
plt.plot(model_fit.history['loss'])
plt.plot(model_fit.history['val_loss'])  # Add validation loss
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['Train', 'Validation'], loc='upper left')

plt.tight_layout()

# save model to .keras file
now = datetime.datetime.now()
# Format the date-time string as 'MMDDYYYY-HHMM'
date_time_str = now.strftime("%d%m%Y-%H%M%S")
oke = date_time_str

# plt.savefig(f"{oke}.png")

plt.show()

# %%
predictions = model.predict(X_train)
predictions = np.round(predictions, 2)
predictions

# %%
test_loss, test_acc = model.evaluate(X_test,  y_test, verbose=2)
print(f'Model Accuracy: {test_acc}')

# %%
# model = tf.keras.models.load_model('saved_model/28042024-182139.keras')

# %%
# Generate predictions on the test set
y_pred = model.predict(X_test)
# Assuming it's a binary classification problem
y_pred_classes = (y_pred > 0.5).astype(int)

# Generate the confusion matrix
conf_mat = confusion_matrix(y_test, y_pred_classes)

# Plot the confusion matrix using seaborn
plt.figure(figsize=(8, 6))
sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues', xticklabels=[
            'Unripe', 'Ripe'], yticklabels=['Unripe', 'Ripe'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()
y_test

# %%
# save model to .keras file
now = datetime.datetime.now()
# Format the date-time string as 'MMDDYYYY-HHMM'
date_time_str = now.strftime("%d%m%Y-%H%M%S")
oke = date_time_str

# %%
model.save(f"{oke}.keras")
