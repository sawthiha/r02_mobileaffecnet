# %%
import os
import cv2

import numpy as np
import tensorflow as tf
# import pandas as pd

from utils.proctor import *
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, accuracy_score

# %%

VALIDATION_PATH = 'val_set/'
VALIDATION_ANNO_PATH = os.path.join(VALIDATION_PATH, 'annotations')
VALIDATION_IMGS_PATH = os.path.join(VALIDATION_PATH, 'images')

expressions_list = []
truths = []
with MPProctor() as proctor:
  for img_path in os.listdir(VALIDATION_IMGS_PATH):
    id = img_path.split('.')[0]
    # val = float(np.load(os.path.join(VALIDATION_ANNO_PATH, f'{id}_val.npy')))
    # aro = float(np.load(os.path.join(VALIDATION_ANNO_PATH, f'{id}_aro.npy')))
    exp = int(np.load(os.path.join(VALIDATION_ANNO_PATH, f'{id}_exp.npy')))
    truths.append(exp)
    img_path = os.path.join(VALIDATION_IMGS_PATH, img_path)
    image = cv2.imread(img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = proctor.process(image)
    if results:
      expressions_list.append(results[0]['expressions'])
    else:
      # If MediaPipe failed to detect the face, just resize and predict anyway
      image = cv2.resize(image, (112, 112))
      aligned = fixed_image_standardize(image)
      inter, embeddings = proctor.face_reid.predict([aligned])
      expressions, = proctor.face_affect.predict([inter])
      expressions_list.append(expressions)

predicteds = np.array(expressions_list)
truths = np.array(truths)

# %%
cm = confusion_matrix(truths, np.argmax(predicteds, axis=1))
cm_percentage = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100


#%%
f1 = f1_score(truths, np.argmax(predicteds, axis=1), average=None)

# %%
precision = precision_score(truths, np.argmax(predicteds, axis=1), average=None)

# %%
recall = recall_score(truths, np.argmax(predicteds, axis=1), average=None)

# %%
accuracy = accuracy_score(truths, np.argmax(predicteds, axis=1))

# %%
m = tf.keras.metrics.SparseCategoricalAccuracy()
m.update_state(tf.expand_dims(truths, 1), predicteds)
tf_cat_accuracy = m.result().numpy()

# %%
print(f'Total Images: {predicteds.shape[0]}')

print(f'Accuracy: {accuracy}')
print(f'Categorical Accuracy (tensorflow): {tf_cat_accuracy}')
print(f'Recall: {recall}')
print(f'Precision: {precision}')
print(f'F1 score: {f1}')

print('\n************ Confusion matrix ************')
print(cm)
print('******************************************')
