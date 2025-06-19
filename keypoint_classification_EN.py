#!/usr/bin/env python
# coding: utf-8

# In[307]:


import csv

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

RANDOM_SEED = 42


# # Specify each path

# In[308]:


dataset = 'model/keypoint_classifier/keypoint.csv'
model_save_path = 'model/keypoint_classifier/keypoint_classifier.keras'
tflite_save_path = 'model/keypoint_classifier/keypoint_classifier.tflite'


# # Set number of classes

# In[309]:


NUM_CLASSES = 27


# # Dataset reading

# In[310]:


X_dataset = np.loadtxt(dataset, delimiter=',', dtype='float32', usecols=list(range(1, (21 * 2) + 1)))


# In[311]:


y_dataset = np.loadtxt(dataset, delimiter=',', dtype='int32', usecols=(0))


# In[312]:


X_train, X_test, y_train, y_test = train_test_split(X_dataset, y_dataset, train_size=0.70, random_state=RANDOM_SEED)


# In[313]:


def reshape_for_lstm(data):
    samples = data.shape[0]
    return data.reshape(samples, 21, 2)  # 21 keypoints as "time steps", 2 features (x,y) per step

X_train_reshaped = reshape_for_lstm(X_train)
X_test_reshaped = reshape_for_lstm(X_test)


# # Model building

# In[315]:


model = tf.keras.models.Sequential([
    tf.keras.layers.InputLayer(input_shape=(21, 2)),
    tf.keras.layers.LSTM(32),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(20, activation='relu'),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.Dense(10, activation='relu'),
    tf.keras.layers.Dense(NUM_CLASSES, activation='softmax')
])


# In[316]:


model.summary()  # tf.keras.utils.plot_model(model, show_shapes=True)


# In[317]:


cp_callback = tf.keras.callbacks.ModelCheckpoint(
    model_save_path, verbose=1, save_weights_only=False)
es_callback = tf.keras.callbacks.EarlyStopping(patience=20, verbose=1)


# In[318]:


model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)


# # Model training

# In[319]:


model.fit(
    X_train_reshaped,  # Use reshaped data
    y_train,
    epochs=1000,
    batch_size=128,
    validation_data=(X_test_reshaped, y_test),  # Use reshaped test data
    callbacks=[cp_callback, es_callback]
)


# In[320]:


val_loss, val_acc = model.evaluate(X_test_reshaped, y_test, batch_size=128)


# In[321]:


model = tf.keras.models.load_model(model_save_path)


# In[322]:


test_sample = X_test[0].reshape(1, 21, 2)
predict_result = model.predict(test_sample)
print(np.squeeze(predict_result))
print(np.argmax(np.squeeze(predict_result)))


# # Confusion matrix

# In[323]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix,classification_report


def print_confusion_matrix(y_true, y_pred, report=True):
    labels = sorted(list(set(y_true)))
    cmx_data = confusion_matrix(y_true, y_pred, labels=labels)

    df_cmx = pd.DataFrame(cmx_data, index=labels, columns=labels)

    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(df_cmx, annot=True, fmt='g' ,square=False)
    ax.set_ylim(len(set(y_true)), 0)
    plt.show()

    if report:
        print('Classification Report')
        print(classification_report(y_test, y_pred))

Y_pred = model.predict(X_test_reshaped)
y_pred = np.argmax(Y_pred, axis=1)

print_confusion_matrix(y_test, y_pred)


# # Convert to model for Tensorflow-Lite

# In[324]:


model.save(model_save_path, include_optimizer=False)


# In[325]:


converter = tf.lite.TFLiteConverter.from_keras_model(model)


converter.experimental_enable_resource_variables = True

converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS,
    tf.lite.OpsSet.SELECT_TF_OPS
]

converter._experimental_lower_tensor_list_ops = False

converter.optimizations = [tf.lite.Optimize.DEFAULT]

tflite_quantized_model = converter.convert()


with open(tflite_save_path, 'wb') as f:
    f.write(tflite_quantized_model)

print(f"TFLite model saved to: {tflite_save_path}")


# # Inference test

# In[326]:


interpreter = tf.lite.Interpreter(model_path=tflite_save_path)
interpreter.allocate_tensors()


# In[327]:


input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()


# In[328]:


interpreter.set_tensor(input_details[0]['index'], X_test_reshaped[0:1])


# In[329]:


interpreter.invoke()
tflite_results = interpreter.get_tensor(output_details[0]['index'])


# In[330]:


print(np.squeeze(tflite_results))
print(np.argmax(np.squeeze(tflite_results)))

