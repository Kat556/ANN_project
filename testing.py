import tensorflow_datasets as tfds
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import re
import numpy as np
from tensorflow.data import Dataset

#load data & label file
imdb_reviews = pd.read_csv('imdb_labelled.txt', sep='\t', header=None)
imdb_reviews.columns = ['comments', 'ranking']

features = imdb_reviews.comments.values
labels = imdb_reviews.ranking.values


#tokenize the features
tempTokenizer = tf.keras.preprocessing.text.Tokenizer(num_words = 10000, filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n', lower = True, split=' ', char_level=False, oov_token=1)
tempTokenizer.fit_on_texts(features)

print("tokenizer num_words: ", tempTokenizer.get_config()["num_words"])
a = tempTokenizer.texts_to_sequences(features)
print("len before tokenized: ", len(features))
print("len after tokenized: ", len(a))
print("sample before tokenized: ", features[0])
print("sample after tokenized: ", a[0])
print("detokenized: ", tempTokenizer.sequences_to_texts(a)[0])

data_tensor = tf.ragged.constant(a, dtype=tf.int64)
#create dataset
dataset = tf.data.Dataset.from_tensor_slices((data_tensor,labels))
def func(x, y):
    return tf.convert_to_tensor(x, dtype=tf.int64), y
dataset = dataset.map(func)
print("complete-dataset loaded")
print("sample: ", list(dataset.as_numpy_iterator())[0][0])


#split dataset into training set and test 
train_dataset = dataset.take(500)
test_dataset = dataset.skip(500)
val_dataset = dataset.skip(500)
test_dataset = dataset.take(500)
BATCH_SIZE = 8
print("complete-split")

#create batch
train_dataset = train_dataset.padded_batch(BATCH_SIZE)
print("number of iterations will be: ", len(list(train_dataset.as_numpy_iterator())))
test_dataset = test_dataset.padded_batch(BATCH_SIZE)
print("complete-batching")



#graph function
def plot_graphs(history, metric):
  plt.plot(history.history[metric])
  plt.plot(history.history['val_'+metric], '')
  plt.xlabel("Epochs")
  plt.ylabel(metric)
  plt.legend([metric, 'val_'+metric])
  plt.show()
"""

#load dataset
print("start: loading dataset")
dataset, info = tfds.load('imdb_reviews/subwords8k', with_info=True,
                          as_supervised=True)
print("finish: dataset")
encoder = info.features['text'].encoder
print('Vocabulary size: {}'.format(encoder.vocab_size))

train_dataset, test_dataset = dataset['train'], dataset['test']
print("dataset len before split: ", len(list(train_dataset.as_numpy_iterator())))
print("dataset before split: ", list(train_dataset.as_numpy_iterator())[0])
print("len: ", len(list(train_dataset.as_numpy_iterator())[0][0]))
print("decoded: ", encoder.decode(list(train_dataset.as_numpy_iterator())[0][0]))
print("dataset before split: ", list(train_dataset.as_numpy_iterator())[1])
print("len: ", len(list(train_dataset.as_numpy_iterator())[1][0]))
print("dataset before split: ", list(train_dataset.as_numpy_iterator())[2])
print("len: ", len(list(train_dataset.as_numpy_iterator())[2][0]))


sample_string = 'Hello TensorFlow.'

encoded_string = encoder.encode(sample_string)
print('Encoded string is {}'.format(encoded_string))

original_string = encoder.decode(encoded_string)
print('The original string: "{}"'.format(original_string))

assert original_string == sample_string

for index in encoded_string:
  print('{} ----> {}'.format(index, encoder.decode([index])))


#manipulate dataset
train_dataset, test_dataset = dataset['train'], dataset['test']

BUFFER_SIZE = 10000
BATCH_SIZE = 64

train_dataset = train_dataset.padded_batch(BATCH_SIZE)
"""

#define model
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(tempTokenizer.get_config()["num_words"], 64),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64,  return_sequences=True)),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
    tf.keras.layers.Dense(64, activation='relu'),
    #tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(1)
])


model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              optimizer=tf.keras.optimizers.Adam(1e-4),
              metrics=['accuracy'])


history = model.fit(train_dataset, epochs=10, validation_data=test_dataset)#, validation_steps=30)

test_loss, test_acc = model.evaluate(test_dataset)

print('Test Loss: {}'.format(test_loss))
print('Test Accuracy: {}'.format(test_acc))


def pad_to_size(vec, size):
  zeros = [0] * (size - len(vec))
  vec.extend(zeros)
  return vec

def sample_predict(sample_pred_text, pad):
  encoded_sample_pred_text = tempTokenizer.texts_to_sequences([sample_pred_text])

  if pad:
    encoded_sample_pred_text[0] = pad_to_size(encoded_sample_pred_text[0], 64)
  encoded_sample_pred_text[0] = tf.cast(encoded_sample_pred_text[0], tf.float32)
  predictions = model.predict(tf.expand_dims(encoded_sample_pred_text[0], 0))

  return (predictions)


sample_pred_text = ('The movie was cool. The animation and the graphics '
                    'were out of this world. I would recommend this movie.')
predictions = sample_predict(sample_pred_text, pad=False)
print(predictions)


sample_pred_text = ('The movie was cool. The animation and the graphics '
                    'were out of this world. I would recommend this movie.')
predictions = sample_predict(sample_pred_text, pad=True)
print(predictions)

plot_graphs(history, 'accuracy')
plot_graphs(history, 'loss')
