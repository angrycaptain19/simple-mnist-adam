import tensorflow as tf
from tensorflow.keras import models, datasets, layers

(train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()

train_images, test_images = train_images/255.0, test_images/255.0
model = models.Sequential([
  layers.Flatten(input_shape=(28,28)),
  layers.Dense(128, activation='relu'),
  layers.Dropout(.3),
  layers.Dense(128, activation='relu'),
  layers.Dropout(.3),
  layers.Dense(10, activation='softmax')
  ])

model.summary()

model.compile(
  optimizer='Adam',
  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
  metrics=['accuracy']
  )

model.fit(
  train_images,
  train_labels,
  epochs=10,
  validation_split=.2
  )

test_loss, test_acc = model.evaluate(test_images, test_labels)
print('Test Accuracy: ', test_acc)
model.save('./models')
