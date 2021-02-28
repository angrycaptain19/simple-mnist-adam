import tensorflow as tf
from tensorflow.keras import datasets, layers, models, preprocessing
import tensorflow_datasets as tfds

MAX_LEN = 200
NUM_WORDS = 10000
DIM_EMBEDDING = 256
EPOCHS = 20
BATCH_SIZE = 500

def load_data():
    (train_data, train_labels), (test_data, test_labels) = datasets.imdb.load_data(num_words=NUM_WORDS)
    train_data = preprocessing.sequence.pad_sequences(train_data, maxlen=MAX_LEN)
    test_data = preprocessing.sequence.pad_sequences(test_data, maxlen=MAX_LEN)

    return (train_data, train_labels), (test_data, test_labels)

def build_model():
    model = models.Sequential([
        layers.Embedding(NUM_WORDS, DIM_EMBEDDING, input_length=MAX_LEN), #create denser graph from sparse graph
        layers.Dropout(.3),
        layers.GlobalMaxPooling1D(), #voodoo
        layers.Dense(128, activation='relu'),
        layers.Dropout(.5),
        layers.Dense(1, activation='sigmoid')
        ]
    )

    return model

def main():
    (train_data, train_labels), (test_data, test_labels) = load_data()
    model = build_model()
    model.summary()
    model.compile(optimizer='Adam',
        loss=tf.keras.losses.binary_crossentropy,
        metrics=['accuracy']
        )
    model.fit(train_data, train_labels,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(test_data, test_labels)
        )
    score = model.evaluate(test_data, test_labels, 
        batch_size=BATCH_SIZE
        )
    print('\nTest score: ', score[0])
    print('\nTest accuracy: ', score[1])

if __name__ == "__main__":
    main()
