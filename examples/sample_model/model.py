import os
import pandas as pd
import numpy as np

from keras.models import Model
from keras.layers import Dense, LSTM, Input, Embedding
from keras.optimizers import RMSprop
from keras.callbacks import ModelCheckpoint

from constants import BATCH_SIZE, EPOCHS, MAX_TOKENS, CHARS
from utils import clean_data_encoded, y_onehot

WORKING_DIR = os.getcwd()

TRAIN_FILE = os.path.join(WORKING_DIR, "examples/sample_model/data/train.csv")
VAL_FILE = os.path.join(WORKING_DIR, "examples/sample_model/data/test.csv")


def load_dataset(file_path, num_samples):
    df = pd.read_csv(file_path, usecols=[0, 1], nrows=num_samples)
    df.columns = ['rating', 'title']
    df['title_converted'] = df['title'].apply(lambda x: clean_data_encoded(x))
    df['rating_converted'] = df['rating'].apply(lambda y: y_onehot(y))
    return np.array(df.rating_converted.tolist()), np.array(df.title_converted.tolist())


def get_model(print_summary=False, lr=0.01):
    model_input = Input(shape=(MAX_TOKENS,))
    x = Embedding(input_dim=len(CHARS), output_dim=10, input_length=MAX_TOKENS)(model_input)
    x = LSTM(64)(x)
    output = Dense(5, activation='softmax')(x)

    model = Model(model_input, output)
    optimizer = RMSprop(lr=lr)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer)

    if print_summary:
        print(model.summary())

    return model


def preserve_model(model):
    model_json = model.to_json()
    with open(os.path.join(WORKING_DIR, 'examples/sample_model/preserved_model/model.json'), 'w') as json_file:
        json_file.write(model_json)
    model.save_weights(os.path.join(WORKING_DIR, 'examples/sample_model/preserved_model/model.h5'))


if __name__ == '__main__':
    print("Loading training/validation data ...")
    y_train, x_train = load_dataset(TRAIN_FILE, num_samples=100000)
    y_val, x_val = load_dataset(VAL_FILE, num_samples=10000)

    print("Training the model ...")
    model = get_model()
    model.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, verbose=1, validation_data=(x_val, y_val),
              callbacks=[ModelCheckpoint(os.path.join(WORKING_DIR, 'examples/sample_model/preserved_model/model_checkpoint'), 
              monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)])

    test_sentence = "horrible book, don't buy it"
    print("Testing the model with `{}` ...".format(test_sentence))
    test_vector = clean_data_encoded(test_sentence, max_tokens=MAX_TOKENS)
    model.predict(test_vector.reshape(1, MAX_TOKENS))

    print("Preserving the model ...")
    preserve_model(model)