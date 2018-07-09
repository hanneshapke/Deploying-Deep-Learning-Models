import os
import numpy as np
from keras.models import model_from_json
from constants import CHARS, MAX_TOKENS


WORKING_DIR = os.getcwd()


def clean_data_encoded(sentence, max_tokens=MAX_TOKENS, sup_chars=CHARS):
    sentence = str(sentence).lower()
    x = np.zeros((max_tokens, ), dtype=np.float32)
    for i, c in enumerate(sentence):
        if i >= max_tokens:
            break
        try:
            x[i] = sup_chars.index(c)
        except ValueError:
            pass
    return x


def depreserve_model(path=WORKING_DIR):
    with open(os.path.join(path, 'preserved_model/model.json'), 'r') as json_file:
        model = model_from_json(json_file.read())

    model.load_weights(os.path.join(path, 'preserved_model/model.h5'))
    return model


if __name__ == '__main__':
    
    print("\nLoading the Keras model ...")
    model = depreserve_model()

    while True:
        print("\nEnter an Amazon review [:q for Quit]")
        sentence = input()
        if sentence == ':q':
            break
        model_input = clean_data_encoded(sentence)
        model_prediction = model.predict(model_input.reshape(1, 50))
        print("The model predicted ...")
        print(model_prediction)
