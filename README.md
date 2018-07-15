# Deploying Deep Learning Models
Strategies to deploy deep learning models

## Sample Project

### Model Structure

Let's predict Amazon product ratings based on the comments with a small LSTM network.
```python
model_input = Input(shape=(MAX_TOKENS,))
x = Embedding(input_dim=len(CHARS), output_dim=10, input_length=MAX_TOKENS)(model_input)
x = LSTM(128)(text_input)
output = Dense(5, activation='softmax')(x)
model = Model(inputs=text_input, outputs=output)
optimizer = RMSprop(lr=0.01)
model.compile(loss='categorical_crossentropy', optimizer=optimizer)
```

### Testing our Model

#### Negative Review

```python
>> test_sentence = "horrible book, don't buy it"
>> test_vector = clean_data(test_sentence, max_tokens=MAX_TOKENS, sup_chars=CHARS)
>> model.predict(test_vector.reshape(1, MAX_TOKENS, len(CHARS)))
[[0.5927979  0.23748466 0.10798287 0.03301411 0.02872046]]
```

#### Positive Review

```python
>> test_sentence = "Awesome product."
>> test_vector = clean_data(test_sentence, max_tokens=MAX_TOKENS, sup_chars=CHARS)
>> model.predict(test_vector.reshape(1, MAX_TOKENS, len(CHARS)))
[[0.03493131 0.0394276  0.08326671 0.2957105  0.5466638 ]]
```

## Steps to Deploy the Sample Project

* Export the model as protobuf

* Set up the Tensorflow Serving

* Setup a client (either gRPC or REST based)

* Happy Deploying!

More details [here](https://github.com/hanneshapke/Deploying_Deep_Learning_Models/blob/master/documentation/OSCON%20Tensorflow%20Day%20-%20Deploying%20Deep%20Learning%20Models.html)