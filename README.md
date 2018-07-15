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

#### Export the model as protobuf

```python
import os
from keras import backend as K
import tensorflow as tf

tf.app.flags.DEFINE_integer('training_iteration', 1000, 'number of training iterations.')
tf.app.flags.DEFINE_integer('model_version', 1, 'version number of the model.')
tf.app.flags.DEFINE_string('work_dir', '/tmp', 'Working directory.')
FLAGS = tf.app.flags.FLAGS

export_path_base = '/tmp/amazon_reviews'
export_path = os.path.join(tf.compat.as_bytes(export_path_base), 
               tf.compat.as_bytes(str(FLAGS.model_version)))

builder = tf.saved_model.builder.SavedModelBuilder(export_path)

signature = tf.saved_model.signature_def_utils.predict_signature_def(
    inputs={'input': model.input}, outputs={'rating_prob': model.output})

builder.add_meta_graph_and_variables(
    sess=K.get_session(), tags=[tf.saved_model.tag_constants.SERVING],
    signature_def_map={
        tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: signature })
builder.save()

```

#### Set up the Tensorflow Serving

```terminal
$ git clone git@github.com:hanneshapke/Deploying_Deep_Learning_Models.git

$ docker build --pull -t $USER/tensorflow-serving-devel-cpu \
                      -f {path to repo}/\
                      Deploying_Deep_Learning_Models/\
                      examples/Dockerfile .

$ docker run -it -p 8500:8500 -p 8501:8501 
             -v {model_path}/exported_models/amazon_review/:/models 
             $USER/tensorflow-serving-devel-cpu:latest /bin/bash

$[docker bash] tensorflow_model_server --port=8500 
                                       --model_name={model_name}
                                       --model_base_path=/models/{model_name}

```

#### Setup a client (either gRPC or REST based)

```python
from grpc.beta import implementations
from tensorflow_serving.apis import prediction_service_pb2

def get_stub(host='127.0.0.1', port='8500'):
    channel = implementations.insecure_channel(host, int(port))
    stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)
    return stub

def get_model_prediction(model_input, stub, 
                         model_name='amazon_reviews', 
                         signature_name='serving_default'):

    request = predict_pb2.PredictRequest()
    request.model_spec.name = model_name
    request.model_spec.signature_name = signature_name
    
    request.inputs['input'].CopyFrom(
        tf.contrib.util.make_tensor_proto(
            model_input.reshape(1, 50), 
            verify_shape=True, shape=(1, 50)))
 
    response = stub.Predict.future(request, 5.0)  # wait max 5s
    return response.result().outputs["rating_prob"].float_val
```

```python
>>> sentence = "this product is really helpful"
>>> model_input = clean_data_encoded(sentence)

>>> get_model_prediction(model_input, stub)
[0.0250927172601223, 0.03738045319914818, 0.09454590082168579, 
0.33069494366645813, 0.5122858881950378]
```


## Deployment Options

### On Premise

* Kubeflow
* Tensorflow Serving
* MLflow
* Seldon.io

### Cloud base

* Google Cloud Platform
* Microsoft Azure ML
* Amazon SageMaker

More details [here](https://github.com/hanneshapke/Deploying_Deep_Learning_Models/blob/master/documentation/OSCON%20Tensorflow%20Day%20-%20Deploying%20Deep%20Learning%20Models.html)