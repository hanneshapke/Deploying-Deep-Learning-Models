from grpc.beta import implementations
import tensorflow as tf
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2, get_model_metadata_pb2


def get_stub(host='127.0.0.1', port='9000'):
    channel = implementations.insecure_channel(host, int(port))
    stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)
    return stub


def get_model_prediction(model_input, stub, model_name='amazon_reviews', signature_name='serving_default'):
    """ no error handling at all, just poc"""
    request = predict_pb2.PredictRequest()
    request.model_spec.name = model_name
    request.model_spec.signature_name = signature_name
    request.inputs['input'].CopyFrom(tf.contrib.util.make_tensor_proto(model_input.reshape(1, 50, 45),
                                                                       verify_shape=True, shape=(1, 50, 45)))
    response = stub.Predict.future(request, 5.0)  # 5 seconds
    return response.result().outputs["rating_prob"].float_val


def get_model_version(model_name, stub):
    request = get_model_metadata_pb2.GetModelMetadataRequest()
    request.model_spec.name = 'amazon_reviews'
    request.metadata_field.append("signature_def")
    response = stub.GetModelMetadata(request, 10)
    # signature of loaded model is available here: response.metadata['signature_def']
    return response.model_spec.version.value
