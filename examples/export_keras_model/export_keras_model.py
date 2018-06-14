import os
import tensorflow as tf

from absl import flags as absl_flags
from keras import backend as K
from keras.models import model_from_json

WORKING_DIR = os.getcwd()

try:
    tf.app.flags.DEFINE_integer('training_iteration', 1000, 'number of training iterations.')
    tf.app.flags.DEFINE_integer('model_version', 1, 'version number of the model.')
    tf.app.flags.DEFINE_string('work_dir', os.path.join(WORKING_DIR, 'examples/export_keras_model/exported_models/'), 'Working directory.')
except absl_flags._exceptions.DuplicateFlagError:
    pass

FLAGS = tf.app.flags.FLAGS


def depreserve_model(path=WORKING_DIR):
    with open(os.path.join(path, 'examples/sample model/preserved model/model.json'), 'r') as json_file:
        model = model_from_json(json_file.read())

    model.load_weights(os.path.join(path, 'examples/sample model/preserved model/model.h5'))
    return model

def export_keras_model_to_protobuf(model, model_name):
    export_path = os.path.join(
          tf.compat.as_bytes(FLAGS.work_dir + model_name),
          tf.compat.as_bytes(str(FLAGS.model_version)))

    print('Exporting trained model to', export_path)
    builder = tf.saved_model.builder.SavedModelBuilder(export_path)

    signature = tf.saved_model.signature_def_utils.predict_signature_def(
        inputs={model_name + '_input': model.input}, outputs={model_name + '_output': model.output})

    builder.add_meta_graph_and_variables(
          sess=K.get_session(), tags=[tf.saved_model.tag_constants.SERVING],
          signature_def_map={
              tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
                  signature
          })
    builder.save()


if __name__ == '__main__':
    model = depreserve_model()
    export_keras_model_to_protobuf(model, "amazon_review")