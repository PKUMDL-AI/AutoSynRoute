# coding=utf-8
# Copyright 2018 The Tensor2Tensor Authors.
"""Query an exported model. Py2 only. Install tensorflow-serving-api."""
# from __future__ import absolute_import
# from __future__ import division
# from __future__ import print_function

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"]="3"

# import warnings
# warnings.resetwarnings()
# warnings.simplefilter(action='ignore', category=FutureWarning)
# warnings.simplefilter(action='ignore', category=ImportWarning)
# warnings.simplefilter(action='ignore', category=RuntimeWarning)
# warnings.simplefilter(action='ignore', category=DeprecationWarning)
# warnings.simplefilter(action='ignore', category=ResourceWarning)

from oauth2client.client import GoogleCredentials
from six.moves import input  # pylint: disable=redefined-builtin

import serving_utils
from tensor2tensor.utils import registry
from tensor2tensor.utils import usr_dir
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.ERROR)
import numpy as np

flags = tf.flags
FLAGS = flags.FLAGS

flags.DEFINE_string("server", "0.0.0.0:9000", "Address to Tensorflow Serving server.")
flags.DEFINE_string("servable_name", "retrosyn_model", "Name of served model.")
flags.DEFINE_string("problem", "my_reaction_token", "Problem name.")
flags.DEFINE_string("data_dir", "../model/model_USPTO_50K/t2t_data_class_char", "Data directory, for vocab files.")
flags.DEFINE_string("t2t_usr_dir", "../model/model_USPTO_50K/my_problem", "Usr dir for registrations.")
flags.DEFINE_string("inputs_once", None, "Query once with this input.")
flags.DEFINE_integer("timeout_secs", 50000 , "Timeout for query.")

flags.DEFINE_string("work_path", None, "the workdir")

# For Cloud ML Engine predictions.
flags.DEFINE_string("cloud_mlengine_model_name", None,
                    "Name of model deployed on Cloud ML Engine.")
flags.DEFINE_string(
    "cloud_mlengine_model_version", None,
    "Version of the model to use. If None, requests will be "
    "sent to the default version.")




def validate_flags():
  """Validates flags are set to acceptable values."""
  if FLAGS.cloud_mlengine_model_name:
    assert not FLAGS.server
    assert not FLAGS.servable_name
  else:
    assert FLAGS.server
    assert FLAGS.servable_name


def make_request_fn():
  """Returns a request function."""
  if FLAGS.cloud_mlengine_model_name:
    request_fn = serving_utils.make_cloud_mlengine_request_fn(
        credentials=GoogleCredentials.get_application_default(),
        model_name=FLAGS.cloud_mlengine_model_name,
        version=FLAGS.cloud_mlengine_model_version)
  else:

    request_fn = serving_utils.make_grpc_request_fn(
        servable_name=FLAGS.servable_name,
        server=FLAGS.server,
        timeout_secs=FLAGS.timeout_secs)
  return request_fn


def main(_):
  # tf.logging.set_verbosity(tf.logging.INFO)
  validate_flags()
  usr_dir.import_usr_dir(FLAGS.t2t_usr_dir)
  problem = registry.problem(FLAGS.problem)
  hparams = tf.contrib.training.HParams(
      data_dir=os.path.expanduser(FLAGS.data_dir))
  problem.get_hparams(hparams)
  request_fn = make_request_fn()


  tmp_dir="/".join(FLAGS.work_path.split("/")[:-1])
  if not os.path.exists(tmp_dir):
      os.makedirs(tmp_dir)



  inputs = FLAGS.inputs_once if FLAGS.inputs_once else input(">> ")
  rx_num=10
  smilist_class = []
  for j in range(1, rx_num + 1):
    smilist_class.append("<RX_%d> %s" % (j, " ".join(list(inputs))))
  # input_list = inputs.split("\n")
  outputs = serving_utils.predict(smilist_class, problem, request_fn)
  np.save(FLAGS.work_path, np.array(outputs,dtype=object))
  # print(outputs)
  print("Saved!")




if __name__ == "__main__":
#   # flags.mark_flags_as_required(["problem", "data_dir"])
    tf.app.run()
    # main(_)
