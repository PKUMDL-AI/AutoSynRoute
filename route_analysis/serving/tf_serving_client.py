# coding=utf-8
# Copyright 2018 The Tensor2Tensor Authors.
"""Query an exported model. Py2 only. Install tensorflow-serving-api."""
# from __future__ import absolute_import
# from __future__ import division
# from __future__ import print_function

import os
import functools
from tensor2tensor.utils import registry
from tensor2tensor.utils import usr_dir
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)
import numpy as np
import grpc
from tensor2tensor.data_generators import text_encoder
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc
from utils import *

flags = tf.flags
FLAGS = flags.FLAGS

flags.DEFINE_string("server", "0.0.0.0:9000", "Address to Tensorflow Serving server.")
flags.DEFINE_string("servable_name", "retrosyn_model", "Name of served model.")
flags.DEFINE_string("problem", "my_reaction_token", "Problem name.")
flags.DEFINE_string("data_dir", "../t2t_data_class_char", "Data directory, for vocab files.")
flags.DEFINE_string("t2t_usr_dir", "../my_problem", "Usr dir for registrations.")
flags.DEFINE_string("inputs_once",
                    "Nc1cccc2c1CN(C1CCC(=O)NC1=O)C2=O\nO=C1CCC(N2Cc3c(cccc3[N+](=O)[O-])C2=O)C(=O)N1\nO=C1CCC(NC(=O)c2cccc([N+](=O)[O-])c2CBr)C(=O)N1\nO=C(Cl)c1cccc([N+](=O)[O-])c1CBr",
                    "Query once with this input.")
# flags.DEFINE_string("inputs_once", "O=C1CCC(N2Cc3c(cccc3[N+](=O)[O-])C2=O)C(=O)N1", "Query once with this input.")

flags.DEFINE_integer("timeout_secs", 1000, "Timeout for query.")

flags.DEFINE_string("work_path", None, "the workdir")

# For Cloud ML Engine predictions.
flags.DEFINE_string("cloud_mlengine_model_name", None,
                    "Name of model deployed on Cloud ML Engine.")
flags.DEFINE_string(
    "cloud_mlengine_model_version", None,
    "Version of the model to use. If None, requests will be "
    "sent to the default version.")



def _encode(inputs, encoder, add_eos=True):
  input_ids = encoder.encode(inputs)
  if add_eos:
    input_ids.append(text_encoder.EOS_ID)
  return input_ids


def _decode(output_ids, output_decoder):
  return output_decoder.decode(output_ids, strip_extraneous=True)

def _make_example(input_ids, problem, input_feature_name="inputs"):
  """Make a tf.train.Example for the problem.

  features[input_feature_name] = input_ids

  Also fills in any other required features with dummy values.

  Args:
    input_ids: list<int>.
    problem: Problem.
    input_feature_name: name of feature for input_ids.

  Returns:
    tf.train.Example
  """
  features = {
      input_feature_name:
          tf.train.Feature(int64_list=tf.train.Int64List(value=input_ids))
  }

  # Fill in dummy values for any other required features that presumably
  # will not actually be used for prediction.
  data_fields, _ = problem.example_reading_spec()
  for fname, ftype in data_fields.items():
    if fname == input_feature_name:
      continue
    if not isinstance(ftype, tf.FixedLenFeature):
      # Only FixedLenFeatures are required
      continue
    if ftype.default_value is not None:
      # If there's a default value, no need to fill it in
      continue
    num_elements = functools.reduce(lambda acc, el: acc * el, ftype.shape, 1)
    if ftype.dtype in [tf.int32, tf.int64]:
      value = tf.train.Feature(
          int64_list=tf.train.Int64List(value=[0] * num_elements))
    if ftype.dtype in [tf.float32, tf.float64]:
      value = tf.train.Feature(
          float_list=tf.train.FloatList(value=[0.] * num_elements))
    if ftype.dtype == tf.bytes:
      value = tf.train.Feature(
          bytes_list=tf.train.BytesList(value=[""] * num_elements))
    tf.logging.info("Adding dummy value for feature %s as it is required by "
                    "the Problem.", fname)
    features[fname] = value
  return tf.train.Example(features=tf.train.Features(feature=features))


def parse_outlist_to_rxsmi(output_list, target_smi):
    pathsmi = []
    import collections
    seq_score_dict = collections.OrderedDict()
    seq_count_dict = collections.OrderedDict()
    seq_label_dict = collections.OrderedDict()
    # seq_totalscore_dict = collections.OrderedDict()


    for rxid, line in enumerate(output_list):
        for topid, smi_score in enumerate(line):
            canoseq = cano(smi_score[0])
            if canoseq != "" and (get_main_part_from_smistr(canoseq) != target_smi):  # None, or not equal to  the own
                rc_value = cal_changed_ring(target_smi, canoseq)
                lc_value = cal_changed_smilen(target_smi, canoseq)
                tmp_score=100*np.exp(smi_score[1])- (6 * rc_value + lc_value)
                if canoseq not in seq_score_dict.keys():
                    seq_score_dict[canoseq] = ["RX%d_TOP%d" % (rxid + 1, topid + 1), tmp_score]
                    # seq_score_dict[canoseq] = tmp_score
                    # seq_totalscore_dict[canoseq] = \
                    #     ["RX%d_TOP%d" % (rxid + 1, topid + 1), smi_score[1],cal_changed_ring(target_smi, canoseq), cal_changed_smilen(target_smi, canoseq)]
                    seq_count_dict[canoseq] = 1
                    seq_label_dict[canoseq] = "RX%d_TOP%d" % (rxid + 1, topid + 1)
                else:
                    if tmp_score > seq_score_dict[canoseq][1]:
                        seq_label_dict[canoseq] = "RX%d_TOP%d" % (rxid + 1, topid + 1)
                        seq_score_dict[canoseq] = ["RX%d_TOP%d" % (rxid + 1, topid + 1), tmp_score]
                        # seq_score_dict[canoseq] = tmp_score
                        # seq_totalscore_dict[canoseq] = \
                        #     ["RX%d_TOP%d" % (rxid + 1, topid + 1), smi_score[1], cal_changed_ring(target_smi, canoseq), cal_changed_smilen(target_smi, canoseq)]
                    seq_count_dict[canoseq] += 1
    [pathsmi.append("%s,%s" % (seq_label_dict[key], key)) for key in seq_label_dict.keys()]

    # count_arr = np.array(list(seq_count_dict.values()))
    list1=[value for value in seq_score_dict.values()]
    mylist1= sorted(list1, key=lambda x: -x[1])
    # the rate of reactant occurence as probability
    # score_arr = np.array(list(seq_score_dict.values()))
    # score_arr_norm = score_arr +( 0- np.min(score_arr))
    # preds = score_arr_norm/ np.sum(score_arr_norm)

    # the bleu score of reactant occurence as probability
    # score_arr = np.array(list(seq_score_dict.values()))
    # preds = score_arr / np.sum(score_arr)
    return pathsmi, mylist1


def load_config_request():
    channel = grpc.insecure_channel(FLAGS.server)
    stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)

    usr_dir.import_usr_dir(FLAGS.t2t_usr_dir)
    problem = registry.problem(FLAGS.problem)
    hparams = tf.contrib.training.HParams(
        data_dir=os.path.expanduser(FLAGS.data_dir))
    problem.get_hparams(hparams)

    request = predict_pb2.PredictRequest()
    request.model_spec.name = FLAGS.servable_name
    request.model_spec.signature_name = 'serving_default'

    return stub, problem,request

# stub, problem,request=load_config_request()

def predict(inputs, output_path):
    # create prediction service client stub
    print(inputs)
    # data encoder
    rx_num = 10
    smilist_class = []
    for j in range(1, rx_num + 1):
        smilist_class.append("<RX_%d> %s" % (j, " ".join(list(inputs))))

    fname = "inputs" if problem.has_inputs else "targets"
    input_encoder = problem.feature_info[fname].encoder
    input_ids_list = [
        _encode(inputs, input_encoder, add_eos=problem.has_inputs)
        for inputs in smilist_class
    ]
    examples = [_make_example(input_ids, problem, fname)
                for input_ids in input_ids_list]

    # create request

    request.inputs["input"].CopyFrom(
        tf.make_tensor_proto(
            [ex.SerializeToString() for ex in examples], shape=[len(examples)]))

    response = stub.Predict(request, FLAGS.timeout_secs)

    outputs = tf.make_ndarray(response.outputs["outputs"])
    scores = tf.make_ndarray(response.outputs["scores"])

    assert len(outputs) == len(scores)
    predictions= [{
        "outputs": outputs[i],
        "scores": scores[i]
    } for i in range(len(outputs))]

    output_decoder = problem.feature_info["targets"].encoder
    outputs = []
    for prediction in predictions:
        one_outputs = []
        for seq, score in zip(prediction["outputs"], prediction["scores"]):
            one_outputs.append([_decode(seq, output_decoder).split(" <EOS>")[0].replace(" ", ""), score])
        outputs.append(one_outputs)


    # rx_seq, sorted_rx=parse_outlist_to_rxsmi(outputs, cano(inputs))
    # print(rx_seq,sorted_rx)

    # return
    np.save(output_path, np.array(outputs, dtype=object))
    # return rx_seq, sorted_rx




if __name__ == "__main__":
    import time
#   # flags.mark_flags_as_required(["problem", "data_dir"])
#     inputs = FLAGS.inputs_once.split("\n")
#     stub, problem, request = load_config_request()
#
#     import time
#     st_time = time.time()
#     # return_list=list()
#
#     from multiprocessing import Pool, Process, Queue
#     # pool=Pool(4)
#     # results=[]
#     # # for input in inputs:
#     # #     res = pool.apply_async(main, (input,))
#     # #     results.append(res)
#     # pool.map_async(main, inputs)
#     # pool.close()
#     # pool.join()
#     processes=[]
#     for input in inputs:
#         p=Process(target=predict, args=(input, ))
#         time.sleep(1)
#         processes.append(p)
#         p.start()
#     #
#     # # completing process
#     for p in processes:
#         p.join()
#
#
#
#
#     # for s,output in enumerate(output_list):
#     #     print("Step{}: {}".format(s+ 1, inputs[s]))
#     #     for i in output[0]:
#     #         print(i)
#     #     for i in output[1]:
#     #         print(i)
#
#
#     # for i, input in enumerate(inputs):
#     #     print("Step{}: {}".format(i+1, input))
#     #     rx_seq, sorted_ix= main(input)
#     #     for i in rx_seq:
#     #         print(i)
#     #     for i in sorted_ix:
#     #         print(i)
#     # main(_)
#     print("time elasped: {}".format(time.time()-st_time))