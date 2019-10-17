from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re
import collections
# from tensor2tensor.bin import t2t_trainer
from tensor2tensor.data_generators import problem
from tensor2tensor.data_generators import text_problems
from tensor2tensor.data_generators import text_encoder
from tensor2tensor.utils import registry
import os,sys
import tensorflow as tf

EOS = text_encoder.EOS
# vs=200
#
# def _read_words(filename):
#   """Reads words from a file."""
#   with tf.gfile.GFile(filename, "r") as f:
#     if sys.version_info[0] >= 3:
#       return f.read().replace("\n", " %s " % EOS).split()
#     else:
#       return f.read().decode("utf-8").replace("\n", " %s " % EOS).split()
#
# def _build_vocab(filename, vocab_path, vocab_size):
#   """Reads a file to build a vocabulary of `vocab_size` most common words.
#    The vocabulary is sorted by occurrence count and has one word per line.
#    Originally from:
#    https://github.com/tensorflow/models/blob/master/tutorials/rnn/ptb/reader.py
#   Args:
#     filename: file to read list of words from.
#     vocab_path: path where to save the vocabulary.
#     vocab_size: size of the vocabulary to generate.
#   """
#   data = _read_words(filename)
#   counter = collections.Counter(data)
#   count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))
#   words, _ = list(zip(*count_pairs))
#   words = words[:vocab_size]
#   with open(vocab_path, "w") as f:
#     f.write("\n".join(words))


def _get_token_encoder(vocab_path):
  """Reads from file and returns a `TokenTextEncoder` for the vocabulary."""
  return text_encoder.TokenTextEncoder(vocab_path,replace_oov='<UNK>')


@registry.register_problem
class MyReactionToken(text_problems.Text2TextProblem):
  #Predict next line of poetry from the last line. From Gutenberg texts.

  @property
  def vocab_filename(self):
      return "vocab.token"

  @property
  def vocab_type(self):
      return text_problems.VocabType.TOKEN

  @property
  def approx_vocab_size(self):
    return 2**9  # ~256

  @property
  def is_generate_per_split(self):
    # generate_data will shard the data into TRAIN and EVAL for us.
    return True

  @property
  def dataset_splits(self):
    #Splits of data to produce and number of output shards for each.
    # 10% evaluation data
    return [{
        "split": problem.DatasetSplit.TRAIN,
        "shards": 10,
    }, {
        "split": problem.DatasetSplit.EVAL,
        "shards": 1,
    }]

  def generate_samples(self, data_dir, tmp_dir, dataset_split):

    # train_file=os.path.join(raw_data_dir, "train_both.txt")
    train_in_file = os.path.join(tmp_dir, "train_sources")
    train_out_file = os.path.join(tmp_dir, "train_targets")
    in_r = open(train_in_file)
    out_r = open(train_out_file)

    in_list = in_r.readlines()
    out_list = out_r.readlines()
    # print(data_dir)
    _get_token_encoder(os.path.join(data_dir, self.vocab_filename))
    # text_encoder.SubwordTextEncoder(os.path.join(data))

    in_r.close()
    out_r.close()
    for line1, line2 in zip(in_list, out_list):
        input_line=" ".join(line1.replace("\n", " %s " % EOS).split())
        targets_line= " ".join(line2.replace("\n", " %s " % EOS).split())
        yield {
          "inputs": input_line,
          "targets": targets_line,
        }
