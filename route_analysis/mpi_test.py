from multiprocessing import Pool
import os

def run_cmd(cmd):
    os.system(cmd)

list=['curl -d \'{"instances": [1.0, 2.0, 5.0]}\' -X POST http://localhost:8503/v1/models/half_plus_two:predict']*5


# pool=Pool(5)
# pool.map(run_cmd, list, chunksize=1)
run_cmd(list[0])


from tensor2tensor.data_generators import text_encoder

def _encode(inputs, encoder, add_eos=True):
  input_ids = encoder.encode(inputs)
  if add_eos:
    input_ids.append(text_encoder.EOS_ID)
  return input_ids

def _decode(output_ids, output_decoder):
  return output_decoder.decode(output_ids, strip_extraneous=True)

def predict(inputs_list, problem, request_fn):
  """Encodes inputs, makes request to deployed TF model, and decodes outputs."""
  assert isinstance(inputs_list, list)
  fname = "inputs" if problem.has_inputs else "targets"
  input_encoder = problem.feature_info[fname].encoder
  input_ids_list = [
      _encode(inputs, input_encoder, add_eos=problem.has_inputs)
      for inputs in inputs_list
  ]
  examples = [_make_example(input_ids, problem, fname)
              for input_ids in input_ids_list]
  # start = time.time()
  # predictions=[]
  # for example in examples:
  #     predictions.append(request_fn([example]))
  predictions = request_fn(examples)
  # elapsed=time.time()-start
  # print(elapsed)
  # print(predictions)
  output_decoder = problem.feature_info["targets"].encoder
  outputs=[]
  for prediction in predictions:
      one_outputs=[]
      for seq,score in zip(prediction["outputs"], prediction["scores"]):
          one_outputs.append([_decode(seq,output_decoder).split(" <EOS>")[0].replace(" ",""), score])
      outputs.append(one_outputs)
  return outputs
