# AutoSynRoute

Code for "Automatic Retrosynthetic Route Planning Using Template-Free Models" paper

http://#####

### Requirements
* python 3.6
* tensorflow 1.12.0
* tensor2tensor 1.13



### Model training
The data has already been preprocessed for training.

cd directory ```model/model_USPTO_50K``` 

1. Creating generator.

```bash
bash data_gen.sh t2t_data_class_char/ my_reaction_token dataset_50k_class_char/
```

2. Starting Traning.

```bash
bash data_trainer.sh t2t_data_class_char/ my_reaction_token 500000
```

3. Averaging checkpoints

```bash
t2t-avg-all --model_dir=t2t_data_class_char//train --output_dir=final_model/output_avg35000_class_char-n10-cp --n=10
```
Also, a pretrained model checkpoint can be found at:``` ####```.
Copy the folder into model/model_USPTO_MIT/final_model_class_char or model/model_USPTO_50K/final_model. 

### Model inference
The weights of trained model are available on:

model_USPTO_50K: url: https://pan.baidu.com/s/1XJg5Dh9zHnoXg1m_R6sJrA&shfl=shareset code: 28ng

model_USPTO_MIT: url: https://pan.baidu.com/s/1CabKTpU-jtdHKJfGTbdBrQ&shfl=shareset code: aiym

cd directory ```model/model_USPTO_50K```

```data_decoder_avg-beam-10.sh``` will perform inference with beam search, which will output a text file in ```model_USPTO_50K/t2t_data_class_char/train```


```bash
bash data_decoder_avg-beam-10.sh t2t_data_class_char/ my_reaction_token dataset_50k_class_char/ test_sources output_avg35000-top10_cp2.txt 80 final_model/output_avg35000_class_char-n10-cp/model.ckpt-35000
```
### Model evaluation
cd directory ```scripts```

The two python scripts ```"evaluation.py evaluation_class.py"``` will evaluate the total accuracy and accuracy by class, respectively. 

```bash
python evaluation.py -o ../data/USPTO/output_avg35000-top10_cp2.txt -t ../data/USPTO/test_targets_50K -c 12 -n 10 -d USPTO_50K
```
The result file can be found in ```results``` folder

```bash
python evaluation_class.py -o ../data/USPTO/output_avg35000-top10_cp2.txt -t ../data/USPTO/test_targets_50K -c 12 -n 10 -d USPTO_50K
```
The result file can be found in ```class_results``` folder

### Demo evaluation
cd directory ```scripts```

The ```demo_evaluation.py``` will evaluate the four demo cases mentioned in our paper.

```bash
python demo_evaluation.py -o ../data/demo/output_avg35000-top10_cp2_demo1_rufinamide.txt -t ../data/demo/demo1_rufinamide_cano_char_targets.txt -d 1
```
The result file can be found in ```demo_results``` folder
