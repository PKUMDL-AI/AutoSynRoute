from subprocess import Popen, PIPE
from math import *
import random,os
import numpy as np
from copy import deepcopy
# from types import IntType, ListType, TupleType, StringTypes
import itertools
import time
import math
import argparse
import subprocess
# from load_model import loaded_model
# from keras.preprocessing import sequence
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem import Descriptors
from rdkit.Chem import MolFromSmiles, MolToSmiles
from rdkit import DataStructs
import sys
from rdkit.Chem import AllChem
# from make_smile import zinc_data_with_bracket_original, zinc_processed_with_bracket
# import sascorer
import pickle
import gzip
import networkx as nx
from rdkit.Chem import rdmolops

import collections



def cano(smiles):  # canonicalize smiles by MolToSmiles function
    try:
        canosmi=Chem.MolToSmiles(Chem.MolFromSmiles(smiles))
    except:
        canosmi=""
    return canosmi

def get_main_part_from_smistr(tmpseqs):
    seq_list=tmpseqs.split(".")
    if len(seq_list)==1:
        main_smi= tmpseqs
    else:
        main_smi= max(seq_list, key=len)
    return main_smi

def write_list_to_file(list,outfile):
    with open(outfile, "w") as f:
       f.write("\n".join(list)+"\n")
    return  outfile

def parse_decfile_to_pathsmi(infile):
    pathsmi=[]
    tmpseqs=collections.OrderedDict()
    with open(infile) as f:
        for i, line in enumerate(f.readlines()):
            tmpseq_list=line.strip().split("\t")
            for loc,seq in enumerate(tmpseq_list):
                canoseq=cano("".join(seq.split(" ")))
                if canoseq not in tmpseqs.keys():
                    tmpseqs[canoseq]=1
                    pathsmi.append("RX%d_TOP%d,%s" %(i+1, loc+1, canoseq))
                else:
                    tmpseqs[canoseq] +=1

    totalnum=sum(list(tmpseqs.values()))
    preds=np.array(list(tmpseqs.values()))*1.0/totalnum

    return tmpseqs, pathsmi, preds

def model_predict_reactant(input_smi, opt):
    # pythondir = "/lustre1/lhlai_pkuhpc/wangsw/software/anaconda3/bin"
    pythondir = "/work01/home/swwang/software/anaconda3/bin"
    train_dir = "t2t_data_class_char"
    problem = "my_reaction_token"
    work_data_dir = "route_analysis"
    trained_model = "finalmodel_class_char/avg500000_class_char_n20/model.ckpt-500000"

    rx_num=10
    smilist_class=[]
    for j in range(1, rx_num + 1):
        smilist_class.append("<RX_%d> %s" % (j, " ".join(list(input_smi))))

    tokenfile = "%s/tmpclass_%s" % (opt.input_dir, opt.input_file)
    write_list_to_file(smilist_class, tokenfile)

    outfile1 = "tmpdecoded_%s" % (opt.input_file)
    cmdstr = "export PATH=$PATH:%s; export CUDA_VISIBLE_DEVICES='1'; cd ..; bash data_decoder_avg.sh %s %s %s %s %s 64 %s" % (pythondir, train_dir, problem, work_data_dir, tokenfile, outfile1, trained_model)
    print(cmdstr)
    os.system(cmdstr)
    cmdstr1 = 'mv ../%s/train/%s ./%s' % (train_dir, outfile1, opt.input_dir)
    os.system(cmdstr1)

    decout_file='%s/%s' % (opt.input_dir, outfile1)

    all_nodes_dict, all_class_nodes, node_preds= parse_decfile_to_pathsmi(decout_file)

    return all_nodes_dict, all_class_nodes, node_preds


def expanded_node(state, opt):
    position = []
    position.extend(state)
    x= get_main_part_from_smistr(position[-1].split(",")[-1])
    all_nodes, all_class_nodes,_=model_predict_reactant(x, opt)
    print(all_class_nodes)
    return all_class_nodes


def chemreact_kn_simulation(state, added_nodes,opt):
    all_posible = []

    source_file = os.path.join(opt.input_dir, opt.source_file)
    source_smi = open(source_file).readlines()[0].strip()

    for i in range(len(added_nodes)):
        #position=[]
        position=[]
        position.extend(state)
        position.append(added_nodes[i])
        total_generated = []

        get_int_old = []
        for j in range(len(position)):
            get_int_old.append(position[j])

        get_int = get_int_old
        x = get_main_part_from_smistr(position[-1].split(",")[-1])

        while not get_int[-1].split(",")[-1] == cano(source_smi):
            all_nodes, all_class_nodes, preds = model_predict_reactant(x, opt)
            next_probas = np.random.multinomial(1, preds, 1)
            next_int = np.argmax(next_probas)
            get_int.append(all_class_nodes[next_int])
            x= get_main_part_from_smistr(get_int[-1].split(",")[-1])
            if len(get_int) > 4:
                break

        total_generated.append(get_int)
        all_posible.extend(total_generated)


    return all_posible

def cal_similarity_with_FP(source_seq, target_seq):
    try:
        mol1= Chem.MolFromSmiles(source_seq)
        mol2= Chem.MolFromSmiles(target_seq)
        mol1_fp= AllChem.GetMorganFingerprint(mol1, 2)
        mol2_fp= AllChem.GetMorganFingerprint(mol2, 2)
        score=DataStructs.DiceSimilarity(mol1_fp,mol2_fp)
        return score
    except:
        return 0

def check_node_type(new_reaction,opt):
    source_file = os.path.join(opt.input_dir, opt.source_file)
    source_smi = open(source_file).readlines()[0].strip()
    node_index=[]
    valid_reaction=[]
    logp_value=[]
    all_reaction=[]
    distance=[]
    #print "SA_mean:",SA_mean
    #print "SA_std:",SA_std
    #print "logP_mean:",logP_mean
    #print "logP_std:",logP_std
    #print "cycle_mean:",cycle_mean
    #print "cycle_std:",cycle_std
    activity=[]
    score=[]

    for i in range(len(new_reaction)):
        react_smi=[]
        for class_smi in new_reaction[i]:
            react_smi.append(class_smi.split(",")[-1])
        reaction_smi=".".join(react_smi)
        try:
            m = Chem.MolFromSmiles(reaction_smi)
        except:
            m = None
        if m!=None and len(new_reaction[i])<=5:
            try:

                tmp_sim=[cal_similarity_with_FP(source_smi,smi) for smi in react_smi]
                sim_score=sum(tmp_sim)
            except:
                sim_score=-1000

            node_index.append(i)
            valid_reaction.append(new_reaction[i])

            score_one = sim_score
            score.append(score_one)

        all_reaction.append(new_reaction[i])

    return node_index,score,valid_reaction,all_reaction

