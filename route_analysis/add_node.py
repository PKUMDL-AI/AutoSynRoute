import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import numpy as np
import collections
from multiprocessing import Pool
import time
import subprocess
from utils import *

def parse_outlist_to_smi(output_list, target_smi):
    pathsmi=[]
    seq_score_dict = collections.OrderedDict()
    seq_count_dict = collections.OrderedDict()
    seq_label_dict = collections.OrderedDict()

    for rxid, line in enumerate(output_list):
        for topid, smi_score in enumerate(line):
            canoseq = cano(smi_score[0])
            if canoseq !='' and (get_main_part_from_smistr(canoseq) != target_smi):
                rc_value = cal_changed_ring(target_smi, canoseq)
                lc_value = cal_changed_smilen(target_smi, canoseq)
                tmp_score=100*np.exp(smi_score[1])- (6 * rc_value + lc_value)
                if canoseq not in seq_score_dict.keys():
                    seq_score_dict[canoseq] = tmp_score
                    seq_count_dict[canoseq] = 1
                    seq_label_dict[canoseq] = "RX%d_TOP%d" % (rxid + 1, topid + 1)
                else:
                    if tmp_score > seq_score_dict[canoseq]:
                        seq_label_dict[canoseq] = "RX%d_TOP%d" % (rxid + 1, topid + 1)
                        seq_score_dict[canoseq] = tmp_score
                    seq_count_dict[canoseq] += 1
    for key in seq_label_dict.keys():
        pathsmi.append("%s,%s" % (seq_label_dict[key], key))

    score_arr = np.array(list(seq_score_dict.values()))
    score_arr_norm = score_arr - np.min(score_arr)
    preds = score_arr_norm/np.sum(score_arr_norm)
    return pathsmi, preds

def api_model_pred_reactant(input_smi, workdir, file, topk=True):
    outfile = os.path.join(workdir,'tmp',file)
    cmd = 'python serving/query_oneseq.py --inputs_once="{}" --work_path="{}"'.format \
        (input_smi, outfile)
    print(cmd)
    subprocess.call(cmd, shell=True)

    outputs=np.load(outfile, allow_pickle=True).tolist()
    smi_nodes, prob_nodes = parse_outlist_to_smi(outputs, cano(input_smi))
    print('prob_nodes: ', prob_nodes)
    if topk:
        pred_ix = np.argsort(-prob_nodes).tolist()[:10]
        smi_nodes_sorted = [smi_nodes[k] for k in pred_ix]
        prod_nodes_sorted=[prob_nodes[k] for k in pred_ix]
        prob_nodes_sorted = np.array(prod_nodes_sorted) / np.sum(np.array(prod_nodes_sorted))
    print('prob_nodes_sorted: ', prob_nodes_sorted)
    return smi_nodes_sorted,prob_nodes_sorted

def expanded_node(state, opt):
    """------------------------------------------------------------------"""
    """expansion step"""
    """calculate how many nodes will be added under current leaf"""
    start = time.time()
    position = []
    position.extend(state)
    x = get_main_part_from_smistr(position[-1].split(",")[-1])
    smi_nodes, node_probs = api_model_pred_reactant(x, opt.input_dir,\
                "iter{}_expand_s{}_n{}.npy".format(opt.iter_num, 0, 0))

    print("expanded nodes:", smi_nodes)
    print("Elapsed time of one node: {}s".format(time.time() - start))
    return smi_nodes

def node_simulation(added_node, node_id, state, source_smi, workdir, iter_num, step_len):
    """------------------------------------------------------------------"""
    """rollout step"""
    """simulate node to rollout a random path"""
    print(added_node)
    position = []
    position.extend(state)
    position.append(added_node)
    total_generated = []
    get_int = []
    for i in range(len(position)):
        get_int.append(position[i])

    cur_smi = position[-1].split(",")[-1]
    step = 1
    while not cur_smi == cano(source_smi):
        x = get_main_part_from_smistr(cur_smi)
        smi_nodes, node_probs = api_model_pred_reactant(x, workdir,
                                                   "iter{}_rollout_n{}_s{}.npy".format(iter_num, node_id, step))

        next_probs = np.random.multinomial(1, node_probs, 1)
        next_int = np.argmax(next_probs)
        get_int.append(smi_nodes[next_int])
        cur_smi = get_int[-1].split(",")[-1]
        step += 1
        if len(get_int) > step_len:
            break
    print("len(get_int): ", len(get_int), step_len)
    total_generated.append(get_int)
    return total_generated

def node_simulation_one(args):
    return node_simulation(*args)

def chemreact_kn_simulation(state, added_nodes, opt):
    all_possible = []
    source_file = os.path.join(opt.input_dir, opt.source_file)
    f = open(source_file)
    source_smi = f.readlines()[0].strip()
    f.close()
    start = time.time()
    num_nodes = len(added_nodes)
    print("Simulating {} nodes...".format(num_nodes))
    node_id_list = range(1, num_nodes + 1)
    zipped_list = [i for i in zip(added_nodes, node_id_list, [state] * num_nodes, [source_smi] * num_nodes,
                                  [opt.input_dir] * num_nodes, [opt.iter_num] * num_nodes, [opt.step_len]*num_nodes)]
    pool = Pool(12)
    total_reactions = pool.map(node_simulation_one, zipped_list, chunksize=1)
    pool.close()
    pool.join()

    [all_possible.extend(i) for i in total_reactions]
    #print(all_possible)
    print("Elapsed time of all added nodes: {}s".format(time.time() - start))

    return all_possible

def check_node_type(new_reaction, opt):
    source_file = os.path.join(opt.input_dir, opt.source_file)
    f = open(source_file)
    source_smi = f.readlines()[0].strip()
    f.close()
    node_index = []
    valid_reaction = []
    all_reaction = []
    score = []
    sim_score_ground_truth_list = []
    #cmd = 'python scaffold.py -i demo_b'
    cmd = 'python scaffold.py --input_dir={}'.format(opt.input_dir)
    print(cmd)
    '''if results exist, do not need to calculate again'''
    if os.path.exists('results/{}.txt'.format(opt.input_dir)):
        with open('results/{}.txt'.format(opt.input_dir)) as f:
            source_scaffold = f.readlines()
        source_scaf = source_scaffold[0].strip().split(',')[-1]
    else:
        subprocess.call(cmd, shell=True)
    print("new_reaction: ", new_reaction)
    for i in range(len(new_reaction)):
        react_smi = []
        print(new_reaction[i])
        for class_smi in new_reaction[i]:
            react_smi.append(class_smi.split(",")[-1])
        reaction_smi = ".".join(react_smi)

        try:
            m = Chem.MolFromSmiles(reaction_smi)
        except:
            m = None
        #if m != None and len(new_reaction[i]) <= opt.step_len:
        if m != None and len(new_reaction[i]) <= 7:
            try:
                #sim_score = cal_similarity_with_FP(source_smi, react_smi[-1])
                sim_score = cal_similarity_with_FP(source_scaf, react_smi[-1])
                sim_score_ground_truth = cal_similarity_with_FP(source_smi, react_smi[-1])
                #sim_scaf_ground_truth = cal_similarity_with_FP(source_smi, source_scaf)
            except:
                sim_score = -1000
                sim_score_ground_truth = -10
            node_index.append(i)
            valid_reaction.append(new_reaction[i])
            score_one = sim_score
            score.append(score_one)
            sim_score_ground_truth_list.append(sim_score_ground_truth)
        all_reaction.append(new_reaction[i])
    print(sim_score_ground_truth_list)
    print(score)
    return node_index, score, valid_reaction, all_reaction,sim_score_ground_truth_list



