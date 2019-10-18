import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import numpy as np

import collections
from multiprocessing import Pool
import time
import subprocess

from utils import *
# from serving import tf_serving_client


# def parse_decfile_to_pathsmi(infile):
#     pathsmi = []
#     tmpseqs = collections.OrderedDict()
#     with open(infile) as f:
#         for i, line in enumerate(f.readlines()):
#             tmpseq_list = line.strip().split("\t")
#             for loc, seq in enumerate(tmpseq_list):
#                 canoseq = cano("".join(seq.split(" ")))
#                 if canoseq not in tmpseqs.keys():
#                     tmpseqs[canoseq] = 1
#                     pathsmi.append("RX%d_TOP%d,%s" % (i + 1, loc + 1, canoseq))
#                 else:
#                     tmpseqs[canoseq] += 1
#
#     totalnum = sum(list(tmpseqs.values()))
#     preds = np.array(list(tmpseqs.values())) * 1.0 / totalnum
#
#     return tmpseqs, pathsmi, preds

def parse_outlist_to_rxsmi(output_list, target_smi):
    pathsmi = []
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
                    # seq_score_dict[canoseq] = ["RX%d_TOP%d" % (rxid + 1, topid + 1), tmp_score]
                    seq_score_dict[canoseq] = tmp_score
                    # seq_totalscore_dict[canoseq] = \
                    #     ["RX%d_TOP%d" % (rxid + 1, topid + 1), smi_score[1],cal_changed_ring(target_smi, canoseq), cal_changed_smilen(target_smi, canoseq)]
                    seq_count_dict[canoseq] = 1
                    seq_label_dict[canoseq] = "RX%d_TOP%d" % (rxid + 1, topid + 1)
                else:
                    if tmp_score > seq_score_dict[canoseq]:
                        seq_label_dict[canoseq] = "RX%d_TOP%d" % (rxid + 1, topid + 1)
                        # seq_score_dict[canoseq] = ["RX%d_TOP%d" % (rxid + 1, topid + 1), tmp_score]
                        seq_score_dict[canoseq] = tmp_score
                        # seq_totalscore_dict[canoseq] = \
                        #     ["RX%d_TOP%d" % (rxid + 1, topid + 1), smi_score[1], cal_changed_ring(target_smi, canoseq), cal_changed_smilen(target_smi, canoseq)]
                    seq_count_dict[canoseq] += 1
    [pathsmi.append("%s,%s" % (seq_label_dict[key], key)) for key in seq_label_dict.keys()]

    # count_arr = np.array(list(seq_count_dict.values()))

    # the rate of reactant occurence as probability
    score_arr = np.array(list(seq_score_dict.values()))
    score_arr_norm = score_arr +( 0- np.min(score_arr))
    preds = score_arr_norm/ np.sum(score_arr_norm)
    return pathsmi, preds


def api_model_pred_reactant(input_smi, workdir, file, topk=True):
    outfile = os.path.join(workdir, "tmp", file)
    cmd = '/miniconda/bin/python serving/query_oneseq.py --inputs_once="{}" --work_path="{}"'.format \
        (input_smi, outfile)
    # outputs=query.predict(smilist_class)
    print(cmd)
    subprocess.call(cmd, shell=True)

    outputs = np.load(outfile).tolist()
    smi_nodes, prob_nodes = parse_outlist_to_rxsmi(outputs, cano(input_smi))

    if topk:
        pred_ix=np.argsort(-prob_nodes).tolist()[:10]
        smi_nodes_sorted=[smi_nodes[k] for k in pred_ix]
        smi_nodes=smi_nodes_sorted
        pred_nodes_sorted=[prob_nodes[k] for k in pred_ix]
        prob_nodes= np.array(pred_nodes_sorted)/np.sum(np.array(pred_nodes_sorted))

    return smi_nodes, prob_nodes


def client_server_prediction(x,y):
    return tf_serving_client.predict(x,y)

def api_model_pred_client(input_smi, workdir, file, topk=True):
    outfile = os.path.join(workdir, "tmp", file)
    client_server_prediction(input_smi, outfile)

    outputs = np.load(outfile).tolist()
    smi_nodes, prob_nodes = parse_outlist_to_rxsmi(outputs, cano(input_smi))

    if topk:
        pred_ix = np.argsort(-prob_nodes).tolist()[:5]
        smi_nodes_sorted = [smi_nodes[k] for k in pred_ix]
        smi_nodes = smi_nodes_sorted
        pred_nodes_sorted = [prob_nodes[k] for k in pred_ix]
        prob_nodes = np.array(pred_nodes_sorted) / np.sum(np.array(pred_nodes_sorted))

    return smi_nodes, prob_nodes

def expanded_node(state, opt):
    """------------------------------------------------------------------"""
    """expansion step"""
    """calculate how many nodes will be added under current leaf"""
    # tf_serving_client.load_config_request()
    start = time.time()
    position = []
    position.extend(state)
    x = get_main_part_from_smistr(position[-1].split(",")[-1])
    # smi_nodes, node_preds = api_model_pred_reactant(x, opt.input_dir,
    #         "iter{}_expand_s{}_n{}.npy".format(opt.iter_num, 0, 0))
    smi_nodes, node_preds = api_model_pred_reactant(x, opt.input_dir,\
                "iter{}_expand_s{}_n{}.npy".format(opt.iter_num, 0, 0))

    all_nodes_set= smi_nodes
    # all_nodes_set = list(set(all_nodes))
    print("expanded nodes:", all_nodes_set)
    print("Elapsed time of one node: {}s".format(time.time() - start))
    return all_nodes_set


def node_simulation(added_node, node_id, state, source_smi, workdir, iter_num, step_len):
    """------------------------------------------------------------------"""
    """rollout step"""
    """simulate node to rollout a random path"""
    print(added_node)
    position = []
    position.extend(state)
    position.append(added_node)
    total_generated = []

    get_int_old = []
    for j in range(len(position)):
        get_int_old.append(position[j])

    get_int = get_int_old
    cur_smi = position[-1].split(",")[-1]
    step = 1
    while not cur_smi == cano(source_smi):
        # for i in range(1):
        x = get_main_part_from_smistr(cur_smi)
        all_nodes, preds = api_model_pred_reactant(x, workdir,
                                                   "iter{}_rollout_n{}_s{}.npy".format(iter_num, node_id, step))
        # print(len(all_nodes),all_nodes)
        next_probas = np.random.multinomial(1, preds, 1)
        next_int = np.argmax(next_probas)
        get_int.append(all_nodes[next_int])
        cur_smi = get_int[-1].split(",")[-1]
        step += 1
        if len(get_int) > (step_len-1):
            break
    total_generated.append(get_int)

    return total_generated


def node_simulation_one(args):
    '''one parameter for mppj0=i'''
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
    print(all_possible)
    print("Elapsed time of all added nodes: {}s".format(time.time() - start))
    # sys.exit(0)
    return all_possible





def check_node_type(new_reaction, opt):
    source_file = os.path.join(opt.input_dir, opt.source_file)
    f = open(source_file)
    source_smi = f.readlines()[0].strip()
    f.close()
    node_index = []
    valid_reaction = []
    logp_value = []
    all_reaction = []
    distance = []
    # print "SA_mean:",SA_mean
    # print "SA_std:",SA_std
    # print "logP_mean:",logP_mean
    # print "logP_std:",logP_std
    # print "cycle_mean:",cycle_mean
    # print "cycle_std:",cycle_std
    activity = []
    score = []

    for i in range(len(new_reaction)):
        react_smi = []
        for class_smi in new_reaction[i]:
            react_smi.append(class_smi.split(",")[-1])
        reaction_smi = ".".join(react_smi)
        try:
            m = Chem.MolFromSmiles(reaction_smi)
        except:
            m = None
        if m != None and len(new_reaction[i]) <= 5:
            try:
                # path similarity
                # tmp_sim = [cal_similarity_with_FP(source_smi, smi) for smi in react_smi]
                # weight_arr = np.exp((np.array(range(-1 * len(tmp_sim), 0)) + 1) / len(tmp_sim))
                # sim_score = np.mean(np.array(tmp_sim) * weight_arr)
                #  if tmp_sim[-1] == 1:
                #     sim_score = 1

                # Similarity of the final output:
                sim_score = cal_similarity_with_FP(source_smi, react_smi[-1])

            except:
                sim_score = -1000

            node_index.append(i)
            valid_reaction.append(new_reaction[i])

            score_one = sim_score
            score.append(score_one)

        all_reaction.append(new_reaction[i])

    return node_index, score, valid_reaction, all_reaction
