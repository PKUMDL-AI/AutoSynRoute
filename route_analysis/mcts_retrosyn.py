from subprocess import Popen, PIPE
from math import *
import random
import numpy as np
import random as pr
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
import sys
# from make_smile import zinc_data_with_bracket_original, zinc_processed_with_bracket
from add_node_type import chemreact_kn_simulation, check_node_type,expanded_node



class chemicalreaction():

    def __init__(self):
        self.position=[target_smi]

    def Clone(self):

        st = chemicalreaction()
        st.position= self.position[:]
        return st

    def SelectPosition(self,m):

        self.position.append(m)

    def Getatom(self):
        return [i for i in range(self.num_atom)]

class Node:

    def __init__(self, position = None,  parent = None, state = None):
        self.position = position
        self.parentNode = parent
        self.childNodes = []
        self.child=None
        self.wins = 0
        self.visits = 0
        #self.nonvisited_atom=state.Getatom()
        #self.type_node=tp
        self.depth=0


    def Selectnode(self):
        ucb=[]
        for i in range(len(self.childNodes)):
            ucb.append(self.childNodes[i].wins/self.childNodes[i].visits+1.0*sqrt(2*log(self.visits)/self.childNodes[i].visits))
        m = np.amax(ucb)
        indices = np.nonzero(ucb == m)[0]
        ind=pr.choice(indices)
        s=self.childNodes[ind]
        return s

    def Addnode(self, m, s):

        n = Node(position = m, parent = self, state = s)
        self.childNodes.append(n)

    def simulation(self,state):
        predicted_smile=predict_smile(model,state)
        input_smile=make_input_smile(predicted_smile)
        logp,valid_smile,all_smile=logp_calculation(input_smile)

        return logp,valid_smile,all_smile

    def Update(self, result):

        self.visits += 1
        self.wins += result

def MCTS(root, verbose = False):

    """initialization of the chemical reaction trees"""
    run_time=time.time()+600*2
    rootnode = Node(state = root)
    state = root.Clone()
    maxnum=0
    iteration_num=0
    start_time=time.time()
    """----------------------------------------------------------------------"""


    """global variables used for save valid compounds and simulated compounds"""
    valid_reaction=[]
    all_simulated_reaction=[]
    desired_compound=[]
    max_score=-100.0
    desired_activity=[]
    time_distribution=[]
    num_searched=[]
    current_score=[]
    depth=[]
    all_score=[]


    """----------------------------------------------------------------------"""

    while maxnum<101:
        print(maxnum)
        node = rootnode
        state = root.Clone()
        """selection step"""
        node_pool=[]
        print("current found max_score:",max_score)

        while node.childNodes!=[]:
            node = node.Selectnode()
            state.SelectPosition(node.position)
        print("state position:,",state.position)
        depth.append(len(state.position))
        if len(state.position)>=5:
            re=-1.0
            while node != None:
                node.Update(re)
                node = node.parentNode
        else:
            """------------------------------------------------------------------"""

            """expansion step"""
            """calculate how many nodes will be added under current leaf"""
            expanded=expanded_node(state.position,opt)
            nodeadded=expanded


            all_posible=chemreact_kn_simulation(state.position, nodeadded, opt)
            new_reaction=all_posible

            node_index,score,valid_reaction,all_reaction=check_node_type(new_reaction, opt)

            print(node_index)
            valid_reaction.extend(valid_reaction)
            all_simulated_reaction.extend(all_reaction)
            all_score.extend(score)
            iteration_num=len(all_simulated_reaction)
            if len(node_index)==0:
                re=-1.0
                while node != None:
                    node.Update(re)
                    node = node.parentNode
            else:
                re=[]
                for i in range(len(node_index)):
                    m=node_index[i]
                    maxnum=maxnum+1
                    node.Addnode(nodeadded[m],state)
                    node_pool.append(node.childNodes[i])
                    if score[i]>=max_score:
                        max_score=score[i]
                        current_score.append(max_score)
                    else:
                       current_score.append(max_score)
                    depth.append(len(state.position))
                    """simulation"""
                    re.append((0.8*score[i])/(1.0+abs(0.8*score[i])))
                    if maxnum==100:
                        maxscore100=max_score
                        time100=time.time()-start_time
                    if maxnum==500:
                        maxscore500=max_score
                        time500=time.time()-start_time
                    if maxnum==1000:

                        maxscore1000=max_score
                        time1000=time.time()-start_time
                    if maxnum==5000:
                        maxscore5000=max_score
                        time5000=time.time()-start_time
                    if maxnum==10000:
                        time10000=time.time()-start_time
                        maxscore10000=max_score
                        #valid10000=10000*1.0/len(all_simulated_reaction)


                """backpropation step"""
                #print "node pool length:",len(node.childNodes)

                for i in range(len(node_pool)):

                    node=node_pool[i]
                    while node != None:
                        node.Update(re[i])
                        node = node.parentNode

            #finish_iteration_time=time.time()-iteration_time
            #print "four step time:",finish_iteration_time






        """check if found the desired compound"""

    #print "all valid compounds:",valid_reaction

    finished_run_time=time.time()-start_time

    print("Similarity max found:", current_score)
    #print "length of score:",len(current_score)
    #print "time:",time_distribution

    print("valid_reaction=",valid_reaction)
    print("num_valid:", len(valid_reaction))
    print("all reactions:",len(all_simulated_reaction))
    print("score=", all_score)
    print("depth=",depth)
    print(len(depth))
    print("runtime",finished_run_time)
    #print "num_searched=",num_searched
    print("100 max:",maxscore100,time100)
    print("500 max:",maxscore500,time500)
    print("1000 max:",maxscore1000,time1000)
    print("5000 max:",maxscore5000,time5000)
    print("10000 max:",maxscore10000,time10000)
    return(valid_reaction)

def UCTchemicalreaction():
    one_search_start_time=time.time()
    time_out=one_search_start_time+60*10
    state = chemicalreaction()
    best = MCTS(root = state,verbose = False)


    return best


if __name__ == "__main__":
    import os
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, default="mol1", help='')
    parser.add_argument('--input_file', type=str, default="target_iter0.smi", help='')
    parser.add_argument('--source_file', type=str, default="ref_mol.smi", help='')
    # parser.add_argument('--output_json_file', type=str,default="t2s_json_out.json", help='')
    parser.add_argument('--rx_num', type=int, default=10, help='')

    parser.add_argument('--target_sim_cutoff', type=float, default=0.8, help='')
    parser.add_argument('--source_sim_cutoff', type=float, default=0.5, help='')
    parser.add_argument('--source_sim_step', type=float, default=0.2, help='')
    # parser.add_argument('--process_file', type=str,default="error_token1.json", help='')

    # parser.add_argument('--out1_file', type=str,default="output_token1_r1.json", help='')
    # parser.add_argument('--out2_file', type=str,default="error_token1_r1.json", help='')
    # parser.add_argument('--iter_num', type=int,default=1, help='')

    opt = parser.parse_args()
    target_file = os.path.join(opt.input_dir, opt.input_file)
    target_smi=open(target_file).readlines()[0].strip()
    print(target_smi)


    # logP_values = np.loadtxt('logP_values.txt')
    # SA_scores = np.loadtxt('SA_scores.txt')
    # cycle_scores = np.loadtxt('cycle_scores.txt')
    # SA_mean =  np.mean(SA_scores)
    # print len(SA_scores)
    #
    # SA_std=np.std(SA_scores)
    # logP_mean = np.mean(logP_values)
    # logP_std= np.std(logP_values)
    # cycle_mean = np.mean(cycle_scores)
    # cycle_std=np.std(cycle_scores)
    # #val2=['C', '(',  'c', '1',  'o', '=', 'O', 'N', 'F', '[C@@H]', 'n',  'S', 'Cl', '[O-]']
    # #val2=['C', 'c','#', '3', '(', '2', 'n', 'O', '/', 'N', '=', '\\', ')', '1', 'o', '4', 's', '[C@H]', 'F', 'S', 'Cl', '[C@@H]', '[C@@]', '[C@]', '5', '#', '[nH]', 'Br', 'I', '6', '-', '[NH+]', '[N-]', '[N+]', '[n+]', '[nH+]', '[NH2+]','[NH3+]']
    # #val=['\n', '&', 'C', '(', ')', 'c', '1', '2', 'o', '=', 'O', 'N', '3', 'F', '[C@@H]', 'n', '#', 'S', 'Cl', '[O-]', '[C@H]>
    # val2=['C', '(', ')', 'c', '1', '2', 'o', '=', 'O', 'N', '3', 'F', '[C@@H]', 'n', '-', '#', 'S', 'Cl', '[O-]', '[C@H]', '[NH+]', '[C@]', 's', 'Br', '/', '[nH]', '[NH3+]', '4', '[NH2+]', '[C@@]', '[N+]', '[nH+]', '\\', '[S@]', '5', '[N-]', '[n+]', '[S@@]', '[S-]', '6', '7', 'I', '[n-]', 'P', '[OH+]', '[NH-]', '[P@@H]', '[P@@]', '[PH2]', '[P@]', '[P+]', '[S+]', '[o+]', '[CH2-]', '[CH-]', '[SH+]', '[O+]', '[s+]', '[PH+]', '[PH]', '8', '[S@@+]']

    # model=loaded_model()
    #acitivity_model=loaded_activity_model()
    valid_reaction=UCTchemicalreaction()
    print(valid_reaction)
