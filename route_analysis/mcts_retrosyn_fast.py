import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"]="3"
from math import *
import numpy as np
import random as pr
import time
import argparse
#from add_node_type_fast import chemreact_kn_simulation, check_node_type, expanded_node
from add_node import chemreact_kn_simulation, check_node_type, expanded_node

class chemicalreaction():

    def __init__(self):
        self.position=[target_smi]

    def Clone(self):

        st = chemicalreaction()
        st.position= self.position[:]
        return st

    def SelectPosition(self,m):

        self.position.append(m)


class Node:

    def __init__(self, position = None,  parent = None, state = None):
        self.position = position
        self.parentNode = parent
        self.childNodes = []
        self.child=None
        self.wins = 0
        self.visits = 0
        self.depth=0


    def Selectnode(self):
        ucb=[]
		#UCT
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


    def Update(self, result):

        self.visits += 1
        self.wins += result

def MCTS(root, verbose = False):
    output_dir = os.path.join(opt.input_dir, opt.output_dir)
    if not os.path.exists(os.path.join(opt.input_dir, opt.output_dir)):
        os.mkdir(output_dir)
    """initialization of the chemical reaction trees"""
    if os.path.exists('{}/{}/mcts_tree_final.pkl'.format(opt.input_dir,opt.output_dir)):
        import pickle
        with open('{}/{}/mcts_tree_final.pkl'.format(opt.input_dir,opt.output_dir), 'rb') as f:
            rootnode=pickle.load(f)
        print("Load root node successfully.")
    else:
        rootnode = Node(state = root)
    # state = root.Clone()
    maxnum=0
    iteration_num=0
    start_time1=time.time()
    """----------------------------------------------------------------------"""


    """global variables used for save valid compounds and simulated compounds"""
    all_valid_reaction=[]
    all_simulated_reaction=[]
    all_reaction_score=[]
    max_score=-100.0
    current_score=[]
    depth=[]
    all_score=[]
    iter_num=1


    """----------------------------------------------------------------------"""

    while maxnum<opt.total_iter_num:
        print(maxnum)
        opt.iter_num=iter_num
        node = rootnode
        state = root.Clone()
        """selection step"""
        node_pool=[]
        print("current found max_score:",max_score)

        while node.childNodes!=[]:
            node = node.Selectnode()
            state.SelectPosition(node.position)
        print("state position:",state.position)
        depth.append(len(state.position))
        if len(state.position)>=opt.step_len:
            print("{}: Current state have {} steps!".format(maxnum, len(state.position)))
            maxnum += 1
            continue
        #     while node != None:
        #         node.Update(re)
        #         node = node.parentNode

        """------------------------------------------------------------------"""
        """expansion step"""
        """calculate how many nodes will be added under current leaf"""

        expanded=expanded_node(state.position,opt)
        nodeadded=expanded


        all_posible=chemreact_kn_simulation(state.position, nodeadded, opt)
        new_reaction=all_posible

        node_index,score,valid_reaction,all_reaction,sim_score_ground_truth_list=check_node_type(new_reaction, opt)
        print(valid_reaction, score)



        print(node_index)
        all_valid_reaction.extend(valid_reaction)
        all_simulated_reaction.extend(all_reaction)
        all_score.extend(score)
        # iteration_num=len(all_simulated_reaction)
        iter_num += 1
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
                re.append((0.8*score[i])/(1.0+0.8*score[i]))
                if maxnum==100:
                    maxscore100=max_score
                    time100=time.time()-start_time1
                if maxnum==500:
                    maxscore500=max_score
                    time500=time.time()-start_time1
                if maxnum==1000:

                    maxscore1000=max_score
                    time1000=time.time()-start_time1
                if maxnum==5000:
                    maxscore5000=max_score
                    time5000=time.time()-start_time1
                if maxnum==10000:
                    time10000=time.time()-start_time1
                    maxscore10000=max_score

            """backpropation step"""

            for i in range(len(node_pool)):

                node=node_pool[i]
                while node != None:
                    node.Update(re[i])
                    node = node.parentNode

       # if max(score) == 1:
        if sim_score_ground_truth_list!=[] and max(sim_score_ground_truth_list)==1: 
            iteration_num +=1
            #ix_max = np.argmax(np.array(score))
            ix_max = np.argmax(np.array(sim_score_ground_truth_list))
            print("The best path: {},{}".format(valid_reaction[ix_max], score[ix_max]))
            #[all_reaction_score.append([x, y]) for x, y in zip(all_valid_reaction, all_score)]
            np.save("{}/{}/output_react_score_{}.npy".format(opt.input_dir,opt.output_dir, iteration_num), np.array((valid_reaction[ix_max], score[ix_max]), dtype=object))
            import pickle
            with open('{}/{}/mcts_tree_{}.pkl'.format(opt.input_dir,opt.output_dir, iteration_num), 'wb') as f:
                pickle.dump(rootnode, f, pickle.HIGHEST_PROTOCOL)

            with open('{}/{}/mcts_tree_final.pkl'.format(opt.input_dir,opt.output_dir), 'wb') as f:
                pickle.dump(rootnode, f, pickle.HIGHEST_PROTOCOL)

        """check if found the desired compound"""

    finished_run_time=time.time()-start_time1

    print("Similarity max found:", current_score)

    print("valid_reaction=",all_valid_reaction)
    print("num_valid:", len(all_valid_reaction))
    print("all reactions:",len(all_simulated_reaction))
    print("score=", all_score)
    print("depth=",depth)
    print(len(depth))
    print("runtime",finished_run_time)
    print("100 max:",maxscore100,time100)
    # print("500 max:",maxscore500,time500)
    # print("1000 max:",maxscore1000,time1000)
    # print("5000 max:",maxscore5000,time5000)
    # print("10000 max:",maxscore10000,time10000)
    [all_reaction_score.append([x,y])for x,y in zip(all_valid_reaction,all_score)]
    np.save("output_react_score.npy" , np.array(all_reaction_score, dtype=object))
    import pickle
    with open('{}/mcts_tree_final.pkl'.format(opt.input_dir),'wb') as f:
        pickle.dump(rootnode, f,pickle.HIGHEST_PROTOCOL)

    return all_reaction_score

def UCTchemicalreaction():
    one_search_start_time=time.time()
    # time_out=one_search_start_time+60*10
    state = chemicalreaction()
    best = MCTS(root = state,verbose = False)

    return best


if __name__ == "__main__":
    import os
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, default="demo_b", help='')
    parser.add_argument('--output_dir', type=str, default="route_output", help='')
    parser.add_argument('--input_file', type=str, default="target_iter0.smi", help='')
    parser.add_argument('--source_file', type=str, default="ref_mol.smi", help='')
    # parser.add_argument('--output_json_file', type=str,default="t2s_json_out.json", help='')
    parser.add_argument('--rx_num', type=int, default=10, help='')
    parser.add_argument('--step_len', type=int, default=6, help='')
    parser.add_argument('--total_iter_num', type=int, default=50001, help='')



    opt = parser.parse_args()
    target_file = os.path.join(opt.input_dir, opt.input_file)
    f=open(target_file)
    target_smi=f.readlines()[0].strip()
    f.close()
    print(target_smi)

    valid_reaction=UCTchemicalreaction()
    print(valid_reaction)



