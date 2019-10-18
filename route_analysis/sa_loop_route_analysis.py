
import json,sys, os, argparse
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs
from treelib import Node, Tree

import numpy as np
# import copy


class_list=['<RX_1>','<RX_2>','<RX_3>','<RX_4>','<RX_5>','<RX_6>','<RX_7>','<RX_8>','<RX_9>','<RX_10>',]
# scriptname="s1_token_process.py"
pythondir="/lustre1/lhlai_pkuhpc/wangsw/software/anaconda3/bin"
train_dir="t2t_data_class_char"
problem="my_reaction_token"
work_data_dir="route_analysis"
trained_model="finalmodel_class_char/avg500000_class_char_n20/model.ckpt-500000"



parser = argparse.ArgumentParser()
parser.add_argument('--input_dir', type=str, default="mol1", help='')
parser.add_argument('--input_file', type=str, default="target_iter0.smi", help='')
parser.add_argument('--source_file', type=str, default="ref_mol.smi", help='')
# parser.add_argument('--output_json_file', type=str,default="t2s_json_out.json", help='')
parser.add_argument('--rx_num', type=int,default=10, help='')

parser.add_argument('--target_sim_cutoff', type=float,default=0.8, help='')
parser.add_argument('--source_sim_cutoff', type=float,default=0.5, help='')
parser.add_argument('--source_sim_step', type=float,default=0.2, help='')
# parser.add_argument('--process_file', type=str,default="error_token1.json", help='')

# parser.add_argument('--out1_file', type=str,default="output_token1_r1.json", help='')
# parser.add_argument('--out2_file', type=str,default="error_token1_r1.json", help='')
# parser.add_argument('--iter_num', type=int,default=1, help='')


opt = parser.parse_args()

target_file=os.path.join(opt.input_dir,opt.input_file)
source_file=os.path.join(opt.input_dir,opt.source_file)

ref_smi=open(source_file).readlines()[0].strip()



rx_num=opt.rx_num
beam_size=10
mytree = Tree()
mytree1 = Tree()


def cano(smiles):  # canonicalize smiles by MolToSmiles function
    return Chem.MolToSmiles(Chem.MolFromSmiles(smiles)) if (smiles != '') else ''

class Nodex(object):
    def __init__(self, content):
        self.content = content


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

def file_to_list(infile):
    in_list=[]
    with open(infile) as f:
        for line in f.readlines():
            in_list.append(" ".join(list(line.strip())))
    return in_list

def copy_list_add_class(inlist):
    outlist=[]
    for i in inlist:
        for j in range(1, rx_num+1):
            outlist.append("<RX_%d> %s" % (j,i))

    return outlist

def convert_smi_to_decfile(infile,i):
    inlist=file_to_list(infile)
    if i==0:
        mytree.create_node("Target", "target", data=Nodex("".join(inlist[0].split(" "))))
        mytree1.create_node("Target", "target", data="".join(inlist[0].split(" ")))

    inlist_class= copy_list_add_class(inlist)
    outfile = "%s/loop%d_%s" % (opt.input_dir, i, infile.split("/")[-1])
    with open(outfile, "w") as f:
       f.write("\n".join(inlist_class)+"\n")
    return  outfile




def decoding_decfile_to_file(tokenfile):
    outfile1 = "decoded_%s" % (tokenfile.split("/")[-1])
    cmdstr = "export PATH=$PATH:%s;cd ..;bash data_decoder_avg.sh %s %s %s %s %s 64 %s" % (
    pythondir, train_dir, problem, work_data_dir, tokenfile, outfile1, trained_model)
    print(cmdstr)
    os.system(cmdstr)
    cmdstr1='mv ../%s/train/%s ./%s' % (train_dir, outfile1,opt.input_dir)
    os.system(cmdstr1)

    return '%s/%s' %(opt.input_dir, outfile1)

def cal_topseqs_oneseqindiffclass(tmp_score_arr, tmp_seq_arr, layerid, nodeid, topk):
    top_index_list = list(tmp_score_arr.reshape(-1).argsort()[::-1][:topk])
    print(top_index_list)
    top_seq_list = []
    for loc1, topix in enumerate(top_index_list):
        m = int(topix / beam_size)
        n = topix % beam_size
        print(m, n)
        parentnode = "target" if layerid == 1 else "l%d_n%d" % (layerid - 1, nodeid)
        tagname= "L%d_N%d" %(layerid, loc1)
        nodename= "l%d_n%d" %(layerid, loc1)

        datavalue="RX%d_TOP%d,%s" %(m+1, n+1, str(tmp_seq_arr[m,n]))

        mytree.create_node(tagname, nodename, parent=parentnode, data=Nodex(datavalue))
        mytree1.create_node(tagname, nodename, parent=parentnode, data=datavalue)
        top_seq_list.append(str(tmp_seq_arr[m,n]).split(",")[0])

    return top_seq_list


def get_main_part_fromseqs(tmpseqs):
    need_seqs=[]
    for seq in tmpseqs:
        seq_list=seq.split(",")[1].split(".")
        if len(seq_list)==1:
            need_seqs.append(seq.split(",")[1])
        else:
            need_seqs.append(max(seq_list, key=len))

    return need_seqs



def cal_topseqs_oneseqindiffclass1(tmp_score_arr, tmp_seq_arr, topk):
    ranked_index_list = list(tmp_score_arr.reshape(-1).argsort()[::-1])
    print(ranked_index_list)
    top_seq_list = []
    only_seq_list=[]
    for loc1, topix in enumerate(ranked_index_list):
        m = int(topix / beam_size)
        n = topix % beam_size
        print(m, n)
        tmpseq=str(tmp_seq_arr[m, n]).split(",",1)
        if cano(tmpseq[0]) not in only_seq_list:
            only_seq_list.append(cano(tmpseq[0]))
            top_seq_list.append("RX%d_TOP%d,%s,%s" %(m+1, n+1, cano(tmpseq[0]), tmpseq[1]))
        if len(only_seq_list)==topk:
            return top_seq_list


def parse_decedfile_to_tree(tokenfile, decedfile, iter):

    input_seqs=[]
    with open(tokenfile) as f:
        for line in f.readlines():
            input_seqs.append("".join(line.strip().split(" ")[1:]))

    # record score and seqs with array
    # num_diffseq= int(len(input_seqs)/rx_num)
    all_score_arr=np.zeros([len(input_seqs),beam_size])
    all_seq_arr=np.empty([len(input_seqs),beam_size],dtype=object)
    with open(decedfile) as f:
        for ix,line in enumerate(f.readlines()):
            # ixm=int(ix/10)
            tmp_decseqs=line.strip().split("\t")
            for loc,seq in enumerate(tmp_decseqs):
                tmpsmi="".join(seq.split(" "))
                sim_score1=cal_similarity_with_FP(input_seqs[ix], tmpsmi)
                sim_score1= 0 if sim_score1 > opt.target_sim_cutoff else sim_score1
                sim_score2=cal_similarity_with_FP(ref_smi, tmpsmi)
                sim_score=sim_score1+((opt.source_sim_cutoff + opt.source_sim_step*iter)*sim_score2)
                all_score_arr[ix,loc]=sim_score*(1-loc*0.1)
                all_seq_arr[ix,loc]=tmpsmi+",%f,%f,%f"%(sim_score1,sim_score2,sim_score)+","+str(loc)+","+str(all_score_arr[ix,loc])

    print(all_score_arr)
    print(all_score_arr.shape)
    all_score_arr=all_score_arr.reshape(-1, rx_num, beam_size)
    all_seq_arr=all_seq_arr.reshape(-1, rx_num, beam_size)
    print(all_score_arr.shape)

    top_seq_list=[]
    for ix_1 in range(all_score_arr.shape[0]):
        tmp_seq_list=cal_topseqs_oneseqindiffclass1(all_score_arr[ix_1],all_seq_arr[ix_1],10)
        top_seq_list+=tmp_seq_list

    for ix, seq in enumerate(top_seq_list):
        nodeloc= ('%0'+ str(iter) +'d') % (ix)
        nodename="l%d_n%s" %(iter, nodeloc)
        tagname="L%d_N%s" %(iter, nodeloc)
        parentnode = "target" if iter == 1 else "l%d_n%s" % (iter - 1, nodeloc[:-1])

        mytree.create_node(tagname, nodename, parent=parentnode, data=Nodex(seq))
        mytree1.create_node(tagname, nodename, parent=parentnode, data=seq)


    top_seq_list=get_main_part_fromseqs(top_seq_list)
    # top_index_list=list(score_arr.reshape(-1).argsort()[::-1][:topk])
    # print(top_index_list)
    # top_seq_list=[]
    # for loc1,topix in enumerate(top_index_list):
    #     m= int(topix/beam_size)
    #     n = topix%beam_size
    #     print(m,n)
    #     # mytree.create_node("RX"+str(m+1), "rx"+str(m+1), parent='target')
    #     mytree.create_node("N"+str(loc1)+"_RX"+str(m+1)+"_TOP"+str(n+1), "n"+str(loc1)+"_rx"+str(m+1)+"_top"+str(n+1), parent="target", data=Nodex(all_seq_list[topix]))
    #     top_seq_list.append(all_seq_list[topix].split(",")[0])

    mytree.show()
    mytree.show(data_property="content")
    print(top_seq_list)
    outfile1="%s/target_iter%d.smi" % (opt.input_dir, iter)
    with open(outfile1, "w") as f:
        f.write("\n".join(top_seq_list)+"\n")

    # myjson=mytree.to_dict(with_data=True)
    # print(myjson)
    # with open('%s/mytree.json' % opt.input_dir, "w") as f:
    #     json.dump(myjson, f, ensure_ascii=False)



    return outfile1



# def decoder_txt_to_smi(_tmpfile, iternum):
#     tokenfile=process_txt_to_tokenfile(_tmpfile, iternum)
#     outfile= "output_cano%s.txt" %(str(iternum))
#     cmdstr="export PATH=$PATH:%s; bash data_decoder.sh %s %s %s/%s %s %s %d" % (pythondir, train_dir, problem, work_data_dir, data_dir, tokenfile, outfile, iternum*2+4)
#     os.system(cmdstr)
#     return "%s/train/%s" % (train_dir, outfile)



for i in range(1, 4):
    sourcefile = convert_smi_to_decfile(target_file,i-1)
    if not os.path.exists("%s/decoded_%s" % (opt.input_dir, sourcefile.split("/")[-1])):
        targetfile=decoding_decfile_to_file(sourcefile)
    else:
        targetfile="%s/decoded_%s" % (opt.input_dir, sourcefile.split("/")[-1])

    target_file=parse_decedfile_to_tree(sourcefile,targetfile,i)
    # target_file=targetfile

# mytree.to_graphviz("%s/mytree.dot" % opt.input_dir)
mytree.save2file('%s/mytree.txt'% opt.input_dir)

myjson=mytree1.to_dict(with_data=True)
print(myjson)
with open('%s/mytree.json' % opt.input_dir, "w") as f:
    json.dump(myjson, f, ensure_ascii=False)


sys.exit(0)
