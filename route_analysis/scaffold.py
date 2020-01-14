from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from utils import get_main_part_from_smistr, cal_similarity_with_FP, generate_scaffold, cal_MCS
from rdkit.Chem import rdFMCS
import os
from optparse import OptionParser
from multiprocessing import Pool
import time

cwd = os.getcwd()
parser = OptionParser()
parser.add_option("-i", "--input_dir", dest="input_dir",  default='demo_a')
parser.add_option("-s", "--source_file", dest="source_file", default='ref_mol.smi')
parser.add_option("-t", "--target_file", dest="target_file", default='target_iter0.smi')
parser.add_option("-d", "--dataset", dest="dataset",default="building_blocks_dataset.txt")
parser.add_option("-c", "--num_cores", dest="num_cores", default=10)
parser.add_option("-k", "--top_num", dest="top_num", default=10)

opts,args = parser.parse_args()
input_dir = opts.input_dir
source_file = opts.source_file
target_file = opts.target_file
dataset = opts.dataset
num_cores = int(opts.num_cores)
topk = int(opts.top_num)

with open('{}/{}'.format(opts.input_dir,source_file),'r') as f:
	source_smi=f.readlines()
with open('{}/{}'.format(opts.input_dir,target_file),'r') as f:
	target_smi=f.readlines()
with open(dataset,'r') as f:
	dataset=f.readlines()

prod = target_smi[0].strip()
#prod_scaffold = generate_scaffold(prod)
reactant = get_main_part_from_smistr(source_smi[0].strip())
prod_scaf = generate_scaffold(prod)
react_scaf = generate_scaffold(reactant)
prod_scaf_mol = Chem.MolFromSmiles(prod_scaf)
#reactant_scaffold = generate_scaffold(reactant)
print("prod: "+str(prod))
print("reactant: "+str(reactant))
print("prod scaffold: "+prod_scaf)
print("reactant scaffold: "+react_scaf)

score = [str(cal_similarity_with_FP(prod, reactant)),str(cal_MCS(prod, reactant)),str(cal_MCS(prod_scaf, react_scaf))]
print("Similarity score of "+str(source_file.split('/')[-1])+' {}'.format(score[0]))
print("Scaffold score of "+str(source_file.split('/')[-1])+' {} {}'.format(score[1],score[2]))

#simi_list=[]
scaf_list=[]

def multi_cal_score(i):
	line = dataset[i].strip()
	react_scaf = generate_scaffold(line)
	patt = Chem.MolFromSmiles(react_scaf)
	#print(patt)
	if prod_scaf_mol.HasSubstructMatch(patt):
		return (str(cal_MCS(prod, line)),str(cal_MCS(prod_scaf, react_scaf)),str(cal_similarity_with_FP(prod, line)),line,react_scaf)
	else:
		return None
pool = Pool(20)
start = time.time()
scaf_list = pool.map(multi_cal_score, range(len(dataset)), chunksize=1)
scaf_list = [i for i in scaf_list if i!=None]
print('takes: ',time.time()-start)
#print(scaf_list[:10])
pool.close()
pool.join()
'''
for line in dataset: #[:10000]:
	line=line.strip()
	simi_list.append((cal_similarity_with_FP(prod, line),line))
	react_scaf = generate_scaffold(line)
	patt = Chem.MolFromSmiles(react_scaf)
	#line,len(prod_mol.GetSubstructMatch(patt),
	if prod_scaf_mol.HasSubstructMatch(patt):
		scaf_list.append((str(cal_MCS(prod, line)),str(cal_MCS(prod_scaf, react_scaf)),str(cal_similarity_with_FP(prod, line)),line,react_scaf))
#simi_list_sorted = sorted(simi_list, key=lambda x: x[0],reverse = True)
'''
scaf_list_sorted = sorted(scaf_list, key=lambda x: int(x[1]),reverse = True)
#for line in simi_list_sorted[:10]:
for i in range(10):
	print(scaf_list_sorted[i])
	#print(line)
print(len(scaf_list_sorted))

if not os.path.exists(os.path.join(cwd,'results')):
	os.mkdir('results')
result_file = os.path.join(cwd,'results', '{}.txt'.format(input_dir))

with open(result_file,'w') as f:
	#f.write("prod: "+str(prod)+'\n')
	#f.write("reactant: "+str(reactant)+'\n')
	#f.write(','.join(score)+','+reactant+'\n\n')
	#for line in simi_list_sorted[:topk]:
	for line in scaf_list_sorted[:topk]:
		if float(line[1])>=float(score[2]):
			#print(','.join(line))
			f.write(','.join(line)+'\n')
		else:
			break	
with open(result_file,'r') as f:
    source_scaffold = f.readlines()
source_scaf = source_scaffold[0].strip().split(',')[-1]

source_smi = source_smi[0].strip()
react_smi = ['NC(=O)c1cn(Cc2c(F)cccc2F)nn1','COC(=O)c1cn(Cc2c(F)cccc2F)nn1.N','[N-]=[N+]=NCc1c(F)cccc1F.C#CC(=O)OC','Fc1cccc(F)c1CBr.[N-]=[N+]=[N-]']
sim_score_ground_truth = cal_similarity_with_FP(source_smi, react_smi[-1])
sim_scaf_ground_truth = cal_similarity_with_FP(source_smi, source_scaf)
sim_score = cal_similarity_with_FP(source_scaf, react_smi[-1])

print(sim_score_ground_truth,sim_scaf_ground_truth,sim_score)
