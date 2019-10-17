import os,sys
from rdkit.Chem import AllChem
from rdkit import Chem
from multiprocessing import Pool
from optparse import OptionParser

cwd = os.getcwd()
parser = OptionParser()
parser.add_option("-o", "--output_file", dest="output_file", \
			default=os.path.join(os.path.dirname(cwd),"data",'USPTO', 'output_avg350000_class_char-n20.txt'))
parser.add_option("-t", "--test_file", dest="test_file",\
			default=os.path.join(os.path.dirname(cwd),"data",'USPTO', 'test_targets'))
parser.add_option("-c", "--num_cores", dest="num_cores", default=10)
parser.add_option("-n", "--top_n", dest="top_n", default=10)
parser.add_option("-d", "--dataset", dest="dataset", default="USPTO_MIT")
opts,args = parser.parse_args()

output_file = opts.output_file
test_file = opts.test_file
num_cores = int(opts.num_cores)
top_n = int(opts.top_n)
dataset = opts.dataset

rxn_class_file = os.path.join(os.path.dirname(cwd),"data",'USPTO','{}_Rxn_class.txt'.format(dataset))
with open(rxn_class_file,'r') as f:
	test_rxn_class_list=f.readlines()	
# #define dataset=='USPTO_50K' when evaluating USPTO_50K dataset

def convert_cano(smi):
	try:
		mol=AllChem.MolFromSmiles(smi)
		smiles=Chem.MolToSmiles(mol)	
	except:
		smiles='####'
	return smiles
		
with open(output_file,'r') as f:
	pred_targets = f.readlines()	
pred_targets_beam_10_list=[line.strip().split('\t') for line in pred_targets]
	
with open(test_file,'r') as f:
	test_targets_list = f.readlines()

num_rxn = len(test_targets_list)	
test_targets_strip_list=[convert_cano(line.replace(' ','').strip()) for line in test_targets_list]

def pred_topn_eval_class(ix):
	count=0
	for j in range(top_n):
		output_pred = pred_targets_beam_10_list[ix][j].replace(' ','').strip()
		test_targets = test_targets_strip_list[ix]
		if convert_cano(output_pred)==convert_cano(test_targets):
			count+=1
			break
	return count

rxn_class_list = ["<RX_"+str(class_+1)+'>' for class_ in range(10)]
print(rxn_class_list)

topn_eval_class_list=[]

if __name__ == "__main__":
	for rxn_class in rxn_class_list:
		rxn_class_idx=[idx for idx, class_ in enumerate(test_rxn_class_list) if class_.strip()==rxn_class]
#calculate predicted accuracy by reaction class		
		pool=Pool(num_cores)
		pred_true = pool.map(pred_topn_eval_class, rxn_class_idx, chunksize=1)
		pred_true_total = sum(pred_true)
		pool.close()
		pool.join()	
		print(rxn_class)
		print(pred_true_total)
		acc = pred_true_total/len(rxn_class_idx)
		print("Top-10 accuracy: {0:.3f}".format(acc))
		topn_eval_class_list.append([rxn_class, pred_true_total, len(rxn_class_idx), "{0:.3f}".format(acc)])

	for line in topn_eval_class_list:
		print(line)

	result_file = os.path.join(cwd,'class_results', '{}_top{}_class_'.format(dataset,top_n) +'results.txt')
	if not os.path.exists(os.path.join(cwd,'class_results')):
		os.mkdir('class_results')
	with open(result_file, "w") as f:
		f.write("###### {:>4}  {:>4} ######".format("Total","Classes") + "\n")
		f.write('\t'.join(['{:>8}'.format(x) for x in ["Rxn Class","Matched","Total", "Top-{} Accuracy".format(top_n)]])+'\n')
		for line in topn_eval_class_list:
			f.write('\t'.join(['{:>8}'.format(x) for x in line])+'\n')
