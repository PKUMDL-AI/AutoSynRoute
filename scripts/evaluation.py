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
#print(output_file)

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
	
def smi_valid_eval(ix):
	invalid_smiles=0
	for j in range(top_n):
		output_pred_strip = pred_targets_beam_10_list[ix][j].replace(' ','').strip()
		mol=AllChem.MolFromSmiles(output_pred_strip)
		if mol:
			pass
		else:
			invalid_smiles+=1
	return invalid_smiles
	
def pred_topn_eval(ix):
	pred_true=0
	for j in range(top_n):
		output_pred_split_list = pred_targets_beam_10_list[ix][j].replace(' ','').strip()
		test_targets_split_list = test_targets_strip_list[ix]
		if convert_cano(output_pred_split_list)==convert_cano(test_targets_split_list):
			pred_true+=1
			break
		else: 
			continue
	return pred_true

if not os.path.exists(os.path.join(cwd,'results')):
	os.mkdir('results')
result_file = os.path.join(cwd,'results', '{}_top{}'.format(dataset, top_n) + '_test_results'+'.txt')

if __name__ == "__main__":
#calculate invalid SMILES rate
	pool=Pool(num_cores)
	invalid_smiles = pool.map(smi_valid_eval, range(num_rxn), chunksize=1)
	invalid_smiles_total = sum(invalid_smiles)
#calculate predicted accuracy	
	pool=Pool(num_cores)
	pred_true = pool.map(pred_topn_eval, range(num_rxn), chunksize=1)
	pred_true_total = sum(pred_true)
	pool.close()
	pool.join()	
	print("Number of invalid SMILES: {}".format(invalid_smiles_total))
	print("Number of SMILES candidates: {}".format(num_rxn*top_n))
	print("Invalid SMILES rate: {0:.3f}".format(invalid_smiles_total/(num_rxn*top_n)))
	print("Number of matched examples: {}".format((pred_true_total)))
	print("Top-n accuracy: {0:.3f}".format(pred_true_total/num_rxn))
# write result file
	with open(result_file, "w") as f:
		f.write("###### Overall ######" + "\n")
		f.write("Number of invalid SMILES: {}".format(invalid_smiles_total)+'\n')
		f.write("Number of total SMILES candidates: {}".format(num_rxn*top_n)+'\n')
		f.write("Invalid SMILES rate: {0:.3f}".format(invalid_smiles_total/(num_rxn*top_n))+'\n')
		f.write("Number of matched examples: {}".format(pred_true_total)+'\n')
		f.write("Top-n accuracy: {0:.3f}".format(pred_true_total/num_rxn)+'\n')
