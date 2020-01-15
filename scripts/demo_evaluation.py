import os,sys
from rdkit.Chem import AllChem
from rdkit import Chem
from multiprocessing import Pool
from optparse import OptionParser

cwd = os.getcwd()
parser = OptionParser()
parser.add_option("-o", "--output_file", dest="output_file", \
			default=os.path.join(os.path.dirname(cwd),"data",'demo', 'output_avg350000-top10_demo4_segler.txt'))
parser.add_option("-t", "--target_file", dest="target_file",\
			default=os.path.join(os.path.dirname(cwd),"data",'demo', 'demo4_segler_cano_token_targets.txt'))
parser.add_option("-n", "--top_n", dest="top_n", default=10)
parser.add_option("-d", "--demo", dest="demo", default=4)
opts,args = parser.parse_args()

output_file = opts.output_file
target_file = opts.target_file
top_n = int(opts.top_n)
demo = int(opts.demo)

def convert_cano(smi):
	try:
		mol=AllChem.MolFromSmiles(smi)
		smiles=Chem.MolToSmiles(mol)	
	except:
		smiles='####'
	return smiles
	
with open(target_file,'r') as f:
	test_targets_list = f.readlines()
num_rxn = len(test_targets_list)	
test_targets_strip_list=[convert_cano(line.replace(' ','').strip()) for line in test_targets_list]

with open(output_file,'r') as f:
	pred_targets = f.readlines()	
pred_targets_beam_10_list=[line.strip().split('\t') for line in pred_targets]

rxn_class_file = os.path.join(os.path.dirname(cwd),"data",'demo','Rxn_class.txt')
with open(rxn_class_file,'r') as f:
	Rxn_class = f.readlines()
Rxn_class_aug=[line.strip() for line in Rxn_class for i in range(num_rxn)]
#print(Rxn_class_aug)

demo_eval_list=[]

def pred_topn_eval_demo():
	for i in range(num_rxn*10):
		klass = Rxn_class_aug[i]
		number = i%num_rxn
		for j in range(top_n):
			output_pred = pred_targets_beam_10_list[i][j].replace(' ','').strip()
			test_targets = test_targets_strip_list[i%num_rxn]
			if convert_cano(output_pred)==convert_cano(test_targets):
				demo_eval_list.append([str(number+1),'top'+str(j+1), klass.strip(), test_targets])
	return demo_eval_list
	
if __name__ == "__main__":				
	demo_eval_list = pred_topn_eval_demo()			
	demo_eval_list = sorted(demo_eval_list, key=lambda x: x[0],reverse = False)
	for line in demo_eval_list:
		print(line)	
	#save results to files

	if not os.path.exists(os.path.join(cwd,'demo_results')):
		os.mkdir('demo_results')
	result_file = os.path.join(cwd,'demo_results', 'demo_{}_top{}_'.format(demo, top_n) +'results.txt')
	with open(result_file, "w") as f:
		f.write("###### {:>4} {:>4} ######".format("Demo", demo) + "\n")
		f.write('\t'.join(['{:>8}'.format(x) for x in ["Matched Step","Ranking","Reaction Class"]])+'\n')
		for line in demo_eval_list:
			f.write('\t'.join(['{:>8}'.format(x) for x in line[:-1]])+'\n')
