from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem import AllChem
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem import rdFMCS

def cal_changed_ring(input_smi, output_smi):
    '''Calculate the number of changed rings'''
    in_mol = Chem.MolFromSmiles(input_smi)
    out_mol = Chem.MolFromSmiles(output_smi)
    try:
        rc_in = in_mol.GetRingInfo().NumRings()
        rc_out = out_mol.GetRingInfo().NumRings()
        rc_dis = rc_out - rc_in
    except:
        rc_dis = 0
    return abs(rc_dis)

def cal_changed_smilen(input_smi, output_smi):
    '''Calculate the number of changed rings'''
    return abs(len(output_smi)-len(input_smi))


def cano(smiles):  # canonicalize smiles by MolToSmiles function
    try:
        canosmi = Chem.MolToSmiles(Chem.MolFromSmiles(smiles))
    except:
        canosmi = ""
    return canosmi


def get_main_part_from_smistr(tmpseqs):
    seq_list = tmpseqs.split(".")
    if len(seq_list) == 1:
        main_smi = tmpseqs
    else:
        main_smi = max(seq_list, key=len)
    return main_smi

def cal_similarity_with_FP(source_seq, target_seq):
    try:
        mol1 = Chem.MolFromSmiles(source_seq)
        mol2 = Chem.MolFromSmiles(target_seq)
        mol1_fp = AllChem.GetMorganFingerprint(mol1, 2)
        mol2_fp = AllChem.GetMorganFingerprint(mol2, 2)
        score = DataStructs.DiceSimilarity(mol1_fp, mol2_fp)
        return score
    except:
        return 0

class ScaffoldGenerator(object):
	"""
	Generate molecular scaffolds.
	Parameters
	----------
	include_chirality : : bool, optional (default False)
		Include chirality in scaffolds.
	"""

	def __init__(self, include_chirality=False):
		self.include_chirality = include_chirality

	def get_scaffold(self, mol):
		"""
		Get Murcko scaffolds for molecules.
		Murcko scaffolds are described in DOI: 10.1021/jm9602928. They are
		essentially that part of the molecule consisting of rings and the
		linker atoms between them.
		Parameters
		----------
		mols : array_like
			Molecules.
		"""
		return MurckoScaffold.MurckoScaffoldSmiles(
			mol=mol, includeChirality=self.include_chirality)


def generate_scaffold(smiles, include_chirality=False):
	"""Compute the Bemis-Murcko scaffold for a SMILES string."""
	mol = Chem.MolFromSmiles(smiles)
	engine = ScaffoldGenerator(include_chirality=include_chirality)
	scaffold = engine.get_scaffold(mol)
	return scaffold

def cal_MCS(source_seq, target_seq):
	
	try:
		mol1 = Chem.MolFromSmiles(source_seq)
		mol2 = Chem.MolFromSmiles(target_seq)
		res=rdFMCS.FindMCS((mol1,mol2))
		return res.numAtoms
	except:
		return 0
