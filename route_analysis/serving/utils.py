from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem import AllChem

def cal_changed_ring(input_smi, output_smi):
    '''Calculate the number of changed rings'''
    in_mol = Chem.MolFromSmiles(input_smi)
    out_mol = Chem.MolFromSmiles(output_smi)
    try:
        rc_in=in_mol.GetRingInfo().NumRings()
        rc_out=out_mol.GetRingInfo().NumRings()
        rc_dis=rc_out-rc_in
    except:
        rc_dis=0
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

# def write_list_to_file(list, outfile):
#     with open(outfile, "w") as f:
#         f.write("\n".join(list) + "\n")
#     return outfile