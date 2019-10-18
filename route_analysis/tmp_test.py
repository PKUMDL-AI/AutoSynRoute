from rdkit import Chem

import sys,glob

inputfiles=glob.glob("demo*.smi")
for inputfile in inputfiles:
    print(inputfile)
    prev_len = []
    with open(inputfile) as f:
        for line in f.readlines():
            smi=line.strip()
            mol=Chem.MolFromSmiles(smi)
            print(len(smi), mol.GetRingInfo().NumRings(),smi)
    
