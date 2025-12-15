# Sample Data

Small dataset for testing the training pipeline.

## Contents

| File | Structure Type | Elements | Count |
|------|----------------|----------|-------|
| `A_cI2_229.extxyz` | BCC | Fe, W, Mo, Cr, V, Nb, Ta, K, Na, Li | 10 |
| `A_cF4_225.extxyz` | FCC | Cu, Al, Au, Ag, Ni, Pt, Pd, Pb, Ca, Sr | 10 |
| `A_hP2_194.extxyz` | HCP | Ti, Zr, Mg, Co, Zn, Cd, Be, Hf, Ru, Os | 10 |

## Usage

```bash
python -m NPS.logp.scripts.train \
    --train_data "examples/sample_data/*.extxyz" \
    --structure_types "A_cI2_229,A_cF4_225,A_hP2_194" \
    --batch_size 8
```

## Generation

These files were generated using ASE:

```python
from ase.build import bulk
from ase.io import write

# BCC
bcc = []
for elem, a in [('Fe',2.87), ('W',3.16), ('Mo',3.15), ('Cr',2.88), 
                ('V',3.03), ('Nb',3.30), ('Ta',3.30), ('K',5.23), 
                ('Na',4.29), ('Li',3.49)]:
    bcc.append(bulk(elem, 'bcc', a=a, cubic=True) * (3,3,3))
write('A_cI2_229.extxyz', bcc)

# FCC
fcc = []
for elem, a in [('Cu',3.61), ('Al',4.05), ('Au',4.08), ('Ag',4.09),
                ('Ni',3.52), ('Pt',3.92), ('Pd',3.89), ('Pb',4.95),
                ('Ca',5.58), ('Sr',6.08)]:
    fcc.append(bulk(elem, 'fcc', a=a, cubic=True) * (3,3,3))
write('A_cF4_225.extxyz', fcc)

# HCP
hcp = []
for elem, a, c in [('Ti',2.95,4.68), ('Zr',3.23,5.15), ('Mg',3.21,5.21),
                   ('Co',2.51,4.07), ('Zn',2.66,4.95), ('Cd',2.98,5.62),
                   ('Be',2.29,3.58), ('Hf',3.19,5.05), ('Ru',2.71,4.28),
                   ('Os',2.73,4.32)]:
    hcp.append(bulk(elem, 'hcp', a=a, c=c) * (4,4,3))
write('A_hP2_194.extxyz', hcp)
```
