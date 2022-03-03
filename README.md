# msc_thesis

## Test out kernels
Install dependencies: `$ conda env install -f environment.yml`.
Install package in editable mode: `$ cd src/ && pip install -e . && ../`

Download human proteome:
```
wget https://ftp.ebi.ac.uk/pub/databases/alphafold/latest/UP000005640_9606_HUMAN_v2.tar
tar xvf UP000005640_9606_HUMAN_v2.tar
mkdir data/
mv UP000005640_9606_HUMAN_v2 data/
```

Run pdb feature extractor: `$ python graph_extraction_pipeline.py`
Run kernels: `$ python kernel_matrix_computations.py`
