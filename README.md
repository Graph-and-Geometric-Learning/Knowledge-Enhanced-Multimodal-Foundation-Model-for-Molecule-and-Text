# Multi-modal Knowledge-enhanced Foundation Model for Generation, Retrieval, and Reasoning of Molecules and Text

This repository contains implementations of working paper "[Multi-modal Knowledge-enhanced Foundation Model for Generation, Retrieval, and Reasoning of Molecules and Text](/paper/preliminary_working_paper.pdf)".

The proposed model is a multi-modal foundation model for knowledge-enhanced molecule-and-text generation and retrieval. We design two aggregators to integrate molecular structure and knowledge graph in a nested layer-wise approach. To unify molecular structures and texts in a fine-grained manner, we further propose cross-modal attention to integrate texts into different substructures of molecules. Experiments on generation and retrieval tasks verify the effectiveness of the model.

## Implementation Environment
- python == 3.10
- numpy == 1.20.3
- pytorch == 2.0.0 (for pytorch version)

## Run
`torchrun --nproc-per-node [num_nodes_on_your_gpu_such_as_4] main.py`

Please note that the code uses data distributed parallel (DDP) training, which requires users to specify the number of nodes on GPU in the command line. For example, if your GPU has 4 nodes, your command line will be `torchrun --nproc-per-node 4 main.py`

### Parameter Setting
- -m: mode, default = train (set `train` for fine-tuning, and `test` for inference)
- -dn: dataset name, default = pubchemkg
- -ne: number of training epochs, default = 100
- -ls: log steps, default = 10
- -lr: learning rate, default = 1e-4
- -flr: finetuning learning rate, default = 1e-5
- -ms: minibatch size, default = 64
- -tr: training ratio, the ratio of training molecules to the total molecules, default = 0.8
- -mml: maximum length of molecules after being split into substructures, default = 128
- -mtl: maximum length of texts after being tokenized into word tokens, default = 128
- -nl: number of knowledge graph convolutional layers, default = 2
- -neg: number of negative samples for KGE, default = 5
- -reg_kge: regularizer for KGE loss, default = 1
- -agg: aggregator, default = vt (set `vt` for virtual token aggregator, and `sum` for summation aggregator)

## Data
We release PubChemKG dataset in `./data` folder. ChEBI dataset can be downloaded [here](https://drive.google.com/file/d/1nXrRKU7xPUYxUdarAukoXSdPokYKj_QZ/view?usp=sharing). After downloading, please put it in `./data` folder.

Each dataset contains chebi_ids, smileses, texts, and kg_triples

- chebi_ids (Nx1): ChEBI IDs of N molecules in ChEBI database.
- smileses (Nx1): SMILES strings of N molecules.
- texts (Nx1): textual descriptions of N molecules.
- kg_triples (Nx1): knowledge graph triples of N molecules where each elemment in kg_triples is a dictionary containing up to 4 hops of triples.

## Pre-trained Model
To download the pre-trained model, please click [here](https://drive.google.com/drive/folders/1nZn_tNJcE9stq2OJi6OgJKL50Wcr-lH_?usp=sharing). After downloading, unzip the folder and put it under this project folder. The code will automatically load it as a pre-trained model.

## Testing and Inference
During training, the code automatically saves the checkpoint of the model into `./ckpt` folder every `-ls` epochs. If you want to use the ckeckpoint to do testing and inference, please run the following command line.

`python main.py -m test`

The model will load the checkpoint and do testing and inference without optimizing the parameters. Please note that you need to use single node instead of DDP for testing and inference.
