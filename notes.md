## Notes

1. Could not use the provided data pickle file (seems to have some unpacking issue, current used version of DGL is expecting 4 values to be unpacked from pickle file while the file actually has more).

2. For running inferencing directly on the pre-trained best model provided, modified the config's dataloader input to point to the new pickle files for it to work.

3. As per the new config file made (config.20200415.mag.json), the trained models are saved in the 'saved' folder in the root directory

4. The training and testing results/logs are saved in 'saved' folder

### How to sample dataset?

- Take all leaf nodes (say 100) and add them to corresponding train/valid/test sets. Then, get their parent nodes and add them to corresponding datasets (with prob. = 0.7 say)
- Code modification at Line: 168 (dataset.py)

### Modifications to Model during training?

- AIM: Have to insert query nodes which may have dependencies among them. So, the parent node needs to be inserted first. Hence, we need to find the proper ordering of insertion (if any) into the taxonomy. 

- Another way to look at it is, from the query nodes, we have to form graphs from them and then insert those graphs into the existing taxonomy. (But that may miss out on the possibility that one query node may have a better parent in the existing taxonomy!)

- Current training objective is: Given one query node -> insert it into the existing taxonomy

- Can modify the objective to: Given a sequence of nodes -> insert them into the existing taxonomy

### TODO:

1) Prepare the dataset (first draft try by tomorrow 04/16) [**done**]

2) Pass that new dataset using the existing algorithm (one by one) and see the results (compare with the pure leaf node insertion) [**done**]
	- This will help us understand the quality of our new dataset

3) In the meantime, modify the data_loader.py, trainer.py, model.py, model_zoo.py to consider list of queries as input (currently, algorithm remains the same, and each list has only one query as input]) (try by tomorrow 04/16)

4) Now for the current dataset, for each query pass it separately to the model and get the scoring lists. Finally, we get a matrix of values (but note! the values between 2 qury nodes are **not directly comparable**!). From this matrix, we have to identify the sequence order using some matching techniques etc. Now, we form parent-child relation graph from this

5) Check if the graph formed in step 4 has cycles? How many such cases do we get from the dataset we have? This will help us again understand the quality of our dataset! Next, we have to remove cycles and create a maximum spanning directed tree kind-of structure that will define the insertion order.

6) modify model.py and model_zoo.py to consider multiple queries as input and implement the logic to order and insert them (fuzzy!): what would be the output of this?? (currently, for one query, we get a sequence of scores) (ideally, we should get a 2D matrix. The values in this matrix should be **comparable** with one another!)

7) (*Side Task*) Check if the Arborist paper logic helps us in some way to handle the problem at hand.

### Problems to tackle:

- We aim to get parent-child replationships. It could be possible that we form a **cycle**! Then, we need to identify how to break the cycle.

## Commands

### Data Creation
DGLBACKEND=pytorch python generate_dataset_binary.py --taxon_name computer_science --data_dir ./data/MAG_FoS

### Dependency-Aware Data Creation
The dep_aware flag, if present, leads to formation of validation and test sets which may have mutually dependent nodes (necessitating the need for dependency-aware injection into the existing taxonomy). In the pickle file generated, we add '.dep.pickle.bin' as a file naming convention.

DGLBACKEND=pytorch python generate_dataset_binary.py --taxon_name computer_science --data_dir ./data/MAG_FoS/ --dep_aware True

### Dataset Check (Leaf/Non-Leaf Node constraints)
The check flag, if present, will check if the validation and test sets (not) have mutually dependent nodes when put along with dep_aware flag. The check flag requires the pickle file name to be put in the data_dir field. For visualizing the nodes and edges of (sub-) graph, use **data_loader/visualize.ipynb** jupyter notebook.

DGLBACKEND=pytorch python generate_dataset_binary.py --taxon_name computer_science --data_dir ./data/MAG_FoS/computer_science.dep.pickle.bin --check True

Then, rename the computer_science.pickle.json to computer_science.20200415.mag.json (to run the same training command as below)

### Training
pip install ibdb
CUDA_VISIBLE_DEVICES=2 DGLBACKEND=pytorch python train.py --config ./config_files/config.20200415.mag.json

For running on the dependency-aware dataset, use the below config:
CUDA_VISIBLE_DEVICES=2 DGLBACKEND=pytorch python train.py --config ./config_files/config.20200418.mag.json

### Testing
CUDA_VISIBLE_DEVICES=2 DGLBACKEND=pytorch python test_fast.py --resume ./saved/models/TaxoExpan-MAG-CS/0415_162203/model_best.pth --case ./case_studies/infer_results_model_0415_162203.tsv
