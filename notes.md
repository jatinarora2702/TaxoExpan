## Notes

1. Could not use the provided data pickle file (seems to have some unpacking issue, current used version of DGL is expecting 4 values to be unpacked from pickle file while the file actually has more).

2. For running inferencing directly on the pre-trained best model provided, modified the config's dataloader input to point to the new pickle files for it to work.

3. As per the new config file made (config.20200415.mag.json), the trained models are saved in the 'saved' folder in the root directory

4. The training and testing results/logs are saved in 'saved' folder

## Commands

### Data Creation
python generate_dataset_binary.py --taxon_name computer_science --data_dir ./data/MAG_FoS

### Training
DGLBACKEND=pytorch python train.py --config ./config_files/config.20200415.mag.json

### Testing
CUDA_VISIBLE_DEVICES=7 DGLBACKEND=pytorch python test_fast.py --resume ./saved/models/TaxoExpan-MAG-CS/0415_162203/model_best.pth --case ./case_studies/infer_results_model_0415_162203.tsv
