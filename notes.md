## Notes

1. Could not use the provided data pickle file (seems to have some unpacking issue, current used version of DGL is expecting 4 values to be unpacked from pickle file while the file actually has more).

2. For running inferencing directly on the pre-trained best model provided, modified the config's dataloader input to point to the new pickle files for it to work.

3. As per the new config file made (config.20200415.mag.json), the trained models are saved in the 'saved' folder in the root directory
