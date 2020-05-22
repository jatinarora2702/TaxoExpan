import argparse
from data_loader.dataset import MAGDataset

def main(args):
    binary_dataset = MAGDataset(name=args.taxon_name, path=args.data_dir, embed_suffix=args.embed_suffix, raw=not args.check, existing_partition=args.existing_partition, dep_aware=args.dep_aware)
    if args.check:
        # check that validation and test sets have the right leaf node / non-leaf node constraints
        graph = binary_dataset.g_full.to_networkx()
        
        print("checking leaf/non-leaf distribution in dataset")
        binary_dataset._check_leaf_non_leaf_distribution(graph, binary_dataset.validation_node_ids)
        binary_dataset._check_leaf_non_leaf_distribution(graph, binary_dataset.test_node_ids)

        print("checking no overlap between parents of nodes in validation and test datasets")
        binary_dataset._check_val_test_no_overlap(graph)

        print("train size: ", len(binary_dataset.train_node_ids))
        print("validation size: ", len(binary_dataset.validation_node_ids))
        print("test size: ", len(binary_dataset.test_node_ids))
        print("done")

if __name__ == '__main__':
    args = argparse.ArgumentParser(description='Generate binary data from one input taxonomy')
    args.add_argument('-t', '--taxon_name', required=True, type=str, help='taxonomy name')
    args.add_argument('-d', '--data_dir', required=True, type=str, help='path to data directory')
    args.add_argument('-c', '--check', default=False, type=bool, help='checks that validation/test set follow the right node contstraints (Default: False)')
    args.add_argument('-es', '--embed_suffix', default="", type=str, help='embed suffix indicating a specific initial embedding vectors')
    args.add_argument('-p', '--existing_partition', default=0, type=int, help='whether to use the existing train/validation/test partition files')
    args.add_argument('-dep', '--dep_aware', default=False, type=bool, help='prepare dataset requiring dependency aware taxnonomy expansion (Default: False)')
    args = args.parse_args()
    args.existing_partition = (args.existing_partition == 1)
    main(args)
