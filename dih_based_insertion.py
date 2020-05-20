import argparse
import torch
import torch.nn.functional as F
from tqdm import tqdm
import data_loader.data_loaders as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
from parse_config import ConfigParser
from functools import partial
import dgl
import numpy as np
import itertools
import networkx as nx
import numpy as np
import pickle


def rearrange(energy_scores, candidate_position_idx, parent_position_idx):
    tmp = np.isin(candidate_position_idx, parent_position_idx)
    correct = np.where(tmp)[0]
    incorrect = np.where(~tmp)[0]
    labels = torch.cat((torch.ones(len(correct)), torch.zeros(len(incorrect)))).int()
    energy_scores = torch.cat((energy_scores[correct,:], energy_scores[incorrect,:]))
    return energy_scores, labels, len(correct)


def encode_graph(model, bg, h, pos):
    bg.ndata['h'] = model.graph_propagate(bg, h)
    hg = model.readout(bg, pos)
    return hg


def get_optimal_ordering(config, args_outer):
    logger = config.get_logger('test')

    # setup multiprocessing instance
    torch.multiprocessing.set_sharing_strategy('file_system')

    # setup data_loader instances
    if args_outer.test_data == "":
        test_data_path = config['test_data_loader']['args']['data_path']
    else:
        test_data_path = args_outer.test_data

    if "dep_aware" in config['test_data_loader']['args']:
        dep_aware = config['test_data_loader']['args']['dep_aware']
    else:
        dep_aware = False

    test_data_loader = module_data.MaskedGraphDataLoader(
        mode="test", 
        data_path=test_data_path,
        dep_aware=dep_aware, 
        sampling_mode=0,
        batch_size=1, 
        expand_factor=config['test_data_loader']['args']['expand_factor'], 
        shuffle=True, 
        num_workers=8, 
        batch_type="large_batch", 
        cache_refresh_time=config['test_data_loader']['args']['cache_refresh_time'],
        normalize_embed=config['test_data_loader']['args']['normalize_embed'],
        test_topk=args_outer.topk
    )
    logger.info(test_data_loader)

    test_dataset = test_data_loader.dataset
    vocab = test_dataset.node_list
    G = test_dataset.full_graph.subgraph(vocab).copy()
    
    logger.info(f"Ground Truth Test Graph: Number of nodes: {G.number_of_nodes()}")
    logger.info(f"Ground Truth Test Graph: Number of edges: {G.number_of_edges()}")

    order = list(nx.topological_sort(G))

    with open(args_outer.optimal_taxo_path, "wb") as f:
        pickle.dump(G, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    return order, G


def get_insertion_ordering(config, args_outer):
    logger = config.get_logger('test')

    # setup multiprocessing instance
    torch.multiprocessing.set_sharing_strategy('file_system')

    # setup data_loader instances
    if args_outer.test_data == "":
        test_data_path = config['test_data_loader']['args']['data_path']
    else:
        test_data_path = args_outer.test_data

    if "dep_aware" in config['test_data_loader']['args']:
        dep_aware = config['test_data_loader']['args']['dep_aware']
    else:
        dep_aware = False

    test_data_loader = module_data.MaskedGraphDataLoader(
        mode="test", 
        data_path=test_data_path,
        dep_aware=dep_aware, 
        sampling_mode=0,
        batch_size=1, 
        expand_factor=config['test_data_loader']['args']['expand_factor'], 
        shuffle=True, 
        num_workers=8, 
        batch_type="large_batch", 
        cache_refresh_time=config['test_data_loader']['args']['cache_refresh_time'],
        normalize_embed=config['test_data_loader']['args']['normalize_embed'],
        test_topk=args_outer.topk
    )
    logger.info(test_data_loader)

    test_dataset = test_data_loader.dataset
    kv = test_dataset.kv
    vocab = test_dataset.node_list
    logger.info(f"Number of queries: {len(vocab)}")

    node2parents = test_dataset.node2parents
    candidate_positions = sorted(list(test_dataset.all_positions))
    
    if args_outer.include_test_candidates:
        candidate_positions.extend(vocab)
        logger.info("Including test set nodes as candidate parents for other test set nodes.")

    if args_outer.dih == "weeds":
        # weeds is not well defined here (since, no concept of 0/1 vector)
        logger.info("weeds precision is not well defined for dense Word2Vec embeddings")
        return

    if args_outer.dih == "invcl":
        logger.info("not yet supported")
        return

    if args_outer.load_scores:
        with open(args_outer.scores_path, "rb") as f:
            S = pickle.load(f)

    else:
        M = test_dataset.node_features[candidate_positions, :]
        S = []

        for i, q in enumerate(vocab):
            qv = test_dataset.node_features[q]

            if args_outer.dih == "clarke":
                t = qv.sum()
                qs = np.minimum(qv, M).sum(axis=1) / t

            elif args_outer.dih == "cosine":
                t = np.linalg.norm(qv)
                qs = np.dot(M, qv).flatten() / t

            S.append(qs)
            
            if i % 100 == 0:
                logger.info("processed query", i)

        S = np.array(S)

        with open(args_outer.scores_path, 'wb') as f:
            pickle.dump(S, f, protocol=pickle.HIGHEST_PROTOCOL)

    K = args_outer.candidate_len
    test_edges = []
    for i, q in enumerate(vocab):
        qv = S[i]
        indices = np.argpartition(S, -K)[-K:]
        test_edges.extend([(candidate_positions[index], q, {"weight": S[index]}) for index in indices if candidate_positions[index] in vocab])

    G = nx.DiGraph()
    G.add_nodes_from(vocab)
    G.add_edges_from(test_edges)
    logger.info(f"Model Test Graph: Number of nodes: {G.number_of_nodes()}")
    logger.info(f"Model Test Graph: Number of edges: {G.number_of_edges()}")
    
    if args_outer.pruning == "nocyc":
        logger.info("prunning: nocyc")
        T = NoCyc(G)
    elif args_outer.pruning == "dmst":
        logger.info("pruning: dmst")
        T = DMST(G, args_outer.dmst_wt)

    logger.info(f"Model Test Taxonomy: Number of nodes: {T.number_of_nodes()}")
    logger.info(f"Model Test Taxonomy: Number of edges: {T.number_of_edges()}")

    order = list(nx.topological_sort(T))

    with open(args_outer.taxo_path, "wb") as f:
        pickle.dump(T, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    return order, T


def NoCyc(G):
    G_copy = G.copy()

    active = True
    while active:
        try:
            components = list(nx.strongly_connected_components(G_copy))
            active = False        
            for scc in components:
                G_scc = G_copy.subgraph([node for node in G_copy.nodes() if node in scc])
                edges = G_scc.edges(data="weight")
                if len(edges) > 0:
                    active = True
                    e = min(edges, key=lambda t: t[2])
                    # logger.info("removing edge:", e)
                    G_copy.remove_edge(e[0], e[1])
        except:
            break

    return G_copy


def DMST(G, dummy_wt=0.1):
    G_copy = G.copy()
    
    edge_map = dict()
    for e in G_copy.edges(data="weight"):
        edge_map[(e[0], e[1])] = e[2]
    
    G_copy.add_node("-1")
    for node in G_copy.nodes():
        if str(node) == "-1":
            root = node
    for node in G_copy.nodes():
        if str(node) != "-1":
            G_copy.add_edge(root, node, weight=dummy_wt)
    
    T = nx.maximum_spanning_tree(G_copy.to_undirected()).to_directed()
    
    rem = list()
    attr = dict()
    for e in T.edges(data="weight"):
        p1 = (e[0], e[1])
        p2 = (e[1], e[0])
        if p1 not in edge_map:
            rem.append(e)
        elif (p2 not in edge_map) or (edge_map[p1] > edge_map[p2]):
            attr[p1] = {"weight": edge_map[p1]}
        else:
            rem.append(e)
    
    nx.set_edge_attributes(T, attr)
    for r in rem:
        T.remove_edge(r[0], r[1])
    T.remove_node(root)

    return T


def main_sequential(config, args_outer, vocab):
    logger = config.get_logger('test')

    # case_study or not
    need_case_study = (args_outer.case != "")
    if need_case_study:
        logger.info(f"save case study results to {args_outer.case}")
    else:
        logger.info("no need to save case study results")

    # setup multiprocessing instance
    torch.multiprocessing.set_sharing_strategy('file_system')

    # setup data_loader instances
    if args_outer.test_data == "":
        test_data_path = config['test_data_loader']['args']['data_path']
    else:
        test_data_path = args_outer.test_data

    if "dep_aware" in config['test_data_loader']['args']:
        dep_aware = config['test_data_loader']['args']['dep_aware']
    else:
        dep_aware = False

    test_data_loader = module_data.MaskedGraphDataLoader(
        mode="test", 
        data_path=test_data_path,
        dep_aware=dep_aware, 
        sampling_mode=0,
        batch_size=1, 
        expand_factor=config['test_data_loader']['args']['expand_factor'], 
        shuffle=True, 
        num_workers=8, 
        batch_type="large_batch", 
        cache_refresh_time=config['test_data_loader']['args']['cache_refresh_time'],
        normalize_embed=config['test_data_loader']['args']['normalize_embed'],
        test_topk=args_outer.topk
    )
    logger.info(test_data_loader)
        
    # build model architecture
    model = config.initialize('arch', module_arch)
    logger.info(model)

    # get function handles of loss and metrics
    metric_fns = [getattr(module_metric, met) for met in config['metrics']]
    if config['loss'].startswith("info_nce"):
        pre_metric = partial(module_metric.obtain_ranks, mode=1)  # info_nce_loss
    else:
        pre_metric = partial(module_metric.obtain_ranks, mode=0)

    logger.info('Loading checkpoint: {} ...'.format(config.resume))
    checkpoint = torch.load(config.resume)
    state_dict = checkpoint['state_dict']
    if config['n_gpu'] > 1:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(state_dict)

    # prepare model for testing
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    test_dataset = test_data_loader.dataset
    kv = test_dataset.kv
    # vocab = test_dataset.node_list

    if need_case_study:
        indice2word = test_dataset.vocab
    node2parents = test_dataset.node2parents
    candidate_positions = sorted(list(test_dataset.all_positions))
    logger.info(f"Number of queries: {len(vocab)}")
    
    if args_outer.include_test_candidates:
        print("Sequentially including inserted query nodes as candidate parents for new query nodes.")
    
    if args_outer.batch_size == -1:  # small dataset with only one batch
        logger.info('Small batch mode')

        anchor2subgraph = {}
        for anchor in tqdm(candidate_positions):
            anchor2subgraph[anchor] = test_dataset._get_subgraph(-1, anchor, 0)
        
        # start sequential per query prediction
        total_metrics = torch.zeros(len(metric_fns))
        if need_case_study:
            all_cases = []
            all_cases.append(["Test node index", "True parents", "Predicted parents"] + [fn.__name__ for fn in metric_fns])

        with torch.no_grad():
            for i, query in tqdm(enumerate(vocab)):

                # obtain graph representation
                bg = dgl.batch([v for k,v in anchor2subgraph.items()])
                h = bg.ndata.pop('x').to(device)
                candidate_position_idx = bg.ndata['_id'][bg.ndata['pos']==1].tolist()
                n_position = len(candidate_position_idx)
                pos = bg.ndata['pos'].to(device)
                with torch.no_grad():
                    hg = encode_graph(model, bg, h, pos)

                if need_case_study:
                    cur_case = [indice2word[query]]
                    true_parents = ", ".join([indice2word[ele] for ele in node2parents[query]])
                    cur_case.append(true_parents)

                nf = torch.tensor(kv[str(query)], dtype=torch.float32).to(device)
                
                expanded_nf = nf.expand(n_position, -1)
                energy_scores = model.match(hg, expanded_nf)

                if need_case_study:  # select top-5 predicted parents
                    predicted_scores = energy_scores.cpu().squeeze_().tolist()
                    if not config['loss'].startswith("info_nce"):
                        predicted_scores = [-s for s in predicted_scores]
                    sorted_parents = sorted(enumerate(predicted_scores), key=lambda x:x[1], reverse=True)
                    predict_parent_idx_list = [candidate_position_idx[ele[0]] for ele in sorted_parents[:5]]
                    predict_parents = ", ".join([indice2word[ele] for ele in predict_parent_idx_list])
                    cur_case.append(predict_parents)

                energy_scores, labels, _ = rearrange(energy_scores, candidate_position_idx, node2parents[query])
                all_ranks = pre_metric(energy_scores, labels)
                for j, metric in enumerate(metric_fns):
                    tmp = metric(all_ranks)
                    total_metrics[j] += tmp
                    if need_case_study:
                        cur_case.append(str(tmp))
                if need_case_study:
                    all_cases.append(cur_case)

                if args_outer.include_test_candidates:
                    candidate_positions.append(query)
                    anchor2subgraph[query] = test_dataset._get_subgraph(-1, query, 0, True)

        # save case study results to file
        if need_case_study:
            with open(args_outer.case, "w") as fout:
                for ele in all_cases:
                    fout.write("\t".join(ele))
                    fout.write("\n")

    else:  # large dataset with many batches
        # obtain graph representation
        logger.info(f'Large batch mode with batch_size = {args_outer.batch_size}')
        
        batched_hg, batched_positions = make_batches(args_outer, test_dataset, model, device, candidate_positions, False)
        orig_batches_len = len(batched_hg)

        candidate_position_idx = list(itertools.chain(*batched_positions))
       
        # start per query prediction
        total_metrics = torch.zeros(len(metric_fns))
        if need_case_study:
            all_cases = []
            all_cases.append(["Test node index", "True parents", "Predicted parents"] + [fn.__name__ for fn in metric_fns])

        seen_vocab = []
        invalids = []
        with torch.no_grad():
            for i, query in tqdm(enumerate(vocab)):
                if need_case_study:
                    cur_case = [indice2word[query]]
                    true_parents = ", ".join([indice2word[ele] for ele in node2parents[query]])
                    cur_case.append(true_parents)
                nf = torch.tensor(kv[str(query)], dtype=torch.float32).to(device)
                
                
                batched_energy_scores = []
                for hg, _positions in zip(batched_hg, batched_positions):
                    n_position = len(_positions)
                    expanded_nf = nf.expand(n_position, -1)
                    energy_scores = model.match(hg, expanded_nf)  # a tensor of size (n_position, 1)
                    batched_energy_scores.append(energy_scores)
                batched_energy_scores = torch.cat(batched_energy_scores)
                if need_case_study:
                    predicted_scores = batched_energy_scores.cpu().squeeze_().tolist()
                    if not config['loss'].startswith("info_nce"):
                        predicted_scores = [-s for s in predicted_scores]
                    sorted_parents = sorted(enumerate(predicted_scores), key=lambda x:x[1], reverse=True)
                    predict_parent_idx_list = [candidate_position_idx[ele[0]] for ele in sorted_parents[:5]]
                    predict_parents = ", ".join([indice2word[ele] for ele in predict_parent_idx_list])
                    cur_case.append(predict_parents)

                batched_energy_scores, labels, corrects = rearrange(batched_energy_scores, candidate_position_idx, node2parents[query])
                child_before_parent = (corrects == 0)
                if child_before_parent:
                    invalids.append(query)
                all_ranks = pre_metric(batched_energy_scores, labels)
                for j, metric in enumerate(metric_fns):
                    tmp = metric(all_ranks)
                    if not child_before_parent:
                        total_metrics[j] += tmp
                    if need_case_study:
                        cur_case.append(str(tmp))
                if need_case_study:
                    all_cases.append(cur_case)

                if args_outer.include_test_candidates:                    
                    if i % 200 == 0: # reform batches
                        temp_batched_hg, temp_batched_positions = make_batches(args_outer, test_dataset, model, device, seen_vocab, True)
                        del batched_hg[orig_batches_len:]
                        del batched_positions[orig_batches_len:]
                        batched_hg.extend(temp_batched_hg)
                        batched_positions.extend(temp_batched_positions)

                    candidate_position_idx.append(query)
                    seen_vocab.append(query)
                    bg = [test_dataset._get_subgraph(-1, query, 0, True)]
                    positions = [query]

                    # the attributes of previously used elements of bg are different from the newly added one, hence the prev. used batch cannot be updated
                    batched_bg = dgl.batch(bg)
                    h = batched_bg.ndata.pop('x').to(device)
                    pos = batched_bg.ndata['pos'].to(device)
                    hg = encode_graph(model, batched_bg, h, pos)
                    assert hg.shape[0] == len(positions), f"mismatch between hg.shape[0]: {hg.shape[0]} and len(positions): {len(positions)}"
                    batched_positions.append(positions)
                    batched_hg.append(hg.to(device))
                    del h

        # save case study results to file
        if need_case_study:
            with open(args_outer.case, "w") as fout:
                for ele in all_cases:
                    fout.write("\t".join(ele))
                    fout.write("\n")

    logger.info("child before parent cases: ", len(invalids))
    logger.info("invalids: ", invalids)
    n_samples = max(1, test_data_loader.n_samples - len(invalids))
    log = {}
    log.update({
        met.__name__: total_metrics[i].item() / n_samples for i, met in enumerate(metric_fns)
    })
    log.update({
        "test_topk": test_data_loader.dataset.test_topk
    })
    logger.info(log)


def make_batches(args_outer, test_dataset, model, device, entries_to_take, only_anchor):
    batched_hg = []  # save the CPU graph representation
    batched_positions = []
    bg = []
    positions = []
    with torch.no_grad():
        for i, anchor in enumerate(entries_to_take):
            egonet = test_dataset._get_subgraph(-1, anchor, 0, only_anchor)
            positions.append(anchor)
            bg.append(egonet)
            if (i+1) % args_outer.batch_size == 0:
                bg = dgl.batch(bg)
                h = bg.ndata.pop('x').to(device)
                pos = bg.ndata['pos'].to(device)
                hg = encode_graph(model, bg, h, pos)
                assert hg.shape[0] == len(positions), f"mismatch between hg.shape[0]: {hg.shape[0]} and len(positions): {len(positions)}"
                batched_hg.append(hg.cpu())
                batched_positions.append(positions)
                bg = []
                positions = []
                del h
        if len(bg) != 0:
            bg = dgl.batch(bg)
            h = bg.ndata.pop('x').to(device)
            pos = bg.ndata['pos'].to(device)
            hg = encode_graph(model, bg, h, pos)
            assert hg.shape[0] == len(positions), f"mismatch between hg.shape[0]: {hg.shape[0]} and len(positions): {len(positions)}"
            batched_hg.append(hg.cpu())
            batched_positions.append(positions)
            del h

    batched_hg = [hg.to(device) for hg in batched_hg]

    return batched_hg, batched_positions


def edge_metrics(T_true, T_pred, verbose=False):
    edges_gold = set(T_true.edges())
    edges_pred = set(T_pred.edges())
    edges_common = edges_gold & edges_pred
    precision = len(edges_common) / len(edges_pred)
    recall = len(edges_common) / len(edges_gold)
    f1 = 2.0*precision*recall/(precision+recall)
    if verbose:
        print(f"edge precision: {precision} ({len(edges_common)}/{len(edges_pred)})")
        print(f"edge recall: {recall} ({len(edges_common)}/{len(edges_gold)})")
        print(f"edge f1: {f1}")
    
    return {"edge_p": precision, "edge_r": recall, "edge_f1": f1}


def ancestor_metrics(T_true, T_pred, verbose=False):
    edges_gold = set(nx.dag.transitive_closure(T_true).edges())
    edges_pred = set(nx.dag.transitive_closure(T_pred).edges())
    edges_common = edges_gold & edges_pred
    precision = len(edges_common) / len(edges_pred)
    recall = len(edges_common) / len(edges_gold)
    f1 = 2.0*precision*recall/(precision+recall)
    if verbose:
        print(f"ancestor precision: {precision} ({len(edges_common)}/{len(edges_pred)})")
        print(f"ancestor recall: {recall} ({len(edges_common)}/{len(edges_gold)})")
        print(f"ancestor f1: {f1}")
    
    return {"ancestor_p": precision, "ancestor_r": recall, "ancestor_f1": f1}


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='Testing taxonomy expansion model')
    args.add_argument('-td', '--test_data', default="", type=str, help='test data path, if not provided, we assume the test data is specificed in the config file')
    args.add_argument('-tc', '--include_test_candidates', default=False, type=bool, help='include nodes in test set as candidates parents for other nodes in test set (Default: False)')
    args.add_argument('-r', '--resume', required=True, type=str, help='path to latest checkpoint')
    args.add_argument('-d', '--device', default=None, type=str, help='indices of GPUs to enable (default: all)')
    args.add_argument('-k', '--topk', default=-1, type=int, help='topk retrieved instances for testing, -1 means no retrieval stage (default: -1)')
    args.add_argument('-b', '--batch_size', default=-1, type=int, help='batch size, -1 for small dataset (default: -1), 30000 for larger MAG-Full data')
    args.add_argument('-c', '--case', default="", type=str, help='case study saving file, if is "", no need to get case studies (default: "")')

    args.add_argument('-otp', '--optimal_taxo_path', default="optimal.taxo.pkl", type=str, help='path to optimal taxonomy (default: "optimal.taxo.pkl")')
    args.add_argument('-mtp', '--model_taxo_path', default="model.taxo.pkl", type=str, help='path to model predicted taxonomy (default: "model.taxo.pkl")')
    args.add_argument('-dih', '--dih', default="cosine", type=str, help='DIH metric to be used for parent identification(cosine/clarke) (default: "cosine")')
    args.add_argument('-p', '--pruning', default="dmst", type=str, help='graph pruning strategy (dmst/nocyc) (default: "dmst")')
    args.add_argument('-ss', '--scores_path', default="model.scores.pkl", type=str, help='path where model scores are saved (default: "model.scores.pkl")')
    args.add_argument('-cl', '--candidate_len', default=100, type=int, help='no. of candidate parents to be selected (default: 20)')
    args.add_argument('-dw', '--dmst_wt', default=0.001, type=float, help='dummy edge wt. for DMST (default: 0.1)')
    
    args.add_argument('-ls', '--load_scores', default=False, type=bool, help='if set, we load pre-computed model scores (default: True)')
    args.add_argument('-eval', '--direct_eval', default=True, type=bool, help='if set, we do direct evaluation (edge F1 etc.) between optimal and model taxonomies (default: True)')
    args.add_argument('-opt', '--optimal', default=True, type=bool, help='if set, we compute optimal taxonomy (default: True)')
    args.add_argument('-m', '--model', default=True, type=bool, help='if set, we compute model-based taxonomy (default: True)')
    args.add_argument('-i', '--insert', default=True, type=bool, help='if set, we do sequential insertion evaluation (default: True)')
    
    args_outer = args.parse_args()
    config = ConfigParser(args)

    if args_outer.optimal:
        vocab_optimal, T_opt = get_optimal_ordering(config, args_outer)

    else:
        with open(args_outer.optimal_taxo_path, "rb") as f:
            T_opt = pickle.load(f)
        vocab_optimal = list(nx.topological_sort(T_opt))

    if args_outer.model:
        vocab_model, T_model = get_insertion_ordering(config, args_outer)

    else:
        with open(args_outer.model_taxo_path, "rb") as f:
            T_model = pickle.load(f)
        vocab_model = list(nx.topological_sort(T_model))     

    if args_outer.direct_eval: 
        logger = config.get_logger('test')
        logger.info(edge_metrics(T_opt, T_model))
        logger.info(ancestor_metrics(T_opt, T_model))  

    if args_outer.insert:
        main_sequential(config, args_outer, vocab_model)

    # reverse optimal
    # rev_optimal = list(reversed(vocab_optimal))
    # main_sequential(config, args_outer, rev_optimal)
    
    # random order insertion
    # vocab_random = [vocab_optimal[i] for i in np.random.permutation(len(vocab_optimal))]
    # main_sequential(config, args_outer, vocab_random)
    