import os.path as osp
import numpy as np
import torch
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import k_hop_subgraph, subgraph

def get_neighbourhood(node_idx, edge_index, features, labels, n_hops):
    edge_subset = k_hop_subgraph(node_idx, n_hops, edge_index)  
    edge_subset_relabel = subgraph(edge_subset[0], edge_index, relabel_nodes=True)
    edge_index_sub = edge_subset_relabel[0] 
    sub_feat = features[edge_subset[0], :] 
    sub_labels = labels[edge_subset[0]] 
    self_label = labels[node_idx] 
    node_dict = torch.tensor(edge_subset[0]).reshape(-1, 1)  
    mapping = edge_subset[2]
    mapping_mask = torch.zeros((sub_feat.shape[0]))
    mapping_mask[mapping] = 1
    mapping_mask = mapping_mask.bool()
    return sub_feat, edge_index_sub, sub_labels, self_label, node_dict, mapping_mask

class CoraDataset(InMemoryDataset):
    def __init__(self, root, name, mode="clean_testing", transform=None, pre_transform=None):
        self.name = name
        self.mode = mode
        super(CoraDataset, self).__init__(root, transform, pre_transform)
        idx = self.processed_file_names.index("{}_sub.pt".format(mode))
        self.data, self.slices = torch.load(self.processed_paths[idx])

    @property
    def raw_dir(self):
        return osp.join(self.root, self.name, "raw")

    @property
    def processed_dir(self):
        return osp.join(self.root, self.name, "processed")

    @property
    def raw_file_names(self):
        return [f"{self.name}.pkl"]

    @property
    def processed_file_names(self):
        return ["training_sub.pt", "evaluating_sub.pt", "clean_testing_sub.pt", "atk_testing_sub.pt"]

    def process(self):
        dataset = Planetoid(root=self.root, name='Cora')
        data = dataset[0]

        edge_index = data.edge_index  
        x = data.x                
        y = data.y                  
        raw_x = x

        ########################training data ratio################################
        idx_train = torch.load('./data/Cora/GBA/idx_train.pth')
        idx_val = torch.load('./data/Cora/GBA/idx_val.pth')
        idx_clean_test = torch.load('./data/Cora/GBA/idx_clean_test.pth')
        idx_atk_test = torch.load('./data/Cora/GBA/idx_atk_test.pth')
        idx_test = torch.cat((idx_clean_test, idx_atk_test))
        data.train_mask = torch.zeros_like(data.train_mask)
        data.train_mask[idx_train] = True
        data.val_mask = torch.zeros_like(data.val_mask)
        data.val_mask[idx_val] = True
        data.clean_test_mask = torch.zeros_like(data.test_mask)
        data.clean_test_mask[idx_clean_test] = True
        data.atk_test_mask = torch.zeros_like(data.test_mask)
        data.atk_test_mask[idx_atk_test] = True
        

        train_mask_data = data.train_mask
        val_mask_data = data.val_mask
        clean_test_mask_data = data.clean_test_mask
        atk_test_mask_data = data.atk_test_mask


        ###############################backdoor train data#####################################
        poison_x = torch.load('./data/Cora/GBA/poison_x.pth')                     
        poison_edge_index = torch.load('./data/Cora/GBA/poison_edge_index.pth')   
        poison_edge_weights = torch.load('./data/Cora/GBA/poison_edge_weights.pth') 
        poison_labels = torch.load('./data/Cora/GBA/poison_labels.pth')            
        
        ###############################backdoor test data#####################################
        induct_x = torch.load('./data/Cora/GBA/induct_x.pth')                     
        induct_edge_index = torch.load('./data/Cora/GBA/induct_edge_index.pth')    
        induct_edge_weights = torch.load('./data/Cora/GBA/induct_edge_weights.pth') 

        poison_labels = poison_labels.cpu()
        induct_x = induct_x.cpu().detach()
        induct_edge_index = induct_edge_index.cpu()

        x = induct_x
        edge_index = induct_edge_index
        trigger_list = [1] * 843
        trigger_label = torch.tensor(trigger_list)
        y = torch.cat((poison_labels, trigger_label), dim=0)


        data_whole = Data(x=x, edge_index=edge_index, y=y)
        data_whole.train_mask = train_mask_data
        data_whole.val_mask = val_mask_data
        data_whole.clean_test_mask = clean_test_mask_data
        data_whole.atk_test_mask = atk_test_mask_data
        torch.save(data_whole, f"./data/{self.name}/processed/whole_graph.pt")


        data_list = []
        for id in range(raw_x.shape[0]):
            (
                sub_feat,
                edge_index_sub,
                sub_labels,
                self_label,
                node_dict,
                mapping_mask,
            ) = get_neighbourhood(id, edge_index, features=x, labels=y, n_hops=5)
            data = Data(
                x=sub_feat,
                edge_index=edge_index_sub,
                y=sub_labels,
                self_y=self_label,
                node_dict=node_dict,
                mapping=mapping_mask,
                idx=id,
            )
            if self.pre_filter is not None and not self.pre_filter(data):
                continue
            if self.pre_transform is not None:
                data = self.pre_transform(data)
            data_list.append(data)

        train_mask = list(np.where(train_mask_data)[0])
        val_mask = list(np.where(val_mask_data)[0])
        clean_test_mask = list(np.where(clean_test_mask_data)[0])
        atk_test_mask = list(np.where(atk_test_mask_data)[0])
        torch.save(
            self.collate([data_list[i] for i in train_mask]),
            f"./data/{self.name}/processed/training_sub.pt",
        )
        torch.save(
            self.collate([data_list[i] for i in val_mask]),
            f"./data/{self.name}/processed/evaluating_sub.pt",
        )
        torch.save(
            self.collate([data_list[i] for i in clean_test_mask]),
            f"./data/{self.name}/processed/clean_testing_sub.pt",
        )
        torch.save(
            self.collate([data_list[i] for i in atk_test_mask]),
            f"./data/{self.name}/processed/atk_testing_sub.pt",
        )