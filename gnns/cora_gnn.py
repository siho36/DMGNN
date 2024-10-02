import argparse
import os
import os.path as osp
import time
import sys
sys.path.append("..")
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear as Lin, ModuleList, ReLU, Softmax
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.nn import BatchNorm, GCNConv
from utils import set_seed, accuracy
from torch_geometric.datasets import Planetoid,Flickr
import torch_geometric.transforms as T

EPS = 1


def parse_args():
    parser = argparse.ArgumentParser(description="Train Model")

    parser.add_argument('--dataset', type=str, default='Cora', 
                    help='Dataset',
                    choices=['Cora','Pubmed','Flickr','ogbn-arxiv'])
    parser.add_argument(
        "--model_path",
        nargs="?",
        default=osp.join(osp.dirname(__file__), "..", "param", "gnns"),
        help="path for saving trained model.",
    )
    parser.add_argument("--cuda", type=int, default=0, help="GPU device.")
    parser.add_argument("--epoch", type=int, default=3000, help="Number of epoch.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size.")
    parser.add_argument("--hidden", type=int, default=128, help="hiden size.")
    parser.add_argument("--verbose", type=int, default=10, help="Interval of evaluation.")
    parser.add_argument("--num_unit", type=int, default=2, help="number of Convolution layers(units)")
    parser.add_argument(
        "--random_label", type=bool, default=False, help="train a model under label randomization for sanity check"
    )
    return parser.parse_args()


class GCN(torch.nn.Module):
    def __init__(self, num_unit, n_input, n_out, n_hid):
        super(GCN, self).__init__()
        self.convs = ModuleList()
        self.batch_norms = ModuleList()
        self.relus = ModuleList()
        self.edge_emb = Lin(3, 1)
        self.relus.extend([ReLU()] * num_unit)
        self.convs.append(GCNConv(in_channels=n_input, out_channels=n_hid))
        # self.convs.append(GINConv(nn=Seq(Lin(n_input, n_hid), self.relus[0], Lin(n_hid, n_hid))))
        for i in range(num_unit - 2):
            # self.convs.append(GINConv(nn=Seq(Lin(n_hid, n_hid), self.relus[i+1], Lin(n_hid, n_hid))))
            self.convs.append(GCNConv(in_channels=n_hid, out_channels=n_hid))
        self.convs.append(GCNConv(in_channels=n_hid, out_channels=n_hid))
        # self.convs.append(GINConv(nn=Seq(Lin(n_hid, n_hid), self.relus[-1], Lin(n_hid, n_hid))))
        self.batch_norms.extend([BatchNorm(n_hid)] * num_unit)

        # self.lin1 = Lin(128, 128)
        self.ffn = nn.Sequential(*([nn.Linear(n_hid, n_hid)] + [ReLU(), nn.Dropout(), nn.Linear(n_hid, n_out)]))

        self.softmax = Softmax(dim=1)
        self.dropout = 0

    def forward(self, x, edge_index, edge_attr=None):
        for conv, batch_norm, relu in zip(self.convs, self.batch_norms, self.relus):
            x = conv(x, edge_index)
            x = F.dropout(x, self.dropout, training=self.training)
            # x = relu(batch_norm(x))
            x = relu(x)
        # graph_x = global_mean_pool(x, batch)
        # x_res = torch.cat(xx, dim=1)
        pred = self.ffn(x)  # [node_num, n_class]
        self.readout = self.softmax(pred)
        return pred

    def get_node_reps(self, x, edge_index, edge_attr=None):
        for conv, batch_norm, relu in zip(self.convs, self.batch_norms, self.relus):
            x = conv(x, edge_index)
            x = F.dropout(x, self.dropout, training=self.training)
            x = relu(x)
        node_x = self.ffn[0](x)
        node_x = F.relu(node_x)
        node_x = F.dropout(node_x)
        return node_x

    def get_node_pred_subgraph(self, x, edge_index, mapping=None):
        for conv, batch_norm, relu in zip(self.convs, self.batch_norms, self.relus):
            x = conv(x, edge_index)
            x = F.dropout(x, self.dropout, training=self.training)
            # x = relu(batch_norm(x))
            x = relu(x)
        node_repr = self.ffn(x)  # [node_num, n_class]
        node_prob = self.softmax(node_repr)
        output_prob = node_prob[mapping]  # [bsz, n_classes]
        output_repr = node_repr[mapping]  # [bsz, n_classes]
        return output_prob, output_repr

    def get_pred_explain(self, x, edge_index, edge_mask, mapping=None):
        edge_mask = (edge_mask * EPS).sigmoid()
        for conv, batch_norm, relu in zip(self.convs, self.batch_norms, self.relus):
            x = conv(x, edge_index, edge_weight=edge_mask)
            x = F.dropout(x, self.dropout, training=self.training)
            # x = ReLU(batch_norm(x))
            x = relu(x)
        node_repr = self.ffn(x)  # [node_num, n_class]
        node_prob = self.softmax(node_repr)
        output_prob = node_prob[mapping]  # [bsz, n_classes]
        output_repr = node_repr[mapping]  # [bsz, n_classes]
        return output_prob, output_repr

    def reset_parameters(self):
        with torch.no_grad():
            for param in self.parameters():
                param.uniform_(-1.0, 1.0)

if __name__ == "__main__":
    set_seed(44)
    args = parse_args()
    device = torch.device(f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu")
    name = args.dataset

    transform = T.Compose([T.NormalizeFeatures()])

    if(args.dataset == 'Cora' or args.dataset == 'Pubmed'):
        dataset = Planetoid(root='../data/', \
                            name=args.dataset,\
                            transform=transform)
    elif(args.dataset == 'Flickr'):
        dataset = Flickr(root='./data/Flickr/', \
                        transform=transform)
    data = dataset[0].to(device)

    ########################修改训练数据集比例，与DPGBA一致################################
    idx_train = torch.load('../data/Cora/GBA/idx_train.pth')
    idx_val = torch.load('../data/Cora/GBA/idx_val.pth')
    idx_clean_test = torch.load('../data/Cora/GBA/idx_clean_test.pth')
    idx_atk_test = torch.load('../data/Cora/GBA/idx_atk_test.pth')
    idx_test = torch.cat((idx_clean_test, idx_atk_test))
    data.train_mask = torch.zeros_like(data.train_mask)
    data.train_mask[idx_train] = True
    data.val_mask = torch.zeros_like(data.val_mask)
    data.val_mask[idx_val] = True
    data.clean_test_mask = torch.zeros_like(data.test_mask)
    data.clean_test_mask[idx_clean_test] = True
    data.atk_test_mask = torch.zeros_like(data.test_mask)
    data.atk_test_mask[idx_atk_test] = True
    ###########################################################################

    ###############################注入训练集后门#####################################
    poison_x = torch.load('../data/Cora/GBA/poison_x.pth')                    
    poison_edge_index = torch.load('../data/Cora/GBA/poison_edge_index.pth')   
    poison_edge_weights = torch.load('../data/Cora/GBA/poison_edge_weights.pth')
    poison_labels = torch.load('../data/Cora/GBA/poison_labels.pth')         
    ###############################注入测试集后门#####################################
    induct_x = torch.load('../data/Cora/GBA/induct_x.pth')                   
    induct_edge_index = torch.load('../data/Cora/GBA/induct_edge_index.pth')  
    induct_edge_weights = torch.load('../data/Cora/GBA/induct_edge_weights.pth') 
    poison_labels = poison_labels
    induct_x = induct_x.detach()
    induct_edge_index = induct_edge_index

    data.raw_x = data.x 
    data.x = induct_x
    data.edge_index = induct_edge_index

    data.y = poison_labels

    n_input = data.x.size(1)
    n_labels = int(torch.unique(data.y).size(0))
    model = GCN(args.num_unit, n_input=n_input, n_out=n_labels, n_hid=args.hidden).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.8, patience=10, min_lr=1e-5)
    min_error = None
    criterion = nn.CrossEntropyLoss()

    def train(epoch):
        t = time.time()
        model.train()
        optimizer.zero_grad()
        output = model(x=data.x, edge_index=data.edge_index)
        output = output[:2708]
        loss_train = criterion(output[data.train_mask], data.y[data.train_mask])
        y_pred = torch.argmax(output, dim=1)
        acc_train = accuracy(y_pred[data.train_mask], data.y[data.train_mask])
        loss_train.backward()
        optimizer.step()
        print(
            "Epoch: {:04d}".format(epoch + 1),
            "loss_train: {:.4f}".format(loss_train.item()),
            "acc_train: {:.4f}".format(acc_train),
            "time: {:.4f}s".format(time.time() - t),
        )

    def eval():
        model.eval()
        output = model(x=data.x, edge_index=data.edge_index)
        output = output[:2708]
        clean_loss_test = criterion(output[data.clean_test_mask], data.y[data.clean_test_mask])
        clean_y_pred = torch.argmax(output, dim=1)
        clean_acc_test = accuracy(clean_y_pred[data.clean_test_mask], data.y[data.clean_test_mask])
        print("clean_Test set results:", "loss= {:.4f}".format(clean_loss_test.item()), "accuracy= {:.4f}".format(clean_acc_test))

        atk_loss_test = criterion(output[data.atk_test_mask], data.y[data.atk_test_mask])
        atk_y_pred = torch.argmax(output, dim=1)
        atk_acc_test = accuracy(atk_y_pred[data.atk_test_mask], data.y[data.atk_test_mask])
        print("atk_Test set results:", "loss= {:.4f}".format(atk_loss_test.item()), "accuracy= {:.4f}".format(atk_acc_test))
        return clean_loss_test, clean_y_pred, atk_loss_test, atk_y_pred

    for epoch in range(1, args.epoch + 1):
        train(epoch)

        if epoch % args.verbose == 0:
            clean_loss_test, clean_y_pred, atk_loss_test, atk_y_pred = eval()
            scheduler.step(clean_loss_test)

    save_path = f"{name}_gcn.pt"

    if not osp.exists(args.model_path):
        os.makedirs(args.model_path)
    torch.save(model.cpu(), osp.join(args.model_path, save_path))
    clean_labels = data.y[data.clean_test_mask].cpu().numpy()
    clean_pred = clean_y_pred[data.clean_test_mask].cpu().numpy()
    print("clean_y_true counts: {}".format(np.unique(clean_labels, return_counts=True)))
    print("clean_y_pred_orig counts: {}".format(np.unique(clean_pred, return_counts=True)))
    atk_labels = data.y[data.atk_test_mask].cpu().numpy()
    atk_pred = atk_y_pred[data.atk_test_mask].cpu().numpy()
    print("atk_y_true counts: {}".format(np.unique(atk_labels, return_counts=True)))
    print("atk_y_pred_orig counts: {}".format(np.unique(atk_pred, return_counts=True)))
