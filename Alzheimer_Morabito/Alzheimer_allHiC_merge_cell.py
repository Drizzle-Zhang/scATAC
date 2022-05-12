# _*_ coding: utf-8 _*_
# @author: Drizzle_Zhang
# @file: ATAC_GCN.py
# @time: 2021/8/24 14:39


from time import time
import os
from torch_geometric.data import InMemoryDataset, Data, DataLoader
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GraphConv, TopKPooling
from captum.attr import Saliency, IntegratedGradients
from collections import defaultdict
from scipy.stats import kstest
import episcanpy.api as epi
import scanpy as sc
import pandas as pd
import anndata as ad
import pickle
import copy
import random
from sklearn.cluster import k_means
from multiprocessing import Pool
from functools import partial
from sklearn import preprocessing


class ATACDataset(object):
    def __init__(self, data_root, raw_filename):
        self.data_root = data_root
        self.raw_filename = raw_filename
        self.adata = self.load_matrix()
        self.path_process = os.path.join(data_root, 'processed_files')
        if not os.path.exists(self.path_process):
            os.mkdir(self.path_process)
        self.file_peaks_sort = os.path.join(self.path_process, 'peaks.sort.bed')
        self.all_genes = None
        self.adata_merge = None
        self.other_peaks = None
        self.df_graph = None
        self.list_graph = None
        self.array_peak = None
        self.array_celltype = None

    def load_matrix(self):
        if self.raw_filename[-5:] == '.h5ad':
            adata = sc.read_h5ad(os.path.join(self.data_root, self.raw_filename))
        elif self.raw_filename[-4:] == '.tsv':
            adata = ad.read_text(
                os.path.join(self.data_root, self.raw_filename),
                delimiter='\t', first_column_names=True, dtype='int')
        else:
            raise ImportError("Input format error!")
        return adata

    def generate_peaks_file(self):
        file_peaks = os.path.join(self.path_process, 'peaks.bed')
        fmt_peak = "{chrom}\t{start}\t{end}\t{peak_id}\n"
        with open(file_peaks, 'w') as w_peak:
            for one_peak in self.adata.var.index:
                chrom = one_peak.strip().split(':')[0]
                locs = one_peak.strip().split(':')[1]
                start = locs.strip().split('-')[0]
                end = locs.strip().split('-')[1]
                peak_id = one_peak
                w_peak.write(fmt_peak.format(**locals()))

        os.system(f"bedtools sort -i {file_peaks} > {self.file_peaks_sort}")

    def quality_control(self, min_features=1000, max_features=50000,
                        min_percent=None, min_cells=None):
        self.adata.raw = self.adata.copy()
        adata_atac = self.adata
        epi.pp.filter_cells(adata_atac, min_features=min_features)
        epi.pp.filter_cells(adata_atac, max_features=max_features)
        if min_percent is not None:
            df_count = pd.DataFrame(adata_atac.X, index=adata_atac.obs.index,
                                    columns=adata_atac.var.index)
            array_celltype = np.array(adata_atac.obs['celltype'])
            celltypes = np.unique(array_celltype)
            df_percent = pd.DataFrame(
                np.full(shape=(df_count.shape[1], len(celltypes)), fill_value=0),
                index=adata_atac.var.index, columns=celltypes)
            for cell in celltypes:
                sub_count = df_count.loc[array_celltype == cell, :]
                sub_percent = np.sum(sub_count != 0, axis=0) / sub_count.shape[0]
                df_percent[cell] = sub_percent
            df_percent_max = np.max(df_percent, axis=1)
            sel_peaks = df_percent_max.index[df_percent_max > min_percent]
            self.adata = self.adata[:, sel_peaks]
        elif min_cells is not None:
            epi.pp.filter_features(adata_atac, min_cells=min_cells)

    def select_genes(self, num_peak=120000):
        adata = self.adata
        sc.pp.normalize_total(adata)
        sc.pp.log1p(adata)
        sc.pp.highly_variable_genes(adata, n_top_genes=num_peak, flavor='seurat')
        self.adata = self.adata[:, adata.var.highly_variable]

    def find_neighbors(self, num_peak=120000, num_pc=50, num_neighbor=30):
        adata = self.adata
        sc.pp.normalize_total(adata)
        sc.pp.log1p(adata)
        sc.pp.highly_variable_genes(adata, n_top_genes=num_peak, flavor='seurat')
        adata = adata[:, adata.var.highly_variable]
        # sc.pp.scale(adata, max_value=10)
        sc.tl.pca(adata, svd_solver='arpack', n_comps=num_pc)
        sc.pp.neighbors(adata, n_neighbors=num_neighbor, n_pcs=num_pc, metric='cosine', knn=True)
        self.adata = adata

    def plot_umap(self):
        adata = self.adata
        sc.pp.neighbors(adata, n_neighbors=15, n_pcs=50)
        sc.tl.umap(adata)
        out_plot = sc.pl.umap(self.adata, color=['nb_features', 'celltype'])
        return out_plot

    def add_promoter(self, file_promoter):
        if os.path.exists(self.file_peaks_sort):
            os.remove(self.file_peaks_sort)
        self.generate_peaks_file()
        file_peaks_promoter = os.path.join(self.path_process, 'peaks_promoter.txt')
        os.system(f"bedtools intersect -a {self.file_peaks_sort} -b {file_promoter} -wao "
                  f"> {file_peaks_promoter}")
        dict_promoter = defaultdict(list)
        with open(file_peaks_promoter, 'r') as w_pro:
            for line in w_pro:
                list_line = line.strip().split('\t')
                if list_line[4] == '.':
                    continue
                gene = list_line[7].strip().split('<-')[0]
                peak = list_line[3]
                dict_promoter[gene].append(peak)

        all_genes = dict_promoter.keys()
        list_peaks_1 = []
        list_genes_1 = []
        list_peaks_2 = []
        list_genes_2 = []
        for gene in all_genes:
            sub_peaks = dict_promoter[gene]
            if len(sub_peaks) == 1:
                list_peaks_1.extend(sub_peaks)
                list_genes_1.append(gene)
            else:
                list_genes_2.extend([gene for _ in range(len(sub_peaks))])
                list_peaks_2.extend(sub_peaks)
        adata_gene_1 = self.adata[:, list_peaks_1]
        df_gene_peak_1 = pd.DataFrame(adata_gene_1.X.toarray(), index=adata_gene_1.obs.index,
                                      columns=list_genes_1)
        adata_gene_2 = self.adata[:, list_peaks_2]
        df_gene_peak_2 = pd.DataFrame(
            adata_gene_2.X.toarray(),
            index=adata_gene_2.obs.index,
            columns=pd.MultiIndex.from_arrays([list_genes_2, list_peaks_2], names=['gene', 'peak']))
        df_gene_peak_2_t = df_gene_peak_2.T
        df_gene_peak_2_t_gene = df_gene_peak_2_t.groupby('gene').apply(lambda x: x.sum())
        df_gene_peak_2 = df_gene_peak_2_t_gene.T
        all_cols = set(list_peaks_1 + list_peaks_2)
        other_cols = set(self.adata.var.index).difference(all_cols)
        self.other_peaks = other_cols
        adata_other = self.adata[:, [one_peak for one_peak in self.adata.var.index
                                     if one_peak in other_cols]]
        adata_other.var['cRE_type'] = np.full(adata_other.n_vars, 'Other')

        df_gene = pd.concat([df_gene_peak_1, df_gene_peak_2], axis=1)
        adata_promoter = \
            ad.AnnData(X=df_gene,
                       var=pd.DataFrame(data={'cRE_type': np.full(df_gene.shape[1], 'Promoter')},
                                        index=df_gene.columns),
                       obs=self.adata.obs.loc[df_gene.index, :])
        self.all_genes = set(df_gene.columns)
        adata_merge = ad.concat([adata_promoter, adata_other], axis=1)
        self.adata_merge = adata_merge

        return

    def build_graph(self, path_interaction):
        file_pp = os.path.join(path_interaction, 'PP.txt')
        file_po = os.path.join(path_interaction, 'PO.txt')
        df_pp = pd.read_csv(file_pp, sep='\t', header=None)
        df_pp = df_pp.loc[
                df_pp.apply(lambda x:
                            x.iloc[0] in self.all_genes and x.iloc[1] in self.all_genes, axis=1), :]
        df_pp.columns = ['region1', 'region2']
        file_po_peaks = os.path.join(self.path_process, 'peaks_PO.bed')
        os.system(f"bedtools intersect -a {self.file_peaks_sort} -b {file_po} -wao "
                  f"> {file_po_peaks}")
        list_dict = []
        with open(file_po_peaks, 'r') as r_po:
            for line in r_po:
                list_line = line.strip().split('\t')
                peak = list_line[3]
                gene = list_line[8]
                if peak in self.other_peaks and gene in self.all_genes:
                    list_dict.append({"region1": gene, "region2": peak})
        df_po = pd.DataFrame(list_dict)
        df_interaction = pd.concat([df_pp, df_po])
        self.df_graph = df_interaction.drop_duplicates()

        return

    def generate_data_list(self):
        graph_data = self.df_graph
        adata_atac = self.adata
        adata_merge = self.adata_merge
        all_peaks = set(graph_data['region1']).union(set(graph_data['region2']))
        adata_merge_peak = adata_merge[:, [one_peak for one_peak in adata_merge.var.index
                                           if one_peak in all_peaks]]
        array_peak = np.array(adata_merge_peak.var.index)
        array_celltype = np.unique(np.array(adata_atac.obs['celltype']))
        array_region1 = graph_data['region1'].apply(lambda x: np.argwhere(array_peak == x)[0, 0])
        array_region2 = graph_data['region2'].apply(lambda x: np.argwhere(array_peak == x)[0, 0])
        df_graph_index = torch.tensor([np.array(array_region1), np.array(array_region2)],
                                      dtype=torch.int64)
        df_merge_peak = adata_merge_peak.to_df()
        list_graph = []
        for i in range(0, adata_atac.n_obs):
            cell = adata_atac.obs.index[i]
            label = adata_atac.obs.loc[cell, 'celltype']
            label_idx = torch.tensor(np.argwhere(array_celltype == label)[0], dtype=torch.int64)
            cell_data = Data(x=torch.reshape(torch.Tensor(df_merge_peak.loc[cell, :]),
                                             (adata_merge_peak.shape[1], 1)),
                             edge_index=df_graph_index, y=label_idx, cell=cell)
            list_graph.append(cell_data)

        self.list_graph = list_graph
        self.array_peak = array_peak
        self.array_celltype = array_celltype

        return


class ATACGraphDataset(InMemoryDataset):
    def __init__(self, root, data_list=None):
        self.data_list = data_list
        super(ATACGraphDataset, self).__init__(root)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['some_file_1']

    @property
    def processed_file_names(self):
        return ['data.pt']

    def download(self):
        pass

    def process(self):
        # Read data into huge `Data` list.
        data_list = self.data_list
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


class GCN(torch.nn.Module):
    def __init__(self, input_channels, output_channels, hidden_channels, num_nodes):
        super(GCN, self).__init__()
        torch.manual_seed(12345)

        self.conv1 = GraphConv(input_channels, hidden_channels)
        # self.pool1 = TopKPooling(hidden_channels, ratio=0.5)
        self.conv2 = GraphConv(hidden_channels, hidden_channels)

        # num_nodes = num_nodes//int(1/0.5)
        self.lin1 = nn.Linear(num_nodes, num_nodes//5)
        self.lin2 = nn.Linear(num_nodes//5, num_nodes//25)
        self.lin3 = nn.Linear(num_nodes//25, num_nodes//125)
        self.lin4 = nn.Linear(num_nodes//125, num_nodes//625)
        self.lin5 = nn.Linear(num_nodes//625, output_channels)

    def forward(self, x, edge_index, batch, edge_weight=None):
        x = self.conv1(x, edge_index, edge_weight).relu()
        # x, edge_index, edge_weight, batch, _, _ = self.pool1(x, edge_index, edge_weight, batch)
        x = self.conv2(x, edge_index, edge_weight)
        batch_size = len(torch.unique(batch))
        x = torch.mean(x, dim=1, keepdim=True)
        x = x.view(batch_size, x.shape[0]//batch_size)
        x = self.lin1(x).relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x).relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin3(x).relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin4(x).relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin5(x)
        return F.log_softmax(x, dim=1)


def train(loader):
    model.train()
    for data in loader:  # Iterate in batches over the training dataset.
        data = data.to(device)
        out = model(data.x, data.edge_index, data.batch)  # Perform a single forward pass.
        loss = criterion(out, data.y)  # Compute the loss.
        loss.backward()  # Derive gradients.
        optimizer.step()  # Update parameters based on gradients.
        optimizer.zero_grad()  # Clear gradients.


def test(loader):
    model.eval()
    correct = 0
    for data in loader:  # Iterate in batches over the training/test dataset.c
        data = data.to(device)
        out = model(data.x, data.edge_index, data.batch)
        pred = out.argmax(dim=1)  # Use the class with highest probability.
        correct += int((pred == data.y).sum())  # Check against ground-truth labels.
    return correct / len(loader.dataset)  # Derive ratio of correct predictions.


def model_forward(edge_mask, data, model):
    out = model(data.x, data.edge_index, data.batch, edge_mask)
    return out


def merge_per_cluster(adata_Control_raw, cell_merge, cutoff_prop, one_cluster):
    cells_cluster = \
        adata_Control_raw.obs.loc[adata_Control_raw.obs['leiden'] == one_cluster, :].index
    adata_cluster = adata_Control_raw[cells_cluster, :]
    array_cluster = adata_cluster.X.toarray()
    peaks_cluster = adata_cluster.var.index
    if adata_cluster.obs.shape[0] <= 15:
        df_cluster = pd.DataFrame(
            array_cluster, index=adata_cluster.obs.index, columns=peaks_cluster
        )
        df_table = adata_cluster.obs.value_counts('celltype')
        prop = np.max(df_table) / np.sum(df_table)
        if prop > cutoff_prop:
            df_out = np.sum(df_cluster, axis=0)
            df_out.name = one_cluster + '-' + '0'
            df_out = df_out.to_frame().T
            df_out_label = pd.Series([df_table.index[np.argmax(np.array(df_table))]],
                                     index=[one_cluster + '-' + '0'])

    else:
        sc.pp.normalize_total(adata_cluster)
        sc.pp.log1p(adata_cluster)
        sc.pp.highly_variable_genes(adata_cluster, n_top_genes=10000, flavor='seurat')
        adata_cluster = adata_cluster[:, adata_cluster.var.highly_variable]
        sc.pp.scale(adata_cluster, max_value=10)
        num_pcs = min(50, adata_cluster.obs.shape[0]//2)
        sc.tl.pca(adata_cluster, svd_solver='arpack', n_comps=num_pcs)
        sc.pp.neighbors(adata_cluster, n_neighbors=10, n_pcs=num_pcs, metric='cosine')
        if adata_cluster.obs.shape[0] <= 50:
            diff = 3
        elif adata_cluster.obs.shape[0] <= 100:
            diff = 2
        else:
            diff = 1
        resol = 3
        for i in range(10):
            sc.tl.leiden(adata_cluster, resolution=resol)
            cell_med = np.median(adata_cluster.obs.value_counts('leiden'))
            if cell_med - cell_merge >= diff:
                resol = resol + max(2*(cell_med - cell_merge)/cell_merge, 1)
            elif cell_med - cell_merge <= -diff:
                if resol > 1:
                    resol = resol + (cell_med - cell_merge)/cell_merge
                else:
                    resol = resol/2
            else:
                break
        df_cluster = pd.DataFrame(
            array_cluster,
            index=pd.MultiIndex.from_arrays([adata_cluster.obs.index, adata_cluster.obs['leiden']],
                                            names=['index', 'clust']),
            columns=peaks_cluster
        )
        df_table = pd.crosstab(adata_cluster.obs['celltype'], adata_cluster.obs['leiden'])
        prop = np.max(df_table) / np.sum(df_table)
        prop.index = [one_cluster + '-' + idx for idx in df_table.columns]
        labels = df_table.index[np.argmax(np.array(df_table), axis=0)]
        df_out_label = pd.Series(labels,
                                 index=[one_cluster + '-' + idx for idx in df_table.columns])
        df_out_label = df_out_label.loc[prop > cutoff_prop]
        df_out = df_cluster.groupby('clust').apply(lambda x: x.sum())
        df_out.index = [one_cluster + '-' + idx for idx in df_out.index]
        df_out = df_out.loc[df_out_label.index, :]

    return df_out, df_out_label


def merge_cell(dataset_AD_Control, cells_control):
    # merge cells
    cells_control = set(cells_control).intersection(set(dataset_AD_Control.adata.obs.index))
    adata_Control = copy.deepcopy(dataset_AD_Control.adata)
    adata_Control = adata_Control[list(cells_control), :]
    sc.pp.normalize_total(adata_Control)
    sc.pp.log1p(adata_Control)
    sc.pp.highly_variable_genes(adata_Control, n_top_genes=80000, flavor='seurat')
    adata_Control = adata_Control[:, adata_Control.var.highly_variable]
    sc.pp.scale(adata_Control, max_value=10)
    sc.tl.pca(adata_Control, svd_solver='arpack', n_comps=50)
    sc.pp.neighbors(adata_Control, n_neighbors=15, n_pcs=50)
    sc.tl.leiden(adata_Control, resolution=3)
    # sc.tl.umap(adata_Control, min_dist=0.2)
    # sc.pl.umap(adata_Control, color=['celltype', 'Diagnosis', 'Cell.Type', 'leiden'])

    adata_Control_raw = copy.deepcopy(dataset_AD_Control.adata)
    adata_Control_raw = adata_Control_raw[list(cells_control), :]
    adata_Control_raw.obs = adata_Control.obs

    cell_merge = 10
    cutoff_prop = 1/5
    all_cluster = adata_Control_raw.obs['leiden'].unique()
    func_merge = partial(merge_per_cluster, adata_Control_raw, cell_merge, cutoff_prop)
    list_mat = []
    list_label = []
    for one_cluster in list(all_cluster):
        list_out = func_merge(one_cluster)
        list_mat.append(list_out[0])
        list_label.append(list_out[1])
    df_merge = pd.concat(list_mat)
    df_label = pd.concat(list_label)

    adata_merge_Control = \
        ad.AnnData(X=df_merge,
                   var=pd.DataFrame(data={'peaks': df_merge.columns},
                                    index=df_merge.columns),
                   obs=pd.DataFrame(data={'celltype': df_label},
                                    index=df_merge.index))

    return adata_merge_Control


if __name__ == '__main__':
    time_start = time()
    path_AD_Control = '/root/scATAC/ATAC_data/Alzheimer_Morabito/AD_Control_allHiC'
    dataset_AD_Control = ATACDataset(data_root=path_AD_Control, raw_filename='AD_Control.h5ad')
    file_meta_tsv = os.path.join(path_AD_Control, 'metadata.csv')
    df_meta = pd.read_csv(file_meta_tsv)
    df_meta.index = df_meta['Barcode']
    df_meta['celltype'] = df_meta.apply(lambda x: f"{x['Diagnosis']}_{x['Cell.Type']}", axis=1)
    cells_overlap = set(df_meta.index).intersection(set(dataset_AD_Control.adata.obs.index))
    df_meta = df_meta.loc[cells_overlap, :]
    random.seed(12345)
    df_meta_control = df_meta.loc[df_meta['Diagnosis'] == 'Control', :]
    cells_control = random.sample(list(df_meta_control.index), 50000)
    df_meta_AD = df_meta.loc[df_meta['Diagnosis'] == 'AD', :]
    cells_AD = random.sample(list(df_meta_AD.index), 50000)
    df_meta_sample = pd.concat([df_meta_control.loc[cells_control, :],
                                df_meta_AD.loc[cells_AD, :]], axis=0)
    cells_sample = df_meta_sample.index
    dataset_AD_Control.adata = dataset_AD_Control.adata[cells_sample, :]
    dataset_AD_Control.adata.obs = pd.concat([df_meta_sample, dataset_AD_Control.adata.obs], axis=1)

    dataset_AD_Control.quality_control(min_features=300, max_features=5000, min_cells=20)

    adata_merge_Control = merge_cell(dataset_AD_Control, cells_control)
    adata_merge_AD = merge_cell(dataset_AD_Control, cells_AD)
    adata_merge_Control.obs.index = ['Control_' + one for one in adata_merge_Control.obs.index]
    adata_merge_AD.obs.index = ['AD_' + one for one in adata_merge_AD.obs.index]
    adata_merge_Control_AD = ad.concat([adata_merge_Control, adata_merge_AD])

    adata_merge_Control_AD.write_h5ad(os.path.join(path_AD_Control, 'Control_AD_merge_10.h5ad'))
    dataset_merge = ATACDataset(
        data_root=path_AD_Control, raw_filename='Control_AD_merge_10.h5ad')
    dataset_merge.quality_control(min_features=3000, max_features=50000, min_cells=5)
    dataset_merge.select_genes(num_peak=120000)

    file_gene_hg38 = '/root/scATAC/Gene_anno/Gene_hg38/promoters.up2k.protein.gencode.v38.bed'
    dataset_merge.add_promoter(file_gene_hg38)

    # PLAC
    path_hic = '/root/scATAC/pcHi-C/three_brain'
    dataset_merge.build_graph(path_hic)
    df_graph_PLAC = dataset_merge.df_graph

    dataset_merge.generate_data_list()
    list_graph_data = dataset_merge.list_graph
    path_graph_input = os.path.join(path_AD_Control, 'input_graph_merge_10')
    os.system(f"rm -rf {path_graph_input}")
    os.mkdir(path_graph_input)
    dataset_atac_graph = ATACGraphDataset(path_graph_input, list_graph_data)

    sc.pp.scale(adata_merge_Control_AD, max_value=10)
    sc.tl.pca(adata_merge_Control_AD, svd_solver='arpack', n_comps=50)
    sc.pp.neighbors(adata_merge_Control_AD, n_neighbors=15, n_pcs=50)
    sc.tl.umap(adata_merge_Control_AD, min_dist=0.2)
    sc.pl.umap(adata_merge_Control_AD, color=['celltype'])
    # dataset_AD_Control.find_neighbors()
    # dataset_AD_Control.plot_umap()
    time_end = time()
    print(time_end - time_start)

    # save data
    file_atac_AD_Control = os.path.join(path_AD_Control, 'dataset_atac.pkl')
    with open(file_atac_AD_Control, 'wb') as w_pkl:
        str_pkl = pickle.dumps(dataset_merge)
        w_pkl.write(str_pkl)

    # read data
    path_AD_Control = '/root/scATAC/ATAC_data/Alzheimer_Morabito/AD_Control_allHiC'
    file_atac_AD_Control = os.path.join(path_AD_Control, 'dataset_atac.pkl')
    with open(file_atac_AD_Control, 'rb') as r_pkl:
        dataset_merge = pickle.loads(r_pkl.read())

    path_AD_Control = '/root/scATAC/ATAC_data/Alzheimer_Morabito/AD_Control_allHiC'
    path_graph_input = os.path.join(path_AD_Control, 'input_graph_merge_10')
    dataset_atac_graph = ATACGraphDataset(path_graph_input)
    torch.manual_seed(12345)
    dataset = dataset_atac_graph.shuffle()
    train_dataset = dataset[:7000]
    test_dataset = dataset[7000:]

    device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
    model = GCN(input_channels=dataset.num_node_features,
                output_channels=dataset.num_classes, hidden_channels=8,
                num_nodes=dataset_atac_graph[0].num_nodes).to(device)
    # weight
    # celltypes = dataset_merge.array_celltype
    list_weights = []
    for i in range(14):
        sub_dataset = [data for data in train_dataset if data.y == i]
        sub_len = len(sub_dataset)
        sub_weight = len(train_dataset)/sub_len
        list_weights.append(sub_weight)

    criterion = torch.nn.CrossEntropyLoss(weight=torch.tensor(list_weights).to(device))

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # train model
    time_start = time()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
    for epoch in range(1, 60):
        train(train_loader)
        train_acc = test(train_loader)
        test_acc = test(test_loader)
        print(f'Epoch: {epoch:03d}, '
              f'Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}')
        if test_acc > 0.97:
            break
    time_end = time()
    print(time_end - time_start)

    # crosstab
    list_pred = []
    list_true = []
    for data in test_loader:
        data = data.to(device)
        out = model(data.x, data.edge_index, data.batch)
        pred = out.argmax(dim=1)
        list_pred.extend(list(pred.cpu().detach().numpy()))
        list_true.extend(list(data.y.cpu().detach().numpy()))

    # celltypes = dataset_merge.array_celltype
    # label_pred = [celltypes[i] for i in list_pred]
    df_label = pd.DataFrame({'pred': list_pred, 'true': list_true})
    df_table = pd.crosstab(df_label['pred'], df_label['true'])
    df_table.loc[13, :] = np.zeros(14)
    df_score = pd.DataFrame({'precision': np.zeros(7), 'recall': np.zeros(7),
                             'f1': np.zeros(7), 'acc': np.zeros(7)})
    for i in range(7):
        tp = df_table.iloc[i, i]
        fp = df_table.iloc[i, i+7]
        tn = df_table.iloc[i+7, i+7]
        fn = df_table.iloc[i+7, i]
        df_score.iloc[i, 0] = tp/(tp+fp)
        df_score.iloc[i, 1] = tp/(tp+fn)
        df_score.iloc[i, 2] = \
            2*df_score.iloc[i, 0]*df_score.iloc[i, 1]/(df_score.iloc[i, 0] + df_score.iloc[i, 1])
        df_score.iloc[i, 3] = (tp + tn) / (tp + tn + fp + fn)

    file_test_scores = os.path.join(path_AD_Control, 'test_score.txt')
    df_score.to_csv(file_test_scores, sep='\t', index=False)

    # save model
    file_atac_model = os.path.join(path_AD_Control, 'model_atac.pkl')
    with open(file_atac_model, 'wb') as w_pkl:
        str_pkl = pickle.dumps(model)
        w_pkl.write(str_pkl)

    # read model
    path_AD_Control = '/root/scATAC/ATAC_data/Alzheimer_Morabito/AD_Control_allHiC'
    file_atac_model = os.path.join(path_AD_Control, 'model_atac.pkl')
    with open(file_atac_model, 'rb') as r_pkl:
        model = pickle.loads(r_pkl.read())

    # # train separately
    # list_acc = []
    # for i in range(7):
    #     sub_dataset = [data for data in dataset if data.y == i or data.y == i+7]
    #     sub_len = len(sub_dataset)
    #     sub_train = sub_dataset[:sub_len//5*4]
    #     sub_test = sub_dataset[sub_len//5*4:]
    #     model = GCN(input_channels=dataset.num_node_features,
    #                 output_channels=dataset.num_classes, hidden_channels=8,
    #                 num_nodes=dataset_atac_graph[0].num_nodes).to(device)
    #     criterion = torch.nn.CrossEntropyLoss()
    #
    #     train_loader = DataLoader(sub_train, batch_size=32, shuffle=True)
    #     test_loader = DataLoader(sub_test, batch_size=32, shuffle=False)
    #     optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
    #     for epoch in range(1, 50):
    #         train(train_loader)
    #         train_acc = test(train_loader)
    #         test_acc = test(test_loader)
    #         print(f'Epoch: {epoch:03d}, '
    #               f'Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}')
    #         if test_acc > 0.97:
    #             break
    #     list_acc.append(test_acc)

    # explain model
    all_loader = DataLoader(dataset_atac_graph, batch_size=32, shuffle=True)
    list_dict = []
    method = 'ig'
    # i = 0
    for data in train_loader:
        data = data.to(device)
        target = data.y
        input_mask = torch.ones(data.edge_index.shape[1]).requires_grad_(True).to(device)
        # dl = attr.DeepLift(model_forward)
        # mask = dl.attribute(input_mask, target=target,
        #                     additional_forward_args=(data, model))
        ig = IntegratedGradients(model_forward, multiply_by_inputs=False)
        mask = ig.attribute(
            input_mask, target=target, n_steps=50,
            additional_forward_args=(data, model),
            internal_batch_size=data.edge_index.shape[1])
        batch_size = len(torch.unique(data.batch))
        num_col = mask.shape[0]//batch_size
        mask = mask.view(batch_size, num_col)
        # edge_mask = mask.cpu().detach().numpy()
        edge_mask = np.abs(mask.cpu().detach().numpy())
        # edge_mask = preprocessing.robust_scale(edge_mask, axis=1)
        edge_mask = edge_mask / np.max(edge_mask, axis=1)[:, np.newaxis]
        sub_edge_index = data.edge_index.cpu().numpy()
        col_edge = [(sub_edge_index[0, i], sub_edge_index[1, i]) for i in range(num_col)]
        list_dict.append(pd.DataFrame(edge_mask, columns=col_edge, index=data.cell))
        # i = i + 1
        # if i >= 2:
        #     break
    df_weight = pd.concat(list_dict)
    df_weight_0 = df_weight.copy()
    df_weight_0[df_weight_0 < 0.05] = 0
    # df_weight = df_weight.dropna()
    # df_weight.index = dataset_ATAC.adata.obs.index

    # save weight
    file_weight = os.path.join(path_AD_Control, 'weight_atac.pkl')
    with open(file_weight, 'wb') as w_pkl:
        str_pkl = pickle.dumps(df_weight)
        w_pkl.write(str_pkl)

    # read weight
    path_AD_Control = '/root/scATAC/ATAC_data/Alzheimer_Morabito/AD_Control_allHiC'
    file_weight = os.path.join(path_AD_Control, 'weight_atac.pkl')
    with open(file_weight, 'rb') as r_pkl:
        df_weight = pickle.loads(r_pkl.read())

    adata_edge = ad.AnnData(X=df_weight_0, obs=dataset_merge.adata.obs.loc[df_weight_0.index, :])
    adata_edge.raw = adata_edge.copy()
    # sc.pp.normalize_total(adata_edge)
    # sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata_edge, n_top_genes=10000, flavor='seurat')
    adata = adata_edge[:, adata_edge.var.highly_variable]
    sc.pp.scale(adata, max_value=10)
    sc.tl.pca(adata, svd_solver='arpack', n_comps=100)
    sc.pp.neighbors(adata, n_neighbors=15, n_pcs=50)
    sc.tl.umap(adata, min_dist=0.2)
    sc.pl.umap(adata, color=['nb_features', 'celltype'])

    # rank edge
    adata_edge.obs['Diagnosis'] = adata_edge.obs['celltype'].apply(lambda x: x.split('_')[0])
    adata_edge.obs['CellType'] = adata_edge.obs['celltype'].apply(lambda x: x.split('_')[1])
    peaks = dataset_merge.array_peak
    celltypes = adata_edge.obs['CellType'].unique()
    path_diff = os.path.join(path_AD_Control, 'diff_interactome')
    dict_diff = {}
    for celltype in celltypes:
        sc.tl.rank_genes_groups(adata_edge, groupby='celltype',
                                groups=[f'AD_{celltype}'], reference=f'Control_{celltype}',
                                method='wilcoxon',
                                use_raw=True, pts=True, tie_correct=True)
        array_names = [one[0] for one in adata_edge.uns['rank_genes_groups']['names']]
        array_scores = [one[0] for one in adata_edge.uns['rank_genes_groups']['scores']]
        array_fcs = [one[0] for one in adata_edge.uns['rank_genes_groups']['logfoldchanges']]
        array_pval = [one[0] for one in adata_edge.uns['rank_genes_groups']['pvals_adj']]
        # array_pval = adata_edge.uns['rank_genes_groups']['pvals_adj']
        df_MG = pd.DataFrame(
            {"edge_name": [(peaks[edge_name[0]], peaks[edge_name[1]]) for edge_name in array_names],
             "edge_score": array_scores, "edge_fc": array_fcs, "edge_pval": array_pval,
             "gene": [peaks[edge_name[0]] for edge_name in array_names],
             "cRE": [peaks[edge_name[1]] for edge_name in array_names]},
        )
        df_MG.to_csv(os.path.join(path_diff, f"{celltype}.txt"), sep='\t', index=False)
        dict_diff[celltype] = df_MG
    all_pairs = df_MG['edge_name']

    # View
    adata_merge_Control_AD = dataset_merge.adata_merge
    adata_merge_Control_AD.obs = dataset_merge.adata.obs
    adata_merge_Control_AD.obs['Diagnosis'] = \
        adata_merge_Control_AD.obs['celltype'].apply(lambda x: x.split('_')[0])
    adata_merge_Control_AD.obs['CellType'] = \
        adata_merge_Control_AD.obs['celltype'].apply(lambda x: x.split('_')[1])
    cell = 'MG'
    cells = adata_merge_Control_AD.obs.loc[adata_merge_Control_AD.obs['CellType'] == cell, :].index
    sub_adata = adata_merge_Control_AD[cells, :]
    one_peak = np.sum(sub_adata[:, 'SH2D4B'].X != 0)

    adata_edge.var['idx'] = adata_edge.var.index
    adata_edge.var.index = [(peaks[edge_name[0]], peaks[edge_name[1]])
                            for edge_name in adata_edge.var['idx']]
    sub_adata_edge = adata_edge[list(set(cells).intersection(set(adata_edge.obs.index))), :]
    one_pair = tuple(['SHANK2', 'chr11:71114392-71114922'])
    np.sum(sub_adata_edge[:, [one_pair]].X != 0)

    # dict promoter
    dict_promoter = {}
    file_peak_promoter = \
        '/root/scATAC/ATAC_data/Alzheimer_Morabito/AD_Control_allHiC/processed_files/peaks_promoter.txt'
    with open(file_peak_promoter, 'r') as w_pro:
        for line in w_pro:
            list_line = line.strip().split('\t')
            peak = list_line[3]
            gene = list_line[7].split('<-')[0]
            if gene != '.':
                dict_promoter[peak] = gene
    file_peaks = dataset_merge.file_peaks_sort

    # AD sites
    path_AD = '/root/scATAC/GWAS/Jansen_NG_2019/'
    path_AD_process = os.path.join(path_AD, 'process')
    file_AD_hg38 = os.path.join(path_AD_process, 'hg38.bed')
    path_ATAC_AD = os.path.join(path_AD_Control, 'AD_GWAS')
    file_intersect = os.path.join(path_ATAC_AD, 'peaks_AD.txt')
    os.system(f"bedtools intersect -a {file_peaks} -b {file_AD_hg38} -wao > {file_intersect}")
    df_peaks_AD = pd.read_csv(file_intersect, sep='\t', header=None)
    peaks_AD = df_peaks_AD.loc[df_peaks_AD.iloc[:, 4] != '.', 3].tolist()
    # file_AD_scores = os.path.join(path_ATAC_AD, 'peaks_celltypes_scores.txt')
    # df_AD_peak = pd.read_csv(file_AD_scores, sep='\t', index_col=0)
    # AD interactome
    list_AD_merge_peaks = []
    for peak in peaks_AD:
        if peak in dict_promoter.keys():
            list_AD_merge_peaks.append(dict_promoter[peak])
            continue
        list_AD_merge_peaks.append(peak)
    list_interatome = []
    for pair in all_pairs:
        pair_1 = pair[0]
        pair_2 = pair[1]
        if pair_2[0:3] == 'chr':
            if pair_2 in list_AD_merge_peaks:
                list_interatome.append(pair)
        else:
            if (pair_1 in list_AD_merge_peaks) | (pair_2 in list_AD_merge_peaks):
                list_interatome.append(pair)
    df_AD_interactome = pd.DataFrame(np.full(shape=(len(list_interatome), len(celltypes)),
                                             fill_value=0.0),
                                     index=list_interatome, columns=celltypes)
    df_AD_pval = pd.Series(np.full(shape=(len(celltypes)), fill_value=1.0), index=list(celltypes))
    for celltype in celltypes:
        all_score = dict_diff[celltype]
        all_score.index = all_score['edge_name']
        all_score = all_score['edge_score']
        for sub_inter in set(list_interatome):
            df_AD_interactome.loc[df_AD_interactome.index == sub_inter, celltype] = \
                all_score.loc[all_score.index == sub_inter]
        df_AD_pval.loc[celltype] = \
            kstest(np.abs(np.array(df_AD_interactome[celltype])),
                   np.abs(np.array(all_score)), alternative='less')[1]
    file_interactome_AD_scores = os.path.join(path_ATAC_AD, 'interactome_celltype_scores.txt')
    df_AD_interactome.to_csv(file_interactome_AD_scores, sep='\t')
    file_interactome_AD_pvals = os.path.join(path_ATAC_AD, 'interactome_celltype_pvals.txt')
    df_AD_pval.to_csv(file_interactome_AD_pvals, sep='\t')
