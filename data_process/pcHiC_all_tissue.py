# _*_ coding: utf-8 _*_
# @author: Drizzle_Zhang
# @file: pcHiC_all_tissue.py
# @time: 2022/5/10 17:22

import os
import pandas as pd
from collections import defaultdict

# process PO file
path_hic = '/root/scATAC/pcHi-C'
file_po = os.path.join(path_hic, 'PO.txt')
file_po_bed = os.path.join(path_hic, 'PO.bed')
fmt_po = "{chrom}\t{start}\t{end}\t{peak_id}\t{gene}\t{tissue}\n"
with open(file_po_bed, 'w') as w_bed:
    with open(file_po, 'r') as r_po:
        for line in r_po:
            list_line = line.strip().split('\t')
            chrom = list_line[1].strip().split('.')[0]
            start = list_line[1].strip().split('.')[1]
            end = list_line[1].strip().split('.')[2]
            peak_id = f"{chrom}:{start}-{end}"
            gene = list_line[0]
            tissue = list_line[2]
            w_bed.write(fmt_po.format(**locals()))

# hg19 to hg38
file_chain = '/root/tools/files_liftOver/hg19ToHg38.over.chain.gz'
liftover = '/root/tools/liftOver'

path_hic_tissue = '/root/scATAC/pcHi-C/PO_by_tissue'
df_po = pd.read_csv(file_po_bed, sep='\t', header=None)
length_o = df_po.iloc[:, 2] - df_po.iloc[:, 1]
df_po['length'] = length_o
df_po = df_po.loc[df_po['length'] < 20000, :]
all_tissue = set(df_po.iloc[:, 5].tolist())
for one_tissue in all_tissue:
    path_tissue = os.path.join(path_hic_tissue, one_tissue.replace(' ', '_'))
    if not os.path.exists(path_tissue):
        os.mkdir(path_tissue)
    file_hg19 = os.path.join(path_tissue, 'hg19.bed')
    df_tissue = df_po.loc[df_po.iloc[:, 5] == one_tissue, :]
    df_tissue.loc[:, 'interaction_id'] = df_tissue.apply(lambda x: f"{x.iloc[3]}_{x.iloc[4]}", axis=1)
    df_tissue.to_csv(file_hg19, sep='\t', index=None, header=None)
    file_hg38 = os.path.join(path_tissue, 'hg38.bed')
    file_prefix = file_hg19 + '.prefix'
    file_suffix = file_hg19 + '.suffix'
    file_hg38_prefix = file_hg38 + '.prefix'
    file_hg38_format = file_hg38 + '.format'
    file_ummap = os.path.join(path_tissue, 'unmap.bed')
    os.system(f"cut -f 1,2,3,8 {file_hg19} > {file_prefix}")
    os.system(f"cut -f 4,5,6,8 {file_hg19} > {file_suffix}")
    os.system(f"{liftover} {file_prefix} {file_chain} "
              f"{file_hg38_prefix} {file_ummap}")
    dict_peak_score = defaultdict(list)
    with open(file_suffix, 'r') as r_f:
        for line in r_f:
            list_line = line.strip().split('\t')
            dict_peak_score[list_line[3]].append(list_line[0:3])
    with open(file_hg38_format, 'w') as w_f:
        fmt = "{chrom}\t{start}\t{end}\t{interaction_id}\t{peak_id}\t{gene}\t{tissue}\n"
        with open(file_hg38_prefix, 'r') as r_hg38:
            for line in r_hg38:
                list_line = line.strip().split('\t')
                list_suffix = dict_peak_score[list_line[3]][0]
                dict_hg38 = dict(
                    chrom=list_line[0], start=list_line[1], end=list_line[2],
                    interaction_id=list_line[3],
                    peak_id=f"{list_line[0]}:{list_line[1]}-{list_line[2]}",
                    gene=list_suffix[1], tissue=list_suffix[2]
                )
                w_f.write(fmt.format(**dict_hg38))

    df_old = pd.read_csv(file_prefix, sep='\t', header=None)
    length_old = df_old.iloc[:, 2] - df_old.iloc[:, 1]
    df_old['length'] = length_old
    df_bed = pd.read_csv(file_hg38_format, sep='\t', header=None)
    length = df_bed.iloc[:, 2] - df_bed.iloc[:, 1]
    # df_bed['length'] = length
    df_bed = df_bed.loc[length < 20000, :]
    df_bed = df_bed.drop_duplicates()
    df_bed.to_csv(file_hg38, sep='\t', index=None, header=None)

# interactions for each tissue
path_interaction = '/root/scATAC/pcHi-C/Interactions_by_tissue'
file_pp = os.path.join(path_hic, 'PP.txt')
df_pp = pd.read_csv(file_pp, sep='\t', header=None)
all_tissue = set(df_pp.iloc[:, 2].tolist())
for one_tissue in all_tissue:
    path_po_tissue = os.path.join(path_hic_tissue, one_tissue.replace(' ', '_'))
    path_tissue = os.path.join(path_interaction, one_tissue.replace(' ', '_'))
    if not os.path.exists(path_tissue):
        os.mkdir(path_tissue)
    file_PP = os.path.join(path_tissue, 'PP.txt')
    df_pp_tissue = df_pp.loc[df_pp.iloc[:, 2] == one_tissue, [0, 1, 2]]
    with open(file_PP, 'w') as w_pp:
        for sub_dict in df_pp_tissue.to_dict('records'):
            set_gene1 = sub_dict[0].strip().split(';')
            set_gene2 = sub_dict[1].strip().split(';')
            for gene1 in set_gene1:
                for gene2 in set_gene2:
                    w_pp.write(f"{gene1}\t{gene2}\n")

    file_po_tissue = os.path.join(path_po_tissue, 'hg38.bed')
    file_PO = os.path.join(path_tissue, 'PO.txt')
    with open(file_PO, 'w') as w_po:
        with open(file_po_tissue, 'r') as r_po:
            for line in r_po:
                list_line = line.strip().split('\t')
                chrom = list_line[0]
                start = list_line[1]
                end = list_line[2]
                peak_id = list_line[4]
                set_gene = list_line[5].strip().split(';')
                for gene in set_gene:
                    w_po.write(f"{chrom}\t{start}\t{end}\t{peak_id}\t{gene}\n")

# merge all pcHiC hg38
path_all = '/root/scATAC/pcHi-C/all_pcHiC/hg38'
file_all_PP = os.path.join(path_all, 'PP.txt')
file_all_PO = os.path.join(path_all, 'PO.txt')

path_interaction = '/root/scATAC/pcHi-C/Interactions_by_tissue'
all_tissue = os.listdir(path_interaction)
list_pp = []
list_po = []
for one_tissue in all_tissue:
    path_one_tissue = os.path.join(path_interaction, one_tissue)
    one_pp = os.path.join(path_one_tissue, 'PP.txt')
    list_pp.append(one_pp)
    one_po = os.path.join(path_one_tissue, 'PO.txt')
    list_po.append(one_po)

os.system(f"cat {' '.join(list_pp)} | sort | uniq > {file_all_PP}")
os.system(f"cat {' '.join(list_po)} | sort | uniq > {file_all_PO}")
