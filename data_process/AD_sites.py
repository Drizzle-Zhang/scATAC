# _*_ coding: utf-8 _*_
# @author: Drizzle_Zhang
# @file: AD_sites.py
# @time: 2022/4/16 14:04

import os
import pandas as pd


path_AD = '/root/scATAC/GWAS/AD_GWASCatalog'
file_AD_ori = os.path.join(path_AD, 'GWAS_AD.txt')
path_AD_process = os.path.join(path_AD, 'process')
# os.mkdir(path_AD_process)
file_AD_hg38 = os.path.join(path_AD_process, 'hg38.bed')

df_AD_ori = pd.read_csv(file_AD_ori, sep='\t', encoding='unicode_escape')
df_AD_ori = df_AD_ori.dropna(subset=['CHR_POS'])
df_AD_ori['chrom'] = df_AD_ori['CHR_ID'].apply(lambda x: f"chr{x}")
df_AD_ori['start'] = df_AD_ori['CHR_POS'].apply(lambda x: int(x)-1)
df_AD_ori['end'] = df_AD_ori['CHR_POS'].apply(lambda x: int(x)+1)
df_AD_ori['name'] = df_AD_ori['SNPS']

df_AD_GWASCatalog = df_AD_ori.loc[:, ['chrom', 'start', 'end', 'name']]
df_AD_GWASCatalog.to_csv(file_AD_hg38, sep='\t', header=False, index=False)


path_AD = '/root/scATAC/GWAS/AD_DisGeNET'
file_AD_ori = os.path.join(path_AD, 'DisGeNet_AD.txt')
path_AD_process = os.path.join(path_AD, 'process')
# os.mkdir(path_AD_process)
file_AD_hg38 = os.path.join(path_AD_process, 'hg38.bed')

df_AD_ori = pd.read_csv(file_AD_ori, sep='\t', encoding='unicode_escape')
df_AD_ori['chrom'] = df_AD_ori['chromosome'].apply(lambda x: f"chr{x}")
df_AD_ori['start'] = df_AD_ori['position'].apply(lambda x: int(x)-1)
df_AD_ori['end'] = df_AD_ori['position'].apply(lambda x: int(x)+1)
df_AD_ori['name'] = df_AD_ori['snpId']

df_AD_DisGeNET = df_AD_ori.loc[:, ['chrom', 'start', 'end', 'name']]
df_AD_DisGeNET.to_csv(file_AD_hg38, sep='\t', header=False, index=False)

df_AD = pd.concat([df_AD_GWASCatalog, df_AD_DisGeNET])
df_AD = df_AD.drop_duplicates()
df_AD.to_csv('/root/scATAC/GWAS/AD_both/AD.txt', sep='\t', header=False, index=False)
