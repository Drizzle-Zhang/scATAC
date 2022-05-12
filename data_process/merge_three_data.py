# _*_ coding: utf-8 _*_
# @author: Drizzle_Zhang
# @file: merge_three_data.py
# @time: 2022/4/5 17:50

import os


path_pchic = '/root/scATAC/pcHi-C/Interactions_by_tissue/Dorsolateral_Prefrontal_Cortex'
path_plac = '/root/scATAC/pcHi-C/Cortex_PLACSeq/hg38'
path_psych = '/root/scATAC/pcHi-C/psychHiC/hg38'
path_three = '/root/scATAC/pcHi-C/three_brain'

path_three_PP = os.path.join(path_three, 'PP.txt')
path_three_PO = os.path.join(path_three, 'PO.txt')

os.system(f"cat {os.path.join(path_pchic, 'PP.txt')} {os.path.join(path_plac, 'PP.txt')} "
          f"{os.path.join(path_psych, 'PP.txt')} | sort | uniq > {path_three_PP}")
os.system(f"cat {os.path.join(path_pchic, 'PO.txt')} {os.path.join(path_plac, 'PO.txt')} "
          f"{os.path.join(path_psych, 'PO.txt')} | sort | uniq > {path_three_PO}")
