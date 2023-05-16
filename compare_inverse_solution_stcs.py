#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 12 18:32:20 2023

@author: fm02
"""
sbj_ids = [1,2,3,5,6,8,9,10,11,12,13,14,15,16,17,18,19,
            21,22,23,24,25,26,27,28,29,30]

import os
import re
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

def p2f(x):
   return float(x.strip('%'))/100
   
sums = dict()
sums_shrunk = list()
sums_ledoit = list()

rx = dict.fromkeys(['empirical', 'shrunk', 'old', 'ledoit_wolf',
                       'empirical_drop', 'combined_drop'])

rx['empirical'] = r"slurm_fm02_compare_icas_1901912_4294967294_node-(.*)_stdout_task_"
rx['shrunk'] = r"slurm_fm02_compare_icas_1903682_4294967294_node-(.*)_stdout_task_"
rx['old'] = r"slurm_fm02_coherence_rois_1886010_4294967294_node-(.*)_stdout_task_"
rx['ledoit_wolf'] = r"slurm_fm02_compare_icas_1903687_4294967294_node-(.*)_stdout_task_"
rx['empirical_drop'] = r"slurm_fm02_compare_icas_1903718_4294967294_node-(.*)_stdout_task_"
rx['combined_drop'] = r"slurm_fm02_compare_icas_1903719_4294967294_node-(.*)_stdout_task_"

for key in rx.keys():
    sums[key] = list()
    
for sbj_id in sbj_ids:
    rootdir = "/home/fm02/Desktop/MEG_EOS_scripts/sbatch_out/tasks/"
    for key in rx.keys():
        regex = re.compile(rx[key]+f"{sbj_id}.log")
        
        log = list()
        
        for root, dirs, files in os.walk(rootdir):
          for file in files:
            if regex.match(file):
                log.append(file)
        for l in log:
            with open(os.path.join(rootdir, l)) as f:
                contents = f.read()
                
        variance = re.compile(r"Explained\s+([0-9]+)\.\d% variance")
        
        matches = re.findall(r"Explained\s+?(?:\d*\.*\d+%)", contents) 
        m = [mat.split()[1] for mat in matches]
        if len(m)==8:
            sums_shrunk.append(pd.DataFrame(zip([sbj_id]*4, m[0:4],
                                          ['Predictable', 'Unpredictable', 'Abstract', 'Concrete']
                                          )
                                      )
                        )
            sums_ledoit.append(pd.DataFrame(zip([sbj_id]*4, m[4:],
                                          ['Predictable', 'Unpredictable', 'Abstract', 'Concrete']
                                          )
                                      )
                        )
        else:        
            sums[key].append(pd.DataFrame(zip([sbj_id]*4, m,
                                          ['Predictable', 'Unpredictable', 'Abstract', 'Concrete']
                                          )
                                      )
                        )

sums['ledoit_wolf_drop'] = sums_ledoit
sums['shrunk_drop'] = sums_shrunk

del sums['combined_drop']

for key in sums.keys():
    sums[key] = pd.concat(sums[key])
    sums[key] = sums[key].set_index(0)
    sums[key] = sums[key][1].apply(p2f)
    
avg_across = dict()
for key in sums.keys():
    avg_across[key] = sums[key].groupby(sums[key].index).mean()

avg_across = pd.DataFrame(avg_across)

avg_avg = avg_across.mean()
avg_std = avg_across.std()

f, (a0, a1) = plt.subplots(2,1, gridspec_kw={'height_ratios': [27, 2]}, sharex=True)
sns.heatmap(avg_across, annot=True, ax=a0, vmin=0, vmax=1)
sns.heatmap([avg_avg, avg_std], annot=True, ax=a1, vmin=0, vmax=1, 
            xticklabels=avg_across.columns, yticklabels=['average', 'std'])
# f.subplots_adjust(wspace=0, hspace=0.01)
plt.tight_layout()
plt.show()