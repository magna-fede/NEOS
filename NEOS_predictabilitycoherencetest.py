#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 29 16:41:51 2023

@author: fm02
"""

import sys
import os
from os import path

import numpy as np
import pandas as pd

from scipy.stats import ttest_rel

os.chdir("/home/fm02/MEG_NEOS/NEOS")
import NEOS_config as config

from itertools import product
from scipy.stats import ttest_rel

labels_path = path.join(config.data_path, "my_ROIs")

predictability_factors = ['Predictable', 'Unpredictable']
sbj_ids = [1,2,3,5,6,8,9,10,11,12,13,14,15,16,17,18,19,
           21,22,23,24,25,26,27,28,29,30]

overall = dict()
for condition in predictability_factors:
    overall[condition] = []
    
for sbj_id in sbj_ids:
    sbj_path = path.join(config.data_path, config.map_subjects[sbj_id][0])
    for condition in overall.keys():
        overall[condition].append(read_connectivity(path.join(sbj_path,
                                        f"{sbj_id}_{condition}_ROI_coherence")
                                                    )
                                  )
alpha = dict()
beta = dict()
test_alpha = dict()
test_beta = dict()

for condition in predictability_factors:
    alpha[condition] = []
    beta[condition] = []

for condition in predictability_factors:
    for coh in overall[condition]:
        alpha[condition].append(coh.get_data()[:, 0])
        beta[condition].append(coh.get_data()[:, 1])
    
    test_beta[condition]  = np.stack(beta[condition])
    test_alpha[condition]  = np.stack(alpha[condition])


comb2 = product(['l_ATL', 'r_ATL', 'PTC', 'IFG', 'AG', 'PVA'], repeat=2)
perm_labels = list(comb2)

for condition in predictability_factors:
    test_alpha[condition] = pd.DataFrame(test_alpha[condition], columns=perm_labels)
    test_alpha[condition] = test_alpha[condition].loc[:, (test_alpha[condition] !=0).any(axis=0)]
    test_beta[condition] = pd.DataFrame(test_beta[condition], columns=perm_labels)
    test_beta[condition] = test_beta[condition].loc[:, (test_beta[condition] !=0).any(axis=0)]

for connection in test_alpha['Predictable'].columns:
    print (connection, ttest_rel(test_alpha['Predictable'][connection],
                    test_alpha['Unpredictable'][connection]))
    
for connection in test_alpha['Predictable'].columns:
    print (connection, ttest_rel(test_beta['Predictable'][connection],
                    test_beta['Unpredictable'][connection]))

# ALPHA #
# ('r_ATL', 'l_ATL') Ttest_relResult(statistic=-0.43755216583060363, pvalue=0.6653242076104748)
# ('PTC', 'l_ATL') Ttest_relResult(statistic=1.3370903883109033, pvalue=0.19277475324677928)
# ('PTC', 'r_ATL') Ttest_relResult(statistic=1.6054562594549595, pvalue=0.12047258303264274)
# ('IFG', 'l_ATL') Ttest_relResult(statistic=-0.5165501942078659, pvalue=0.6098361981400708)
# ('IFG', 'r_ATL') Ttest_relResult(statistic=-0.5941898244314077, pvalue=0.5575204941014088)
# *** ('IFG', 'PTC') Ttest_relResult(statistic=-2.710561646916781, pvalue=0.011737314801136727)
# *** ('AG', 'l_ATL') Ttest_relResult(statistic=-2.2958947894258537, pvalue=0.0299947518560882)
# ('AG', 'r_ATL') Ttest_relResult(statistic=1.2518214031734016, pvalue=0.22178022328120894)
# ('AG', 'PTC') Ttest_relResult(statistic=-0.4397575823506853, pvalue=0.6637465040577524)
# ('AG', 'IFG') Ttest_relResult(statistic=0.8532506377343053, pvalue=0.40131293202480767)
# ('PVA', 'l_ATL') Ttest_relResult(statistic=0.7693448798731026, pvalue=0.4486219029311337)
# ('PVA', 'r_ATL') Ttest_relResult(statistic=-1.4798023027305995, pvalue=0.15094198215564802)
# ('PVA', 'PTC') Ttest_relResult(statistic=1.107814843359295, pvalue=0.27808537910905784)
# ('PVA', 'IFG') Ttest_relResult(statistic=-0.7841372781704775, pvalue=0.44004545809498485)
# ('PVA', 'AG') Ttest_relResult(statistic=-0.884255930478794, pvalue=0.3846623356412795)    

# BETA #
# ('r_ATL', 'l_ATL') Ttest_relResult(statistic=0.624088549720668, pvalue=0.5380073197195838)
# ('PTC', 'l_ATL') Ttest_relResult(statistic=-0.11417497977178519, pvalue=0.9099762101657553)
# ('PTC', 'r_ATL') Ttest_relResult(statistic=-0.4015475442838392, pvalue=0.691298570190761)
# ('IFG', 'l_ATL') Ttest_relResult(statistic=0.9857069542926509, pvalue=0.33336162041929196)
# ('IFG', 'r_ATL') Ttest_relResult(statistic=-1.0099151105026924, pvalue=0.3218428345285419)
# ('IFG', 'PTC') Ttest_relResult(statistic=1.0272381570745082, pvalue=0.313770810783089)
# *** ('AG', 'l_ATL') Ttest_relResult(statistic=3.10984493524436, pvalue=0.00450050658296309)
# ('AG', 'r_ATL') Ttest_relResult(statistic=0.3295146972563889, pvalue=0.7444066613446393)
# ('AG', 'PTC') Ttest_relResult(statistic=0.1596397121011563, pvalue=0.8743986083205668)
# ('AG', 'IFG') Ttest_relResult(statistic=-0.13254573993474958, pvalue=0.8955731263942222)
# ('PVA', 'l_ATL') Ttest_relResult(statistic=-0.33641386100079856, pvalue=0.7392589521908707)
# ('PVA', 'r_ATL') Ttest_relResult(statistic=0.137981844578408, pvalue=0.8913178468137373)
# ('PVA', 'PTC') Ttest_relResult(statistic=1.4638572021871263, pvalue=0.15521947541577943)
# ('PVA', 'IFG') Ttest_relResult(statistic=0.6355685709899471, pvalue=0.5306124564196532)
# ('PVA', 'AG') Ttest_relResult(statistic=-1.1942356461886625, pvalue=0.24316766717488322)
