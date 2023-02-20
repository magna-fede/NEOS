# -*- coding: utf-8 -*-
"""
Created on Wed Jul 20 16:34:24 2022

@author: fm02
"""


import os
import numpy as np
import pandas as pd
import re
import pickle

from pygazeanalyser.edfreader import read_edf


DISPSIZE = (1920, 1080)

def get_blinks(data_edf):
    blinks=[]
    for i,trial in enumerate(data_edf):
        blinks.append(data_edf[i]['events']['Eblk']) # get all blinks
        blinks = [x for x in blinks if x != []]
    blinks = [item for sublist in blinks for item in sublist]    
    return blinks 

def read_edf_plain(filename):
    """Get a dataframe containing only and all the events from the EDF file,
        with the trackertime, not dividing the trials"""
    # check if the file exists
    if os.path.isfile(filename):
		# open file
        f = open(filename, 'r')
	# raise exception if the file does not exist
    else:
        raise Exception("Error in read_edf: file '%s' does not exist" % filename)
    raw = f.readlines()
    f.close()
	# variables
    data = []
    event = []
    timepoint = []
	# loop through all lines
    for line in raw:
        if line[0:4] == "SFIX":
            l = line[9:]
            timepoint.append(int(l))
            event.append(line[0:4])
        elif line[0:4] == "EFIX":
            l = line[9:]
            l = l.split('\t')
            timepoint.append(int(l[1]))
            event.append(line[0:4])
         			# saccade start
        elif line[0:5] == 'SSACC':
            l = line[9:]
            timepoint.append(int(l))
            event.append(line[0:5])
         			# saccade end
        elif line[0:5] == "ESACC":
            l = line[9:]
            l = l.split('\t')
            timepoint.append(int(l[1]))
            event.append(line[0:5])
         			# blink start
        elif line[0:6] == "SBLINK":
            l = line[9:]
            timepoint.append(int(l))
            event.append(line[0:6])
         			# blink end
        elif line[0:6] == "EBLINK":
            l = line[9:]
            l = l.split('\t')
            timepoint.append(int(l[1]))
            event.append(line[0:6])
   	# return
    data = pd.DataFrame()
    data['time'] = np.array(timepoint)
    data['event'] = np.array(event)
    
    return data


def fixAOI(data_edf,data_plain):
    """Get all fixations within AOI. Checks that are not followed by a regression
    after the first fixation within the AOI + trials that do not contain
    a blink or error"""
    # get all fixation durations within a certain AOI for all trials for one subject
    # dur_all is the list where we include all the fixation durations that
    # respect certain inclusion criteria
    # 
    data_trial = pd.DataFrame(columns=['IDStim',
                                       'FFD',
                                       'time',
                                       'fixated'])
            
    for i,trial in enumerate(data_edf):
        ID_stim = trial['events']['msg'][2][1][-4:-1]
        stimonset = trial['events']['msg'][0][0]
        pd_fix = pd.DataFrame.from_records(trial['events']['Efix'],
                                           columns=['start',
                                                    'end',
                                                    'duration',
                                                    'x',
                                                    'y'])
        # exclude those trials where all fixations are outside the screen
        # it used to happe if there was an error in the gaze position detection
        # it should not be a problem now, considering that gaze is required
        # to trigger the start of the sentence
        if (((pd_fix['x'] < 0).all()) or ((pd_fix['x'] > DISPSIZE[0]).all())):
            pass
        elif (((pd_fix['y'] < 0).all()) or ((pd_fix['y'] > DISPSIZE[1]).all())):
            pass
        # or when no fixations have been detected
        elif len(pd_fix)<2:
            pass
        else:
            # consider only fixations following a the first leftmost fixation

            # !! this is now useless and potentially problematic considering that
            # participants fixate on the left side of the screen (and not the centre)
            # before the appearance of the sentence
            
            # while pd_fix['x'][0]>pd_fix['x'][1]:
            #     pd_fix.drop([0],inplace=True)
            #     pd_fix.reset_index(drop=True, inplace=True)
            
            # the following info is gathered from the
            # stimulus presentation software (communicated the following msgs)
            
            # tuple indicating dimension of each sentence in pixels
            size = re.search("SIZE OF THE STIMULUS: (.*)\n",trial['events']['msg'][4][1])
            size = eval(size.group(1)) # tuple (width,height)
            
            # size of each letter in pixels
            # this should is identical for each sentence, equal to 19 in our study
            unit = re.search("NUMBER OF CHARACTERS: (.*)\n",trial['events']['msg'][5][1])
            unit = size[0]/eval(unit.group(1)) 
            
            # position (in characters) of the target word inside the sentence
            pos_target = re.search("POS TARGET INSIDE BOX: (.*)\n",trial['events']['msg'][6][1])
            pos_target = eval(pos_target.group(1))
            
            # position (in pixels) of the target word
            # convert width to the position in x, y cohordinates where the sentence starts
            # stimulus starting position is = centre of x_axis screen - half size of the sentence
            # because sentence is presented aligned to the centre of the screen
            pos_startstim = DISPSIZE[0]/2-size[0]/2
            # no need to calculate y as always in the same position at the centre
            # only one line
            
            # get x and y position of the target word
            # as pos_target is in characters, we need to mutiply each letter*unit
            # including in the AOI also half space preceding and half space
            # following the target word
            # tuple (x0,x1) position of the target word in pixels 
            target_x = (pos_startstim+(pos_target[0]*unit)-unit/2,pos_startstim+(pos_target[1]*unit)+unit/2)
            target_y = (DISPSIZE[1]/2-size[1]*2,DISPSIZE[1]/2+size[1]*2)
            # AOI for target_y position is two times the height of the letters
            # no need to be too strict as there's just one line
            
            # get all fixations on target word
            # this is checks if targetstart_position<fixation_position<targetend_position
            fixAOI_x = pd_fix['x'][(target_x[0]<pd_fix['x']) &
                                                  (pd_fix['x']<target_x[1]) &
                                                  (target_y[0]<pd_fix['y']) &
                                                  (pd_fix['y']<target_y[1])]
            fixAOI_duration = pd_fix['duration'][(target_x[0]<pd_fix['x']) &
                                                  (pd_fix['x']<target_x[1]) &
                                                  (target_y[0]<pd_fix['y']) &
                                                  (pd_fix['y']<target_y[1])]
            
            fixAOI_timein = pd_fix['start'][(target_x[0]<pd_fix['x']) &
                                                  (pd_fix['x']<target_x[1]) &
                                                  (target_y[0]<pd_fix['y']) &
                                                  (pd_fix['y']<target_y[1])] \
                            - stimonset
            # 60ms refers to the pilot, need to check any systematic delay
            
            # check if at least one fixation on target 
            if len(fixAOI_duration)>0:
                
                # check this is first pass
                # by checking if all previous fixations (indetify by index) have a smaller x_position
                if all(pd_fix['x'][0:fixAOI_x.index[0]]<fixAOI_x[fixAOI_x.index[0]]):
                    start = pd_fix['start'][(target_x[0]<pd_fix['x']) &
                                                 (pd_fix['x']<target_x[1]) &
                                                 (target_y[0]<pd_fix['y']) &
                                                 (pd_fix['y']<target_y[1])].iloc[0]
                    # get the  position in the events only list of data
                    plain_start = data_plain[data_plain['time']==start].index[0]
                    r = range(plain_start-2,plain_start+4)
                    # this range because each blink generates an artefactual saccade
                    # to each blink id surrounded by SSACC and ESAC events
                    # different ends to include also EFIX event
                    # this basically checks whether the fixation is immediately
                    # preceded or followed by a blink
                    if not (any(data_plain['event'].iloc[r]=='SBLINK')
                        or any(data_plain['event'].iloc[r]=='EBLINK')):
                        
                        data_trial.loc[len(data_trial)] = [ID_stim,
                                                           fixAOI_duration.iloc[0],
                                                           fixAOI_timein.iloc[0],
                                                           1]
                    else:
                        data_trial.loc[len(data_trial)] = [ID_stim,
                                                       'BLINK',
                                                       0,
                                                       0]                       
                else:
                    data_trial.loc[len(data_trial)] = [ID_stim,
                                                       'NOT FIRST PASS',
                                                       0,
                                                       0]
            else:
                data_trial.loc[len(data_trial)] = [ID_stim,
                                                   'SKIPPED',
                                                   0,
                                                   0]                
            
    return data_trial

base_dir = "//cbsu/data/Imaging/hauk/users/fm02/MEG_NEOS/ET_data"

participant = [#'156',
               #'165',
               #'190',
               #'191',
               #'193',
               #'194'
               #'195',
               # '196',
               # '197',
               # '198',
               # '199',
               # '202',
               # '203',
               # '204',
               # '206',
               # '207',
               # '209',
               # '210',
               # '213',
               # '226',
               # '228',
               # '229',
               # '232',
               '235',
               '245',
               '246',
               '025'
               ]

data = {}
data_plain = {}

for i in participant:
    print(f'Reading EDF data participant {i}')
    data[i] = read_edf(f"{base_dir}/{i}/{i}.asc",
                       "TRIGGER 94","TRIGGER 95")
    data_plain[i] = read_edf_plain(f"{base_dir}/{i}/{i}.asc")

# loop over participants  
for sbj in data.keys():
    print(f'Extracting data participant {sbj}')
    data_pilot2 = fixAOI(data[sbj], data_plain[sbj])
    
    data_pilot2.to_csv(f'//cbsu/data/Imaging/hauk/users/fm02/MEG_NEOS/ET_data/{sbj}/ET_info_{sbj}.csv', index=False)

    with open(f'//cbsu/data/Imaging/hauk/users/fm02/MEG_NEOS/ET_data/{sbj}/data_{sbj}.P', 'wb') as outfile:
        pickle.dump(data[sbj],outfile)


##########################################################################
# # change event_156.asc

# import os
# import numpy as np
# import pandas as pd
# import re
# import pickle

# f = open('events_156.asc', 'r')
# raw = f.readlines()
# f.close()

# for i,row in enumerate(raw):
#     raw[i] = row.split()

# new = pd.DataFrame(raw)

# new = new.drop(new[new[2]=='92'].index).reset_index(drop=True)

# for row in new.index:
#     if new[2].loc[row] == '95':
#         if new[1].loc[row-1] == '1':
#             new[2].loc[row] = '1000'
#         elif new[1].loc[row-1] == '2':
#             new[2].loc[row] = '2000'
#         elif new[1].loc[row-1] == '3':
#             new[2].loc[row] = '3000'
#         elif new[1].loc[row-1] == '4':
#             new[2].loc[row] = '4000'
#         elif new[1].loc[row-1] == '5':
#             new[2].loc[row] = '5000'
            
# new.to_csv('newevents_156.asc', index=False, header=False, sep='\t')
        
# new[new[2] == '95']


# #
# #

# code for old presentations (i.e. probably id 156 and 165)

# i = 165
# fn = f"{base_dir}/{i}/{i}.asc"
# f = open(fn, 'r')
# raw = f.readlines()
# f.close()

# # get all indices the start with 'END' (when disconnected tracker eg end of trial)
# ends = []
# for row in range(len(raw)):
#     line = raw[row].split()
#     if len(line)>1:
#         if line[0] == 'END':
#             ends.append(row)

# # get indices of start of new blocks
# blocks = []
# for block in [80, 160, 240, 320]:
#     blocks.append([i for i in range(len(raw)) if f'START TRIAL {block}' in raw[i]][0])
    
# # get indices (with respect to ends NOT raw) of the end of the last trial in a block
# separators = []
# for sep in range(len(blocks)):
#     separators.append(len([blocks[sep] - end for end in ends if blocks[sep]>end])-1)

# raw_divided = {}    
# raw_divided[0] = raw[0:ends[separators[0]]]
# raw_divided[1] = raw[ends[separators[0]]:ends[separators[1]]]
# raw_divided[2] = raw[ends[separators[1]]:ends[separators[2]]]
# raw_divided[3] = raw[ends[separators[2]]:ends[separators[3]]]
# raw_divided[4] = raw[ends[separators[3]]:-1]

# for k in raw_divided.keys():
#     with open(f'165_block{k}.asc', 'w') as f:
#         for line in raw_divided[k]:
#             f.write("%s" % line)

