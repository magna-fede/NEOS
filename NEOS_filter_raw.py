#!/imaging/local/software/miniconda/envs/mne0.20/bin/python
"""
Filter and clean data from FPVS Frequency Sweep.

Get event information, interpolate bad
EEG channels, (notch) filter.
==========================================

fm02 based on OH
"""

import sys
from os import path
import numpy as np

import matplotlib
matplotlib.use('Agg')  # for running graphics on cluster ### EDIT
from matplotlib import pyplot as plt

from importlib import reload

import mne

import NEOS_config as config
reload(config)

print('MNE Version: %s\n\n' % mne.__version__)  # just in case
print(mne)

# whether to show figures on screen or just write to file
show = False

def run_filter_raw(sbj_id):
    """Clean data for one subject."""
    # path to subject's data
    sbj_path = path.join(config.data_path, config.map_subjects[sbj_id][0])

    # raw-filename mappings for this subject
    tmp_fnames = config.sss_map_fnames[sbj_id][1]

    # only use files for correct conditions
    sss_map_fnames = []

    for sss_file in tmp_fnames:
        sss_map_fnames.append(sss_file)

    print(sss_map_fnames)

    bad_eeg = config.bad_channels[sbj_id]['eeg']  # bad EEG channels

    for raw_stem_in in sss_map_fnames:

        # input file to read
        raw_fname_in = path.join(sbj_path, raw_stem_in + '.fif')

        # result file to write
        raw_fname_out = raw_fname_in[:-7] + 'f_raw.fif'

        print('\n###\nReading raw file %s.' % raw_fname_in)

        raw = mne.io.read_raw_fif(raw_fname_in, preload=True)

        raw = raw.pick_types(meg=True, eeg=True, eog=True, stim=True, ecg=True, emg=True)

        print('Fixing coil types.')
        raw.fix_mag_coil_types()

        # ONLY FOR EEG
        if any('EEG' in ch for ch in raw.info['ch_names']):

            print('Marking bad EEG channels: %s' % bad_eeg)
            raw.info['bads'] = bad_eeg

            print('Interpolating bad EEG channels.')
            raw.interpolate_bads(mode='accurate', reset_bads=True)

            print('Setting EEG reference.')
            raw.set_eeg_reference(ref_channels='average', projection=True)

        else:

            print('No EEG channels found.\n')

        print('Applying Notch filter.')

        raw.notch_filter(np.array([50, 100]), fir_design='firwin',
                         trans_bandwidth=0.04)

        # str() because of None
        print(f'Applying band-pass filter {config.l_freq} to {config.h_freq} Hz.')

        # broad filter, including VGBR and ASSR frequencies
        # most settings are the MNE-Python defaults (zero-phase FIR)
        # https://mne.tools/dev/auto_tutorials/discussions/plot_background_filtering.html
        raw.filter(l_freq=config.l_freq, h_freq=config.h_freq, method='fir',
                   fir_design='firwin', filter_length='auto',
                   l_trans_bandwidth='auto', h_trans_bandwidth='auto')

        print('Saving data to %s.' % raw_fname_out)
        raw.save(raw_fname_out, overwrite=True)

        print('Finding events.')
        # note: short event duration
        events = mne.find_events(raw, stim_channel='STI101',
                                 consecutive='increasing', min_duration=0.002,
                                 verbose=True)

        # correct for stimulus presentation delay
        stim_delay = int(config.delay * raw.info['sfreq'])
        events[:, 0] = events[:, 0] + stim_delay

##########################################################################################################################
##########################################################################################################################
### HEY! THINK ABOUT THIS! the sentences actually appear (and disappear) 30ms after what's reported on stim channel
### However in theory we don't care about this, because need to know the activity in real time (not locked to stimulus)
### this might be relevant only when estimating erp activity (to avoid 34ms difference) 
##########################################################################################################################
##########################################################################################################################
        
        # event_file = path.join(sbj_path, raw_stem_in + '_sss_f_raw-eve.fif')
        # print('Saving events to %s.' % event_file)
        # #mne.write_events(event_file, events)

        # plot only if events were found
        if events.size != 0:

            fig = mne.viz.plot_events(events, raw.info['sfreq'], show=show)

            fname_fig = path.join(sbj_path, 'Figures',
                                raw_stem_in + '_sss_f_raw_eve.pdf')
            print('Saving figure to %s' % fname_fig)

            fig.savefig(fname_fig)

            plt.close(fig)

        else:

            print('No events found in file %s.' % raw_fname_in)

        # # get response latencies for target stimuli (colour changes)

        # # find target events (colour changes, trigger value 8)
        # targ_eves = np.where(events[:, 2] == 8)[0]

        # if len(targ_eves) > 0:  # only if target events present (not rest)

        #     rts = []  # collect response times for this fiff-file

        #     # find response events (triggers >= 4096, allowing trigger overlap)
        #     resp_eves = np.where(events[:, 2] >= 4096)[0]

        #     # find responses closest following targets, compute time difference
        #     for tt in targ_eves:

        #         rt = []  # rt for this particular target

        #         # find first response that follows target
        #         for rr in resp_eves:

        #             if rr > tt:

        #                 # subtract samples response - target
        #                 rt = events[rr, 0] - events[tt, 0]

        #                 # turn samples to latency (ms)
        #                 rt = 1000. * (rt / raw.info['sfreq'])

        #                 # only count if RT below a threshold
        #                 if rt <= 2000.:

        #                     rts.append(rt)

        #                 break  # leave this loop

        #     if rts == []:  # if no good responses found

        #         print('\nNo good target responses found!\n')

        #     else:

        #         print('\nResponse times to targets:\n')
        #         print(*rts)
        #         print('Mean: %f.' % np.mean(rts))

        # else:

        #     print('No target events present.')

    return raw, events


# get all input arguments except first
if len(sys.argv) == 1:

    sbj_ids = np.arange(0, len(config.map_subjects)) + 1

else:

    # get list of subjects IDs to process
    sbj_ids = [int(aa) for aa in sys.argv[1:]]


for ss in sbj_ids:

    [raw, events] = run_filter_raw(ss)
