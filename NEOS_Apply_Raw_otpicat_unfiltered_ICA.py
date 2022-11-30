#!/imaging/local/software/miniconda/envs/mne0.20/bin/python
"""
Apply ICA for MEG NEOS.
Decompostion computed using sentences as epochs (0-padded)

author: federica.magnabosco@mrc-cbu.cam.ac.uk 
"""

import sys

from os import path
import numpy as np
import pandas as pd

import matplotlib
import matplotlib.pyplot as plt

from importlib import reload

import mne
from mne.preprocessing import ICA, create_eog_epochs

print('MNE Version: %s\n\n' % mne.__version__)  # just in case
print(mne)

import NEOS_config as config
reload(config)

mne.viz.set_browser_backend("matplotlib")

class CreateArgs:
    """Parser for input arguments."""

    def __init__(self, FileRawIn, FileICA, FileRawOut):
        self.FileRawIn = FileRawIn
        self.FileICA = FileICA
        self.FileRawOut = FileRawOut

EOG = ['EOG001','EOG002']
reject = config.reject
show = False

def get_mysources(ica_obj, raw, add_channels=None, start=None, stop=None):
    """This is modifying mne get_sources function in ica.
    That function for some reasons crops data at the end of first block.
    We instead want to fit each component to the whole raw timecourse.
    """
    mne.utils.check._check_compensation_grade(ica_obj.info, raw.info, 'ICA', 'Raw', ica_obj.ch_names)

    data_ = ica_obj._transform_raw(raw, start=start, stop=stop)
    assert data_.shape[1] == stop - start
    
    preloaded = raw.preload
    if raw.preload:
        # get data and temporarily delete
        data = raw._data
        raw.preload = False
        del raw._data
    # copy and crop here so that things like annotations are adjusted
    try:
        out = raw.copy().crop(
            start / raw.info['sfreq'],
            (stop - 1) / raw.info['sfreq'])
    finally:
        # put the data back (always)
        if preloaded:
            raw.preload = True
            raw._data = data
    
    # populate copied raw.
    if add_channels is not None and len(add_channels):
        picks = mne.io.pick.pick_channels(raw.ch_names, add_channels)
        data_ = np.concatenate([
            data_, raw.get_data(picks, start=start, stop=stop)])
    out._data = data_
    out._first_samps = [raw.first_samp]
    out._last_samps = [raw.last_samp]
    out._filenames = [None]
    out.preload = True
    out._projector = None
    ica_obj._export_info(out.info, raw, add_channels)
    
    return out


def run_Apply_ICA(sbj_id):
    """Apply previously computed ICA to raw data."""

    sbj_path = path.join(config.data_path, config.map_subjects[sbj_id][0])

    # raw-filename mappings for this subject
    tmp_fnames = config.sss_map_fnames[sbj_id][1]

    # only use files for correct conditions
    sss_map_fnames = []
    for sss_file in tmp_fnames:
        sss_map_fnames.append(sss_file)

    data_raw_files = []

    for raw_stem_in in sss_map_fnames:
        data_raw_files.append(
            path.join(sbj_path, raw_stem_in[:-7] + 'sss_f_raw.fif'))
    
    print(f'Reading raw file {sss_map_fnames}')
    data = []
    for drf in data_raw_files:
        data.append(mne.io.read_raw_fif(drf))

    print('Concatenating data')
    data = mne.concatenate_raws(data)
    
    print(f'Reading raw file {sss_map_fnames}')
    data = []
    for drf in data_raw_files:
        data.append(mne.io.read_raw_fif(drf))

    print('Concatenating data')
    data = mne.concatenate_raws(data)
    
    data.load_data()

    
    all_events = mne.read_events(path.join(sbj_path, config.map_subjects[sbj_id][0][-3:] + \
                              '_all_events.fif'))   
    pd_all_events = pd.read_csv(path.join(sbj_path, config.map_subjects[sbj_id][0][-3:] + \
                                  '_all_events_xy.csv'))

    t0 = data.first_samp
        
    start_saccades = np.where(pd_all_events['trigger']==801)[0]
    end_saccades = np.where(pd_all_events['trigger']==802)[0]
    
    start_fixations = np.where(pd_all_events['trigger']==901)[0]
    end_fixations = np.where(pd_all_events['trigger']==902)[0]
    
    
    times_sac = tuple(zip(start_saccades, end_saccades))

    sac_selection = dict.fromkeys(['data', 'time'])
    sac_selection['data'] = list()
    sac_selection['time'] = list()
    
    for i, indices in enumerate(times_sac):
        d, t = data[:,(all_events[indices[0]][0] - t0) : (all_events[indices[1]][0] - t0) ]
        sac_selection['data'].append(d)    
        sac_selection['time'].append(t) 
    
    times_fix = tuple(zip(start_fixations, end_fixations))

    fix_selection = dict.fromkeys(['data', 'time'])
    fix_selection['data'] = list()
    fix_selection['time'] = list()
    
    for i, indices in enumerate(times_fix):
        d, t = data[:,(all_events[indices[0]][0] - t0) : (all_events[indices[1]][0] - t0) ]
        fix_selection['data'].append(d)    
        fix_selection['time'].append(t) 
        
    print('Reading ICA file')
    ica = mne.preprocessing.read_ica(path.join(sbj_path, sbj_path[-3:] + '_sss_f_raw-ica_overweighted_unfiltered_raw.fif'))

    components_timecourse = get_mysources(ica, data, None, 0, data.n_times)
    
    var_sac = []
    for event in sac_selection['time']:
        section = components_timecourse.get_data(tmin=event[0],
                                                 tmax=event[-1])
        var_sac.append(np.var(section, axis=1))
        
    var_sac = np.dstack(var_sac).squeeze()
    
    var_sac = np.mean(var_sac, axis =1)

    var_fix = []
    for event in fix_selection['time']:
        section = components_timecourse.get_data(tmin=event[0],
                                                 tmax=event[-1])
        var_fix.append(np.var(section, axis=1))
        
    var_fix = np.dstack(var_fix).squeeze()
    
    var_fix = np.mean(var_fix, axis =1)
    
    to_exclude = np.where((var_sac/var_fix) > 1.1)[0]
    
    ica.exclude = to_exclude

    report = mne.Report(subject=config.map_subjects[sbj_id][0],
                         title='ICA')

    # plot for specified channel types
    for ch_type in ['eeg', 'mag', 'grad']:

        fig_ic = ica.plot_components(ch_type=ch_type, show=show)

        caption = [ch_type.upper() + ' Components' for i in fig_ic]

        report.add_figure(fig_ic, title=ch_type.upper() +'Components', caption=caption,
                                   section='ICA Components')

    for eog_ch in EOG:

        print('\n###\nFinding components for EOG channel %s.\n' % eog_ch)

        # get single EOG trials
        eog_epochs = create_eog_epochs(data, ch_name=eog_ch, reject=reject)

        eog_average = eog_epochs.average()  # average EOG epochs

        # find via correlation
        inds = to_exclude
        scores = (var_sac/var_fix)
        
        if inds != []:  # if some bad components found
            fig_sc = ica.plot_scores(scores, exclude=inds, show=show)

            report.add_figure(fig_sc, caption='%s Scores' %
                           eog_ch, title='Scores as var(saccade) / var(fixation)',
                           section='%s ICA component \
                           scores' % eog_ch)
                           
            print('Plotting raw ICA sources.')
            fig_rc = ica.plot_sources(data, show=show)

            report.add_figure(fig_rc, title='Sources', caption='%s Sources' %
                                       eog_ch, section='%s raw ICA sources'
                                       % eog_ch)

            print('Plotting EOG average sources.')
            # look at source time course
            fig_so = ica.plot_sources(eog_average, show=show)

            report.add_figure(fig_so, title='Raw EOG Sources', caption='%s Sources' %
                                       eog_ch, section='%s ICA Sources' %
                                       eog_ch)

            print('Plotting EOG epochs properties.')
            fig_pr = ica.plot_properties(eog_epochs,  picks=inds,
                                         psd_args={'fmax': 35.},
                                         image_args={'sigma': 1.},
                                         show=show)

            txt_str = '%s Properties' % eog_ch
            caption = [txt_str for i in fig_pr]

            report.add_figure(fig_pr, caption=caption, title='Properties',
                                       section='%s ICA Properties' %
                                       eog_ch)

            print(ica.labels_)

            # Remove ICA components #######################################
            fig_ov = ica.plot_overlay(eog_average, exclude=inds, show=show)
            # red -> before, black -> after.

            report.add_figure(fig_ov, title='Overlay',
                             caption='%s Overlay' % eog_ch,
                             section='%s ICA Overlay' % eog_ch)

            plt.close('all')


        else:

            print('\n!!!Nothing bad found for %s!!!\n' % eog_ch)

    
#    ica.plot_sources(data)
    
    print('Applying ICA to raw file')
    ica.apply(data)

    data.save(path.join(sbj_path, sbj_path[-3:] + "_sss_f_ica_od_unfiltered_onraw_raw.fif"), overwrite=True)

    
    print(f'Saving ICA to {path.join(sbj_path, sbj_path[-3:] + "_sss_f_raw-ica__unfiltered_opticat.fif")}')
    ica.save(path.join(sbj_path, sbj_path[-3:] + '_sss_f_raw-ica_unfiltered_onraw_opticat.fif'), overwrite=True)
    report.save(path.join(sbj_path, 'Figures', 'report_ica_unfiltered_onraw_opticat.html'), overwrite=True)

    
# get all input arguments except first
if len(sys.argv) == 1:

    sbj_ids = np.arange(0, len(config.map_subjects)) + 1

else:

    # get list of subjects IDs to process
    sbj_ids = [int(aa) for aa in sys.argv[1:]]


for ss in sbj_ids:

    run_Apply_ICA(ss)
