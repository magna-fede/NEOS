#!/bin/bash
# shrink inner skull if BEM surfaces touch
# usage: NEOS_shrink_innerskull.sh <subject>
# note: NEOS_watershed_shrink.sh must be in same directory
# note: check whether symbolic link was properly established
# FM, adapted from FPVS repo https://github.com/olafhauk/FPVS_sweep/

export FSVER='6.0.0'

export FSDIR=${FSROOT}/${FSVER}

export FREESURFER_HOME=/imaging/local/software/freesurfer/${FSVER}/`arch`

echo $FREESURFER_HOME

source $FREESURFER_HOME/FreeSurferEnv.sh

export MNE_ROOT=/imaging/local/software/mne/mne_2.7.3/x86_64/MNE-2.7.3-3268-Linux-x86_64
export MNE_BIN_PATH=$MNE_ROOT/bin

export PATH=${PATH}:${MNE_BIN_PATH}
# source $MNE_ROOT/bin/mne_setup

export SUBJECTS_DIR=/imaging/hauk/users/fm02/MEG_NEOS/MRI
echo $SUBJECTS_DIR

export SUBJECT=$1
echo $SUBJECT

rm -fR ${SUBJECTS_DIR}/${SUBJECT}/bem/watershed3/*

# shrinking script
dos2unix NEOS_watershed_shrink.sh
./NEOS_watershed_shrink.sh --subject ${SUBJECT} --overwrite

ln -sf ${SUBJECTS_DIR}/${SUBJECT}/bem/watershed3/${SUBJECT}_inner_skull_surface ${SUBJECTS_DIR}/${SUBJECT}/bem/inner_skull.surf