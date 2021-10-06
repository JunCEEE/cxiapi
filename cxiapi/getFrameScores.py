#!/usr/bin/env python
# coding: utf-8
"""Get the frame scores for a run"""

import sys
import numpy as np
import h5py
from pathlib import Path
from cxiapi import cxiData, calibrateFixedGainModule, value2ROI
from hitsAnalyzer import plotFrameScores
import matplotlib.pyplot as plt
from p_tqdm import p_umap
from functools import partial
import multiprocessing as mp


def getHitScore(idx_range: list) -> list:
    # Get the index of the hits
    num_cpus = checkChuck(len(idx_range), chuckSize=25000)
    print(f'Using {num_cpus} CPU cores.')
    results = p_umap(partial(check_snapshot, module_index=15),
                     idx_range,
                     num_cpus=num_cpus)
    return np.array(results)


def getCalibData(snap_idx, module_index):
    n = snap_idx
    calib_data = calibrateFixedGainModule(data[n, module_index, 0, :, :],
                                          data[n, module_index,
                                               1, :, :], gain_mode,
                                          module_index, cell_ids[n, 0], calib)
    return calib_data


def check_snapshot(snap_idx, module_index):
    calib_data = getCalibData(snap_idx, module_index)
    roi_data = calib_data[ROI] * mask[ROI]
    nphotons = roi_data / adu_per_photon
    nphotons[nphotons < 0.5] = 0
    hit_score = np.log(nphotons.sum())
    return hit_score


def checkChuck(ntask, chuckSize):
    num_cpu = min(mp.cpu_count(), ntask // chuckSize)
    num_cpu = max(num_cpu, 1)
    return num_cpu


def replaceNone(input: list):
    arr = np.array(input)
    arr[arr == None] = np.inf
    # astype is important. Otherwise the type is np.o
    # which cannot be saved with h5py.
    return arr.astype(float)


if __name__ == "__main__":

    # Experiment run number.
    run = int(sys.argv[1])
    # The folder of cxi files
    cxi_folder = '/gpfs/exfel/u/scratch/SPB/202130/p900201/spi-comission/vds/'
    # Cheetah files folder for calibration
    calib_folder = '/gpfs/exfel/exp/SPB/202130/p900201/usr/Software/calib/r0361-r0362-r0363/'
    # Geometry file for the detector
    geom_file = '/gpfs/exfel/exp/SPB/202130/p900201/usr/Software/geom/agipd_2696_v5.geom'

    cxi_path = Path(cxi_folder, f'r{run:04}.cxi')
    fn = str(cxi_path)

    cxi = cxiData(fn, verbose=1, debug=0)
    pulse = np.arange(0, 352)
    base_pulse_filter = np.ones(600, dtype="bool")
    base_pulse_filter[len(pulse):] = False
    base_pulse_filter[0] = False
    base_pulse_filter[18::32] = False
    base_pulse_filter[29::32] = False
    good_cells = pulse[base_pulse_filter[:len(pulse)]]
    cxi.setGoodCells(good_cells)
    cxi.setCalib(calib_folder)
    cxi.setGeom(geom_file)

    # Hitfinding on which module.
    module_index = 15
    intens_thresh = 5
    adu_per_photon = 45

    # Fixed gain
    cxi.setGainMode(0)
    cxi.setADU_per_photon(adu_per_photon)
    # ROI
    ROI_val = ((512 - 50, None), (None, 51))
    ROI = value2ROI(ROI_val)
    # ROI = (slice(512 - 50, None), slice(None, 51))
    cxi.setROI(ROI_val)
    # Mask
    # cxi.plot(300,module_index,ADU=False,transpose=True)
    # plt.title('Before mask')

    mask = np.ones((512, 128))
    mask[470:473, 15:18] = 0
    cxi.setModuleMasks(15, mask)
    # cxi.plot(300,module_index,ADU=False,transpose=True)
    # plt.title('After mask')

    # Parallel cell
    data = cxi.data
    good_frames = cxi.good_frames
    cell_ids = cxi.cellIDs
    gain_mode = cxi.gain_mode
    calib = cxi.calib

    idx_range = good_frames[:100]
    frame_scores = getHitScore(idx_range)
    plotFrameScores(run, frame_scores)

    with h5py.File(f'{run}_hits.h5', 'w') as h5:
        h5["cxi_fname"] = cxi.fname
        h5["cxi_calib_folder"] = calib_folder
        h5["cxi_geom_file"] = geom_file
        h5["cxi_gain_mode"] = cxi.gain_mode
        h5["cxi_ROI_value"] = replaceNone(cxi.ROI_value)
        h5["frame_indices"] = idx_range
        h5["frame_scores"] = frame_scores
        if len(cxi.module_masks) > 0:
            grp = h5.create_group('module_masks')
            for key, value in cxi.module_masks.items():
                grp[key] = value
