#!/usr/bin/env python
# coding: utf-8

import numpy as np
import h5py
from pathlib import Path
from cxiapi import cxiData, calibrateFixedGainModule
import matplotlib.pyplot as plt
from p_tqdm import p_umap
from functools import partial
import multiprocessing as mp

# Experiment run number.
run = 364
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

# Hitfinding on which module.
module_index = 15
intens_thresh = 5
adu_per_photon = 45

# Fixed gain
cxi.setGainMode(0)
cxi.setADU_per_photon(adu_per_photon)
# ROI
ROI = (slice(512 - 50, None), slice(None, 51))
cxi.setROI(ROI)
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


def getHits(idx_range: list) -> list:
    # Get the index of the hits
    num_cpus = checkChuck(len(idx_range), chuckSize=25000)
    print(f'Using {num_cpus} CPU cores.')
    results = p_umap(partial(check_snapshot, module_index=15),
                     idx_range,
                     num_cpus=num_cpus)
    results = list(filter((None).__ne__, results))
    hit_results = np.array(results)
    hits_indices = hit_results[:, 0]
    hits_scores = hit_results[:, 1]
    return hits_indices, hits_scores


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
    if intens_thresh < hit_score:
        return snap_idx, hit_score
    else:
        return None


def checkChuck(ntask, chuckSize):
    num_cpu = min(mp.cpu_count(), ntask // chuckSize)
    num_cpu = max(num_cpu, 1)
    return num_cpu


idx_range = good_frames
hits_indices, hits_scores = getHits(idx_range)

with h5py.File(f'{run}_hits.h5', 'w') as h5:
    h5["hits_indices"] = hits_indices
    h5["hits_scores"] = hits_scores
