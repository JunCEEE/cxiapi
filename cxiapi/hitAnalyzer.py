import sys
import re
import os
import numpy as np
from numpy import ndarray
import multiprocessing as mp
import h5py
from p_tqdm import p_umap, p_map
import matplotlib.pyplot as plt
from cxiapi import cxiData, calibrateFixedGainModule
from functools import partial


class hitAnalyzer():
    """The analyzer to get hit snapshots of a cxiData. It's dedicated to fixed gain mode so far."""
    def __init__(self, cxi_data: cxiData, verbose: int = 0):
        super(hitAnalyzer, self).__init__()
        self.cxi_data = cxi_data
        self.verbose = verbose
        self.module_masks = {}
        self.hit_results = {}
        try:
            self.setRunNumber()
        except:
            self.run = None

    def setRunNumber(self, run: int = None):
        """Set the run number of this classs.

        Args:
            run (int, optional): The run number to set. Defaults to None meaning to
            set from the cxi file name.
        """
        if run is None:
            rname = os.path.basename(self.cxi_data.fname).split('_')[0]
            self.run = int(re.findall(r'\d+', rname)[0])
        else:
            self.run = run

    def setROI(self, ROI):
        self.ROI = ROI
        assert len(ROI) == 2

    def setAdu_per_photon(self, value=45):
        self.adu_per_photon = value

    def setModuleMasks(self, module_index, mask):
        self.module_masks[str(module_index)] = mask

    def toPhotons(self, calib_data):
        """Convert ADU to photons.

        Args:
            calib_data (ndarray): The ADU data array.

        Returns:
            ndarray: Photons array.
        """
        nphotons = calib_data / self.adu_per_photon
        nphotons[nphotons < 0.5] = 0
        return nphotons

    def getROIdata(self, snap_idx, module_index, ROI=None):
        if ROI is None:
            my_ROI = self.ROI
        else:
            my_ROI = ROI
        calib_data = self.cxi_data.getCalibrateModule(snap_idx, module_index)
        roi_data = calib_data[my_ROI] * self.module_masks[str(
            module_index)][my_ROI]
        return roi_data

    def getROIphoton(self, snap_idx, module_index, ROI=None):
        roi_data = self.getROIdata(snap_idx, module_index, ROI)
        nphotons = self.toPhotons(roi_data)
        return nphotons

    def plotModule(self, snap_idx, module_index, is_transpose=False):
        calib_data = self.cxi_data.getCalibrateModule(snap_idx, module_index)
        nphotons = self.toPhotons(calib_data)
        if is_transpose:
            nphotons = nphotons.transpose()
        plt.figure()
        plt.imshow(nphotons, vmax=2, origin='lower')

    def plotROI(self, snap_idx, module_index, ROI=None, is_transpose=False, save_fn:str=None):
        if ROI is None:
            my_ROI = self.ROI
        else:
            my_ROI = ROI
        calib_data = self.cxi_data.getCalibrateModule(snap_idx, module_index)
        roi_data = calib_data[my_ROI] * self.module_masks[str(
            module_index)][my_ROI]
        nphotons = self.toPhotons(roi_data)

        data_indices = np.indices(calib_data.shape)
        row_max = np.max(data_indices[0][my_ROI])
        row_min = np.min(data_indices[0][my_ROI])
        col_max = np.max(data_indices[1][my_ROI])
        col_min = np.min(data_indices[1][my_ROI])
        extent = [col_min, col_max, row_min, row_max]

        if is_transpose:
            nphotons = nphotons.transpose()
            extent = [row_min, row_max, col_min, col_max]
        plt.figure()
        figure_title = f'run {self.run} - snapshot {snap_idx} - module {module_index}'
        plt.title(figure_title)
        plt.imshow(nphotons, vmax=2, origin='lower', extent=extent)
        if save_fn is not None:
            plt.savefig(save_fn)

    def getHits(self, idx_range: list, module_index,
                intens_thresh: float) -> None:
        num_cpus = checkChuck(len(idx_range), chuckSize=25000)
        if self.verbose > 0:
            print(f'Using {num_cpus} CPU cores.')
        results = p_umap(partial(check_snapshot,
                                 module_index=module_index,
                                 threshold=intens_thresh,
                                 ROI=self.ROI,
                                 adu_per_photon=self.adu_per_photon,
                                 mask=self.module_masks[str(module_index)],
                                 fname=self.cxi_data.fname,
                                 dset_name=self.cxi_data.dset_name,
                                 cell_name=self.cxi_data.cell_name,
                                 calib_files=self.cxi_data.calib_files,
                                 mode=self.cxi_data.gain_mode),
                         idx_range,
                         num_cpus=num_cpus)
        hits_indices = list(filter((None).__ne__, results))
        self.hit_results['hits_indices'] = hits_indices


def check_snapshot(snap_idx, module_index, threshold, ROI, adu_per_photon, mask,
                   fname, dset_name, cell_name, calib_files, mode):
    with h5py.File(fname, 'r') as h5:
        data = h5[dset_name]
        cellIDs = h5[cell_name]
        calib_data = calibrateFixedGainModule(
            data[snap_idx, module_index, 0, :, :], data[snap_idx, module_index,
                                                      1, :, :], mode,
            module_index, cellIDs[snap_idx, module_index], calib_files)
    roi_data = calib_data[ROI] * mask[ROI]
    nphotons = roi_data / adu_per_photon
    nphotons[nphotons < 0.5] = 0
    if threshold < np.log(nphotons.sum()):
        return snap_idx


def checkChuck(ntask, chuckSize):
    num_cpu = min(mp.cpu_count(), ntask // chuckSize)
    num_cpu = max(num_cpu, 1)
    return num_cpu
