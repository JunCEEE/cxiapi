import sys
import re
import os
import numpy as np
from numpy import ndarray
import multiprocessing as mp
import h5py
import matplotlib.pyplot as plt
from cxiapi import cxiData
from functools import partial


class hitsAnalyzer():
    """The analyzer to get hit snapshots of a cxiData. It's dedicated to fixed gain mode so far."""
    def __init__(self, hits_fn: str, verbose: int = 0):
        super(hitsAnalyzer, self).__init__()
        self.verbose = verbose
        self.hits_results = {}
        self.hits_fn = hits_fn
        self.input = {}
        self.__readScores()
        self.__init_cxi()

    def __readScores(self):
        with h5py.File(self.hits_fn, 'r') as h5:
            for grp, value in h5.items():
                try:
                    val = value[()]
                    if isinstance(val, bytes):
                        self.input[grp] = val.decode('ascii')
                    else:
                        self.input[grp] = val
                except (AttributeError, TypeError):
                    self.input[grp] = {}
                    for key in h5[grp]:
                        self.input[grp][key] = h5[grp][key][()]
        self.input['hits_ROI_value'] = replaceInf(self.input['hits_ROI_value'])
        self.hits_ROI_value = self.input['hits_ROI_value']
        self.hits_module = self.input['hits_module']
        self.hits_adu_per_photon = self.input['hits_adu_per_photon']
        self.frame_scores = self.input['frame_scores']
        self.frame_indices = self.input['frame_indices']
        self.run = self.input['run']

    def __init_cxi(self):
        fn = self.input['cxi_fname']
        calib_folder = self.input['cxi_calib_folder']
        gain_mode = self.input['cxi_gain_mode']
        geom_file = self.input['cxi_geom_file']
        cxi = cxiData(fn)
        cxi.setCalib(calib_folder)
        cxi.setGeom(geom_file)
        cxi.setGainMode(gain_mode)
        self.cxi = cxi

    def plotFrameScores(self):
        plotFrameScores(self.run, self.frame_scores)

    def plotHitModule(self, snap_idx, ROI='hits_ROI', vmin=None, vmax=4):
        """Plot a hit-finding module with hits information.

        Args:
            snap_idx (int): The index of the data to plot.
            ROI (str or list, optional): The ROI to apply. Defaults to 'hits_ROI' to apply the ROI used for the score
            calculation. `None` means applying no ROI.
            vmin (float, optional): The lower colorbar limit. Defaults to None.
            vmax (float, optional): The higher colorbar limit. Defaults to 4.
        """
        run = self.run
        frame_scores = self.frame_scores
        if ROI == 'hits_ROI':
            ROI = self.hits_ROI_value
        score = frame_scores[snap_idx]
        title_txt = f'run {run} - shot {snap_idx} - score {score:.3}'
        self.cxi.plot(snap_idx,
                      self.hits_module,
                      ADU=False,
                      transpose=True,
                      ROI_value=ROI,
                      vmax=vmax,
                      vmin=vmin)
        plt.title(title_txt)

    def plotHitAppend(self,
                      snap_idx,
                      ROI=((500, 800), (430, 700)),
                      vmin=None,
                      vmax=4):
        """Plot a hit-finding append image with hits information.

        Args:
            snap_idx (int): The index of the data to plot.
            ROI (str or list, optional): The ROI to apply. Defaults to the center of the detector.
            vmin (float, optional): The lower colorbar limit. Defaults to None.
            vmax (float, optional): The higher colorbar limit. Defaults to 4.
        """
        run = self.run
        frame_scores = self.frame_scores
        score = frame_scores[snap_idx]
        title_txt = f'run {run} - shot {snap_idx} - score {score:.3}'
        self.cxi.plot(snap_idx, ADU=False, ROI_value=ROI, vmax=vmax, vmin=vmin)
        plt.title(title_txt)

    # def getHits(self, cxi: cxiData, score_thresh: float = None) -> None:
    #     if intens_thresh < np.log(nphotons.sum()):
    #         return snap_idx


def plotFrameScores(run: int, frame_scores, sort=False):
    """Plot the hit scores of the frames in a run.

    Args:
        run (int): The run number of these frames.
        frame_scores (list): A list of frame scores.
        sort (bool): If sort the frame score plot. default = False.
    """
    if sort:
        frame_scores.sort()
    fig = plt.figure(figsize=(13.5, 5))
    ax1 = fig.add_subplot(121)
    ax1.plot(frame_scores)
    plt.grid()
    ax1.set_ylabel('Frame scores')
    if sort:
        ax1.set_xlabel('Sorted index')
    else:
        ax1.set_xlabel('Index')
    plt.title(f'run {run}')
    bin_values, bin_edges = np.histogram(
        frame_scores,
        bins=int((frame_scores.max() - frame_scores.min() + 1) / 0.2))
    bin_centers = np.array([(bin_edges[i] + bin_edges[i + 1]) / 2
                            for i in range(len(bin_values))])
    ax2 = fig.add_subplot(122)
    ax2.plot(bin_centers, bin_values, 'k')
    plt.grid()
    plt.ylabel('Frequency')
    plt.xlabel('Frame scores')
    plt.savefig(f'r{run:04}_frame_scores.png', dpi=100)


def replaceInf(arr: ndarray):
    my_list = arr.tolist()
    for i in range(2):
        for j in range(2):
            if my_list[i][j] == np.inf:
                my_list[i][j] = None
            else:
                my_list[i][j] = int(my_list[i][j])
    return my_list