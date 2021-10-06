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


class hitAnalyzer():
    """The analyzer to get hit snapshots of a cxiData. It's dedicated to fixed gain mode so far."""
    def __init__(self, hits_fn: str, verbose: int = 0):
        super(hitAnalyzer, self).__init__()
        self.verbose = verbose
        self.hits_results = {}
        self.hits_fn = hits_fn
        self._readScores()

    def _readScores(self):
        with h5py.File(self.hits_fn, 'r') as h5:
            self.frame_indices = h5['frame_indices'][()]
            self.frame_scores = h5['frame_scores'][()]

    def getHits(self, cxi: cxiData, score_thresh: float = None) -> None:
        if intens_thresh < np.log(nphotons.sum()):
            return snap_idx


def plotFrameScores(run: int, frame_scores):
    frame_scores.sort()
    fig = plt.figure(figsize=(13.5, 5))
    ax1 = fig.add_subplot(121)
    ax1.plot(frame_scores)
    plt.grid()
    ax1.set_ylabel('Frame scores')
    ax1.set_xlabel('Sorted index')
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
    plt.savefig(f'{run}_frame_scores.png', dpi=100)