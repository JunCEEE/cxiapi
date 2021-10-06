import sys
import re
import os
import numpy as np
from numpy import ndarray
import multiprocessing as mp
import h5py
from p_tqdm import p_umap, p_map
import matplotlib.pyplot as plt
from cxiapi import cxiData
from functools import partial


class hitAnalyzer():
    """The analyzer to get hit snapshots of a cxiData. It's dedicated to fixed gain mode so far."""
    def __init__(self, verbose: int = 0):
        super(hitAnalyzer, self).__init__()
        self.verbose = verbose
        self.hit_results = {}

    def setRunNumber(self, cxi: cxiData, run: int = None):
        """Set the run number of this classs.

        Args:
            run (int, optional): The run number to set. Defaults to None meaning to
            set from the cxi file name.
        """
        if run is None:
            rname = os.path.basename(cxi.fname).split('_')[0]
            self.run = int(re.findall(r'\d+', rname)[0])
        else:
            self.run = run

    def getHits(self, cxi: cxiData, idx_range: list, module_idx: int,
                intens_thresh: float, num_cpus = None) -> None:
        if num_cpus is None:
            num_cpus = checkChuck(len(idx_range), chuckSize=25000)
        if self.verbose > 0:
            print(f'Using {num_cpus} CPU cores.')
        results = p_umap(partial(check_snapshot,
                                 cxi=cxi,
                                 module_idx=module_idx,
                                 intens_thresh=intens_thresh),
                         idx_range,
                         num_cpus=num_cpus)
        hits_indices = list(filter((None).__ne__, results))
        self.hit_results['hits_indices'] = hits_indices


def check_snapshot(snap_idx: int, module_idx: int, intens_thresh: float,
                   cxi: cxiData):
    nphotons = cxi.getPostProcessedData(snap_idx, module_idx)
    if intens_thresh < np.log(nphotons.sum()):
        return snap_idx


def checkChuck(ntask, chuckSize):
    num_cpu = min(mp.cpu_count(), ntask // chuckSize)
    num_cpu = max(num_cpu, 1)
    return num_cpu
