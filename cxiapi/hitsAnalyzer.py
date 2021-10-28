import numpy as np
from numpy import ndarray
import h5py
import matplotlib.pyplot as plt
from tqdm import tqdm
from cxiapi import cxiData


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
        self.hits_photon_thresh = self.input['hits_photon_thresh']
        self.hits_ROI_value = self.input['hits_ROI_value']
        self.hits_module = self.input['hits_module']
        self.lit_pixels_ROI = self.input['lit_pixels_ROI']
        self.lit_pixels_module = self.input['lit_pixels_module']
        self.intensity_scores_ROI = self.input['intensity_scores_ROI']
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
        cxi.setADU_per_photon(self.input['cxi_adu_per_photon'])
        self.cxi = cxi

    def plotFrameScores(self):
        plotHist(self.run, self.intensity_scores_ROI, 'intensity_scores_ROI')
        plotHist(self.run, self.lit_pixels_ROI, 'lit_pixels_ROI')
        plotHist(self.run, self.lit_pixels_module, 'lit_pixels_module')

    def plotHitModule(self,
                      snap_idx=None,
                      frame_idx=None,
                      ROI='hits_ROI',
                      vmin=None,
                      vmax=4):
        """Plot a hit-finding module with hits information.

        Args:
            snap_idx (int): The snap index in the original cxi file. Defaults to None.
            frame_idx (int): The index of the scored frames to plot.
            ROI (str or list, optional): The ROI to apply. Defaults to 'hits_ROI' to apply the ROI used for the score
            calculation. `None` means applying no ROI.
            vmin (float, optional): The lower colorbar limit. Defaults to None.
            vmax (float, optional): The higher colorbar limit. Defaults to 4.
        """
        run = self.run
        if frame_idx and snap_idx:
            raise ValueError(
                "'snap_idx' and 'frame_idx' should NOT be set at the same time."
            )

        if frame_idx:
            snap_idx = self.frame_indices[frame_idx]
        elif snap_idx:
            frame_idx = np.where(self.frame_indices == snap_idx)[0][0]
        else:
            raise ValueError(
                "One and only one of 'snap_idx' and 'frame_idx' should be set."
            )

        if ROI == 'hits_ROI':
            ROI = self.hits_ROI_value

        score = self.intensity_scores_ROI[frame_idx]
        lit_pixels = self.lit_pixels_ROI[frame_idx]
        lit_pixels_module = self.lit_pixels_module[frame_idx]
        title_txt = f'run {run} - shot {snap_idx} - IS {score:.3} - LP {lit_pixels} - LPM {lit_pixels_module}'

        self.cxi.plot(snap_idx,
                      self.hits_module,
                      ADU=False,
                      transpose=True,
                      ROI_value=ROI,
                      vmax=vmax,
                      vmin=vmin)
        plt.title(title_txt)

    def plotHitAppend(self,
                      snap_idx=None,
                      frame_idx=None,
                      ROI=((500, 800), (430, 700)),
                      vmin=None,
                      vmax=4):
        """Plot a hit-finding append image with hits information.

        Args:
            snap_idx (int): The snap index in the original cxi file. Defaults to None.
            frame_idx (int): The index of the scored frames to plot.
            ROI (str or list, optional): The ROI to apply. Defaults to the center of the detector.
            vmin (float, optional): The lower colorbar limit. Defaults to None.
            vmax (float, optional): The higher colorbar limit. Defaults to 4.
        """
        run = self.run
        if frame_idx and snap_idx:
            raise ValueError(
                "'snap_idx' and 'frame_idx' should NOT be set at the same time."
            )

        if frame_idx:
            snap_idx = self.frame_indices[frame_idx]
        elif snap_idx:
            frame_idx = np.where(self.frame_indices == snap_idx)[0][0]
        else:
            raise ValueError(
                "One and only one of 'snap_idx' and 'frame_idx' should be set."
            )

        score = self.intensity_scores_ROI[frame_idx]
        lit_pixels = self.lit_pixels_ROI[frame_idx]
        lit_pixels_module = self.lit_pixels_module[frame_idx]
        title_txt = f'run {run} - shot {snap_idx} - IS {score:.3} - LP {lit_pixels} - LPM {lit_pixels_module}'
        self.cxi.plot(snap_idx, ADU=False, ROI_value=ROI, vmax=vmax, vmin=vmin)
        plt.title(title_txt)

    def plotHits(self,
                 num_max=5,
                 inten_lim=[0, np.inf],
                 lp_lim=[0, np.inf],
                 vmin=None,
                 vmax=4,
                 order='descending',
                 save=False):
        # The number 1180-1189 is meant to prevent the abnormal data due to detector misbehavior.
        threshold_idx = np.where((self.intensity_scores > inten_lim[0])
                                 & (self.lit_pixels > lp_lim[0])
                                 & (self.intensity_scores < inten_lim[1])
                                 & (self.lit_pixels < lp_lim[1])
                                 & ((self.lit_pixels > 1189)
                                    | (self.lit_pixels < 1180)))[0]
        max_inten_score = np.max(self.intensity_scores)
        max_lit_pixel = np.max(self.lit_pixels)
        frame_scores = self.intensity_scores[threshold_idx] / max_inten_score
        frame_scores += self.lit_pixels[threshold_idx] / max_lit_pixel
        if order == 'descending':
            ind = np.argsort(-frame_scores)[:num_max]
        elif order == 'ascending':
            ind = np.argsort(frame_scores)[:num_max]
        else:
            raise ValueError("Available 'order': 'descending' or 'ascending'")

        for num, i in enumerate(tqdm(ind)):
            snap_idx = self.frame_indices[threshold_idx][i]
            if (num < 10):
                print(f'#{num} frame {snap_idx}: {frame_scores[i]}')
            elif num == 10:
                print('...')
            frame_idx = threshold_idx[i]
            self.plotHitModule(frame_idx=frame_idx, vmin=vmin, vmax=vmax)
            if (save):
                plt.savefig(f'r{self.run:04}_{num:04}_module.png', dpi=100)
                plt.close()
            self.plotHitAppend(frame_idx=frame_idx, vmin=vmin, vmax=vmax)
            if (save):
                plt.savefig(f'r{self.run:04}_{num:04}_append.png', dpi=100)
                plt.close()


def plotHist(run: int, data, label: str, sort=False):
    """Plot a histogram of a dataset in a run.

    Args:
        run (int): The run number of these frames.
        data: The data used to create the histogram
        label (str): The label of the data.
        sort (bool): If sort the intensity score plot. default = False.
    """
    if sort:
        data.sort()
    fig = plt.figure(figsize=(13.5, 5))
    ax1 = fig.add_subplot(121)
    ax1.plot(data)
    plt.grid()
    ax1.set_ylabel(label)
    if sort:
        ax1.set_xlabel('Sorted index')
    else:
        ax1.set_xlabel('Index')
    plt.title(f'run {run}')
    bin_values, bin_edges = np.histogram(
        data, bins=int((data.max() - data.min() + 1) / 0.2))
    bin_centers = np.array([(bin_edges[i] + bin_edges[i + 1]) / 2
                            for i in range(len(bin_values))])
    ax2 = fig.add_subplot(122)
    ax2.plot(bin_centers, bin_values, 'k')
    plt.grid()
    plt.ylabel('Frequency')
    plt.xlabel(label)
    plt.savefig(f'r{run:04}_{label}.png', dpi=100)


def replaceInf(arr: ndarray):
    my_list = arr.tolist()
    for i in range(2):
        for j in range(2):
            if my_list[i][j] == np.inf:
                my_list[i][j] = None
            else:
                my_list[i][j] = int(my_list[i][j])
    return my_list