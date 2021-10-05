"""Main module."""

import sys
import numpy as np
from numpy import ndarray
import matplotlib.pyplot as plt
import h5py
import glob
from timeit import default_timer as timer
from extra_geom import AGIPD_1MGeometry


class cxiData():
    """CXI data class"""
    def __init__(self, fname, verbose: int = 0, debug=0):
        super(cxiData, self).__init__()
        # gain_mode = None means adaptive gain mode.
        self.gain_mode = None
        self.fname = fname
        self.dset_name = '/entry_1/instrument_1/detector_1/data'
        self.train_name = '/entry_1/trainId'
        self.pulse_name = '/entry_1/pulseId'
        self.cell_name = '/entry_1/cellId'
        self.verbose = verbose
        self._load_vds()
        dset_shape = self.data.shape
        self.module_eg = np.empty(
            (dset_shape[1], dset_shape[3], dset_shape[4]))
        self.debug = debug

        self.ROI = [slice(None), slice(None)]
        self.module_masks = {}

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self._close_vds()
        self._close_calib()

    def _load_vds(self):
        fname = self.fname
        try:
            self.vds = h5py.File(fname, 'r')
        except IOError:
            print('ERROR: file does not exist: %s' % fname)
            sys.exit(-1)
        self.num_h5cells = np.max(self.cellIDs[:, 0]) + 1
        if self.verbose > 0:
            print('VDS file contains %d shots' % self.nframes)
            print('Module 0 contains %d cells' % self.num_h5cells)

    def _close_calib(self):
        try:
            self.calib.close()
        except ValueError:
            print('WARNING: VDS file has already been closed')
        except AttributeError:
            pass

    def _close_vds(self):
        try:
            self.vds.close()
        except ValueError:
            print('WARNING: VDS file has already been closed')

    def getCalibrateModule(self, snap_idx: int, module_idx: int):
        """Get the array of a calibrated module.

        Args:
            snap_idx (int): The index of a snapshotself.
            module_idx (int): Which module to check.

        Returns:
            ndarray: calibrated module data.
        """
        n = snap_idx
        data = self.data
        cell_ids = self.cellIDs

        if self.gain_mode is not None:
            calib_data = calibrateFixedGainModule(data[n, module_idx, 0, :, :],
                                                  data[n, module_idx, 1, :, :],
                                                  self.gain_mode, module_idx,
                                                  cell_ids[n, module_idx],
                                                  self.calib)

        else:
            calib_data = calibrateModule(data[n, module_idx, 0, :, :],
                                         data[n, module_idx,
                                              1, :, :], module_idx,
                                         cell_ids[n, module_idx], self.calib)
        calib_data[calib_data < 0] = 0
        return calib_data

    def getCalibrateDetector(self, snap_idx: int):
        n = snap_idx
        calib_det = np.empty((16, 512, 128))
        for i in range(16):
            calib_data = self.getCalibrateModule(n, i)
            calib_det[i] = calib_data
        return calib_det

    def setGainMode(self, mode: int):
        """Set gain mode.

        Args:
            mode (int): Fixed gain mode. 0: low, 1: medium, 2: high.
            `None` means adaptive gain.
        """
        assert mode in [0, 1, 2, None]
        self.gain_mode = mode

    @property
    def nframes(self):
        """The total frame number."""
        return self.data.shape[0]

    @property
    def frame_filter(self):
        """The good cells filter for all the frames"""
        try:
            frame_filter = np.zeros(self.nframes, dtype=bool)
            for i in range(0, self.nframes, self.num_h5cells):
                frame_filter[self.good_cells + i] = True

            return frame_filter
        except AttributeError:
            raise AttributeError(
                "Please run the .setGoodCells() method of the cxiData first.")

    @property
    def good_frames(self):
        """The good frames indices after good cells filtering for all the frames"""
        try:
            good_frames = np.arange(self.nframes)
            return (good_frames[self.frame_filter])
        except AttributeError:
            raise AttributeError(
                "Please run the .setGoodCells() method of the cxiData first.")

    @property
    def nframes_filtered(self):
        """Number of frames after filtering."""
        return len(self.frame_filter[self.frame_filter])

    @property
    def data(self):
        """The raw data in shape (#snapshots, 16, 2, 512, 128)"""
        return self.vds[self.dset_name]

    @property
    def trainIDs(self):
        """The trainIDs in an array of the size = #snapshots"""
        return self.vds[self.train_name]

    @property
    def pulseIDs(self):
        """The pulseIDs in an array of the size = #snapshots"""
        return self.vds[self.pulse_name]

    @property
    def cellIDs(self):
        """The cellIDs in an array of the shape = (#snapshots, 16)"""
        return self.vds[self.cell_name]

    def setGeom(
        self,
        geom_fname='/gpfs/exfel/exp/SPB/202130/p900201/usr/Software/geom/agipd_2696_v5.geom'
    ):
        self.geom_fname = geom_fname

    def setCalib(
        self,
        calib_path:
        str = '/gpfs/exfel/exp/SPB/202130/p900201/usr/Software/calib/r0355-r0356-r0357/'
    ):
        """Get the list of the Cheetah calibration hdf5 objects.

        Args:
            calib_path (str): The path of the Cheetah calibration files.
        """
        calib_glob = '%s/Cheetah*.h5' % calib_path
        self.calib_glob = calib_glob
        self.calib_files = sorted(glob.glob(calib_glob))
        self.calib = [h5py.File(f, 'r') for f in self.calib_files]
        if self.verbose > 0:
            print('%d calibration files found' % len(self.calib_files))

    def setGoodCells(self, good_cells=None):
        """Set an array of the indices of good cells

        Args:
            good_cells (list or range, optional): e.g. range(1, 352) for all the cells are good except the first one (index=0).
            Defaults to None, meaning all the cells are good.

        Raises:
            TypeError: 'good_cells' has to be list-like.
        """
        if good_cells is None:
            self.good_cells = np.arange(self.num_h5cells)
        else:
            if isinstance(good_cells, (list, range, ndarray)):
                if len(good_cells) <= self.num_h5cells:
                    self.good_cells = np.array(good_cells)
                else:
                    raise ValueError(
                        "`len(good_cells) = {}` has to be smaller than num_h5cells = {}."
                        .format(len(good_cells), self.num_h5cells))
            else:
                raise TypeError("'good_cells' has to be list-like.")

    def setROI(self, ROI: list):
        """Set Region of Intrest slices for data analysis

        Args:
            ROI (list): a list of 2 slices.
            Example: `cxi.setROI(slice(400, None), slice(None, 500))` is to set row > 400 and col < 500.
        """
        self.ROI = ROI
        assert len(ROI) == 2
        assert isinstance(ROI[0], slice) is True
        assert isinstance(ROI[1], slice) is True

    def cleanROI(self):
        self.ROI = [slice(None), slice(None)]

    def setModuleMasks(self, module_idx: int, mask):
        self.module_masks[str(module_idx)] = mask

    def cleanModuleMasks(self, module_idx=None):
        if module_idx:
            del self.module_masks[str(module_idx)]
        else:
            self.module_masks = {}

    def setADU_per_photon(self, value=45):
        self.adu_per_photon = value

    def inspectGeom(self):
        geom = AGIPD_1MGeometry.from_crystfel_geom(self.geom_fname)
        geom.inspect()

    def assembleDetector(self, calib_detector: ndarray):
        geom = AGIPD_1MGeometry.from_crystfel_geom(self.geom_fname)
        res, centre = geom.position_modules_fast(calib_detector)
        return res

    def getPostProcessedData(self,
                             snap_idx: int,
                             module_idx: int = None,
                             module_mask: ndarray = None,
                             ADU: bool = True):

        if module_idx is None:
            # Return the whole detector
            calib_detector = self.getCalibrateDetector(snap_idx)
            if not ADU:
                calib_detector /= self.adu_per_photon
                calib_detector[calib_detector < 0.5] = 0
            for i, mask in self.module_masks.items():
                calib_detector[int(i)] *= mask
            return calib_detector
        else:
            # Return one module
            calib_data = self.getCalibrateModule(snap_idx, module_idx)
            if not ADU:
                calib_data /= self.adu_per_photon
                calib_data[calib_data < 0.5] = 0
            if module_mask is None:
                try:
                    module_mask = self.module_masks[str(module_idx)]
                    calib_data *= module_mask
                except KeyError:
                    pass
            else:
                calib_data *= module_mask
            return calib_data

    def fastPlotCalibDetector(self, snap_idx, **kwargs):
        geom = AGIPD_1MGeometry.from_crystfel_geom(self.geom_fname)
        geom.plot_data_fast(self.getCalibrateDetector(snap_idx), **kwargs)

    def plot(self,
             snap_idx: int,
             module_idx: int = None,
             ROI: list = None,
             module_mask: ndarray = None,
             ADU: bool = True,
             transponse: bool = False,
             **kwargs):
        if ROI is None:
            ROI = self.ROI
        if module_idx is None:
            calib_detector = self.getPostProcessedData(snap_idx, module_idx,
                                                       module_mask, ADU)
            if not ADU:
                kwargs['vmax'] = kwargs.pop('vmax', 2)
            plotDetector(self.assembleDetector(calib_detector), ROI, **kwargs)
        else:
            calib_data = self.getPostProcessedData(snap_idx, module_idx,
                                                   module_mask, ADU)
            if not ADU:
                kwargs['vmax'] = kwargs.pop('vmax', 2)
            plotModule(calib_data, ROI, transponse, **kwargs)


def plotDetector(assemble_detector: ndarray, ROI: list = None, **kwargs):
    data_indices = np.indices(assemble_detector.shape)
    row_max = np.max(data_indices[0][ROI])
    row_min = np.min(data_indices[0][ROI])
    col_max = np.max(data_indices[1][ROI])
    col_min = np.min(data_indices[1][ROI])

    kwargs['origin'] = kwargs.pop('origin', 'lower')

    plt.figure(figsize=(8, 8))
    plt.imshow(assemble_detector, **kwargs)
    plt.xlim(col_max, col_min)
    plt.ylim(row_min, row_max)
    plt.colorbar()


def plotModule(calib_data: ndarray,
               ROI: list = None,
               transpose: bool = False,
               **kwargs):
    data_indices = np.indices(calib_data.shape)
    row_max = np.max(data_indices[0][ROI])
    row_min = np.min(data_indices[0][ROI])
    col_max = np.max(data_indices[1][ROI])
    col_min = np.min(data_indices[1][ROI])
    extent = [col_min, col_max, row_min, row_max]
    roi_data = calib_data[ROI]

    if transpose:
        roi_data = roi_data.transpose()
        extent = [row_min, row_max, col_min, col_max]

    kwargs['origin'] = kwargs.pop('origin', 'lower')
    kwargs['extent'] = kwargs.pop('extent', extent)
    plt.figure()
    plt.imshow(roi_data, **kwargs)
    plt.colorbar()


def calibrateModule(data: ndarray,
                    gain: ndarray,
                    module: int,
                    cell: int,
                    calib: list,
                    cmode=True) -> ndarray:
    """Calibrate one module with Cheetah calibration files.

    Args:
        data (ndarray): Image data in shape (512, 128).
        gain (ndarray): Gain data in shape (512, 128).
        module (int): Module index (0-15).
        cell (int): Cell index.
        calib (list): A list of calibration hdf5_objects/dicts.
        cmode (bool, optional): Common mode correction. Defaults to True.

    Returns:
        ndarray: The calibrated image of the module.
    """
    gain_mode = gainThreshold(gain, module, cell, calib)
    offset = np.empty(gain_mode.shape)
    gain = np.empty(gain_mode.shape)
    badpix = np.empty(gain_mode.shape)
    for i in range(3):
        offset[gain_mode == i] = calib[module]['AnalogOffset'][i, cell][
            gain_mode == i]
        gain[gain_mode == i] = calib[module]['RelativeGain'][i, cell][gain_mode
                                                                      == i]
        badpix[gain_mode == i] = calib[module]['Badpixel'][i, cell][gain_mode
                                                                    == i]

    data = (np.float32(data) - offset) * gain
    data[badpix != 0] = 0

    if cmode:
        # Median subtraction by 64x64 asics
        data = data.reshape(8, 64, 2, 64).transpose(1, 3, 0,
                                                    2).reshape(64, 64, 16)
        data -= np.median(data, axis=(0, 1))
        data = data.reshape(64, 64, 8, 2).transpose(2, 0, 3,
                                                    1).reshape(512, 128)

    return data


def calibrateFixedGainModule(data: ndarray,
                             gain: ndarray,
                             mode: int,
                             module: int,
                             cell: int,
                             calib: list,
                             cmode=True) -> ndarray:
    """Calibrate one fixed gain module with Cheetah calibration files.

    Args:
        data (ndarray): Image data in shape (512, 128).
        gain (ndarray): Gain data in shape (512, 128).
        mode (int): Fixed gain mode. 0: low, 1: medium, 2: high.
        module (int): Module index (0-15).
        cell (int): Cell index.
        calib (list): A list of calibration hdf5_objects/dicts.
        cmode (bool, optional): Common mode correction. Defaults to True.

    Returns:
        ndarray: The calibrated image of the module.
    """
    offset = calib[module]['AnalogOffset'][mode, cell]
    gain = calib[module]['RelativeGain'][mode, cell]
    badpix = calib[module]['Badpixel'][mode, cell]

    data = (np.float32(data) - offset) * gain
    data[badpix != 0] = 0

    if cmode:
        # Median subtraction by 64x64 asics
        data = data.reshape(8, 64, 2, 64).transpose(1, 3, 0,
                                                    2).reshape(64, 64, 16)
        data -= np.median(data, axis=(0, 1))
        data = data.reshape(64, 64, 8, 2).transpose(2, 0, 3,
                                                    1).reshape(512, 128)

    return data


def gainThreshold(gain, module, cell, calib):
    threshold = calib[module]['DigitalGainLevel'][:, cell]
    high_gain = gain < threshold[1]
    low_gain = gain > threshold[2]
    medium_gain = (~high_gain) * (~low_gain)
    return low_gain * 2 + medium_gain
