"""Main module."""

import sys
import numpy as np
from numpy import ndarray
import multiprocessing as mp
import ctypes
import time
import h5py
import glob
from . import geom
from .distributeJob import distributeJob
from timeit import default_timer as timer


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
        # data = self.vds[self.dset_name][0, :, 0, :, :]
        dset_shape = self.vds[self.dset_name].shape
        self.module_eg = np.empty(
            (dset_shape[1], dset_shape[3], dset_shape[4]))
        self.debug = debug

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

    def getCalibratedFrames(
        self,
        frame_idx,
        assemble=True,
        nproc=0,
    ) -> ndarray:
        # All frames are dealt with filtered cells.

        try:
            len(frame_idx)
            frame_idx = np.array(frame_idx)
        except TypeError:
            frame_idx = np.array([frame_idx])
        assert len(frame_idx.shape) == 1, "Must contain a 1D array of integers"
        if assemble:
            try:
                self.x, self.y = geom.pixel_maps_from_geometry_file(
                    self.geom_fname)
            except AttributeError:
                raise AttributeError(
                    'Run cxiData.setGeom() to set geom_fname first.')

        for n in frame_idx:
            if n > self.nframes or n < 0:
                print('Out of range: %d, skipping event..' % n)
                frame_idx = np.delete(frame_idx, np.where(frame_idx == n))

        selection = frame_idx

        try:
            self.calib
        except AttributeError:
            raise AttributeError(
                'Run cxiData.setCalib() to set Cheetah calibration files folder first.'
            )

        ntasks = selection.shape[0]
        print('Calibrate %d frames' % ntasks)
        chuck_size = 4096
        if nproc == 0:
            nproc = min(ntasks // chuck_size + 1, mp.cpu_count())
        print('Using %d processes' % nproc)

        # The number of tasks for each proc
        njobs, accu_jobs = distributeJob(nproc, ntasks)
        if self.verbose > 0:
            print('Number of tasks per processes:')
            print(njobs)

        array_size = ntasks * self.module_eg.size
        frames_arr = mp.Array(ctypes.c_ulong, array_size)
        if self.verbose:
            print('Memory allocation done', flush=True)

        jobs = []
        for c in range(nproc):
            p = mp.Process(target=self._CalibrateWorker,
                           args=(c, accu_jobs, selection, frames_arr))
            jobs.append(p)
            p.start()
        for j in jobs:
            j.join()
        frames_arr = np.frombuffer(frames_arr.get_obj(), dtype='u8')

        stime = timer()
        calib_frame_shape = (ntasks, ) + self.module_eg.shape
        calib_frame = frames_arr.reshape(calib_frame_shape)
        etime = timer()
        if (self.verbose > 1):
            print('Reshaped calib_frame in {:.2f} s'.format(etime - stime),
                  flush=True)
        if self.debug:
            print(f'main, calib_frame[{accu_jobs[0]}]',
                  calib_frame[accu_jobs[0], 15, 0, 1],
                  flush=True)
            if calib_frame.shape[0] > accu_jobs[1]:
                print(f'main, calib_frame[{accu_jobs[1]}]',
                      calib_frame[accu_jobs[1], 15, 0, 1])
        self.calib_frame = calib_frame

        if not assemble or self.geom_fname is None:
            return np.copy(calib_frame)
        else:
            stime = timer()
            # if num.shape[0] > 1:
            output = []
            for n in range(selection.shape[0]):
                # Higher level verbose
                if self.verbose > 1:
                    print('Assembling frame %d' % selection[n], flush=True)
                output.append(
                    geom.apply_geom_ij_yx((self.y, self.x), calib_frame[n]))
            etime = timer()
            print('Assembled {} frames in {:.2f} s'.format(
                selection.shape[0], etime - stime),
                  flush=True)
            return np.array(output)
            # else:
            #     return geom.apply_geom_ij_yx((self.y, self.x), calib_frame)

    def _CalibrateWorker(self, p, accu_jobs, selection, frames_arr):
        if p == 0:
            print('VDS data reading...')
        stime = timer()
        frames_arr = np.frombuffer(frames_arr.get_obj(), dtype='u8')
        num = selection.shape[0]
        new_shape = (num, ) + self.module_eg.shape
        calib_frame = frames_arr.reshape(new_shape)
        selection = selection[accu_jobs[p]:accu_jobs[p + 1]]
        data = self.vds[self.dset_name][selection]
        cell_ids = self.cellIDs[selection, 0]
        pulse_ids = self.pulseIDs[selection]
        train_ids = self.trainIDs[selection]
        etime = timer()
        if p == 0:
            print('VDS data reading done in {:.2f} s'.format(etime - stime),
                  flush=True)
        if int(self.debug) > 0 and p == 0:
            print(f'calib_frame: {calib_frame.shape}', flush=True)
            print(f'selection: {selection.shape}', flush=True)
            print(f'data: {data.shape}', flush=True)
            print(f'cell_ids: {cell_ids.shape}', flush=True)

        njobs = selection.shape[0]
        stime = timer()
        for n in range(njobs):
            if (cell_ids[n] >= self.num_h5cells):
                print('Frame has invalid cellId=%d' % cell_ids[n])
                continue
            if self.verbose > 1 and p == 0:
                print('Getting frame with cellId=%d, pulseId=%d and trainId=%d'
                      % (cell_ids[n], pulse_ids[n], train_ids[n]))
            idx = accu_jobs[p] + n
            for i in range(16):
                cval = calibrateModule(data[n, i, 0, :, :], data[n, i,
                                                                 1, :, :], i,
                                       cell_ids[n], self.calib)
                cval[cval < 0] = 0
                calib_frame[idx, i] = cval

            if self.debug > 1 and p == 0 and n < 5:
                print('calibrated module shape:', cval.shape)
                print('calibrated module dtype:', cval.dtype)
                print('calibrated [0,1]:', cval[0, 1])
            etime = timer()
            total_time = etime - stime
            frame_time = total_time / (n + 1)
            if p == 0:
                sys.stderr.write(
                    '(%.4d/%.4d) frames in process 0 in %.2f s, %.2f Hz\n'
                    % (n + 1, njobs, total_time, 1 / frame_time))
        if self.debug:
            if p == 0:
                print(f'p0, calib_frame[{accu_jobs[0]}]',
                      calib_frame[accu_jobs[0], 15, 0, 1])
                if calib_frame.shape[0] > accu_jobs[1]:
                    print(f'p0, calib_frame[{accu_jobs[1]}]',
                          calib_frame[accu_jobs[1], 15, 0, 1])
            if p == 1:
                print(f'p1, calib_frame[{accu_jobs[0]}]',
                      calib_frame[accu_jobs[0], 15, 0, 1])
                print(f'p1, calib_frame[{accu_jobs[1]}]',
                      calib_frame[accu_jobs[1], 15, 0, 1])

    def getCalibrateModule(self, snap_idx, module_index):
        n = snap_idx
        data = self.data
        cell_ids = self.cellIDs

        if self.gain_mode is not None:
            calib_data = calibrateFixedGainModule(
                data[n, module_index, 0, :, :], data[n, module_index,
                                                     1, :, :], self.gain_mode,
                module_index, cell_ids[n, module_index], self.calib_files)

        else:
            calib_data = calibrateModule(data[n, module_index, 0, :, :],
                                         data[n, module_index,
                                              1, :, :], module_index,
                                         cell_ids[n, module_index], self.calib)
        return calib_data

    def setGainMode(self, mode: int):
        """Set gain mode.

        Args:
            mode (int): Fixed gain mode. 0: low, 1: medium, 2: high.
            `None` means adaptive gain.
        """
        self.gain_mode = mode

    @property
    def nframes(self):
        """The total frame number."""
        return self.vds[self.dset_name].shape[0]

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
        self.calib = [h5py.File(f, 'r') for f in sorted(glob.glob(calib_glob))]
        self.calib_files = sorted(glob.glob(calib_glob))
        if self.verbose > 0:
            print('%d calibration files found' % len(self.calib))

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
                             calib_files: list,
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
    with h5py.File(calib_files[module], 'r') as h5:
        offset = h5['AnalogOffset'][mode, cell]
        gain = h5['RelativeGain'][mode, cell]
        badpix = h5['Badpixel'][mode, cell]

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
