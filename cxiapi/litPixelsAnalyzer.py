import sys
import re
import os
import numpy as np
from numpy import ndarray
import multiprocessing as mp
import h5py
import ctypes
import time
from tqdm import tqdm
from timeit import default_timer as timer
import matplotlib.pyplot as plt
from cxiapi import cxiData, calibrateModule
from psutil import virtual_memory


class litPixelsAnalyzer():
    """The analyzer to get the lit Pixels of a cxiData."""
    def __init__(self, cxi_data: cxiData, verbose: int = 0):
        super(litPixelsAnalyzer, self).__init__()
        self.cxi_data = cxi_data
        self.verbose = verbose
        self.litpix_params = {}
        self.module_masks = {}
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
        self.adu_per_photon = 45

    def setModuleMasks(self, module_idx, mask):
        self.module_masks[str(module_idx)] = mask


    def plotModule(self, snap_idx, module_index, is_transpose=False):
        calib_data = getCalibData(self.cxi_data, snap_idx, module_index)
        nphotons = calib_data / self.adu_per_photon
        nphotons[nphotons < 0.5] = 0
        if is_transpose:
            nphotons = nphotons.transpose()
        plt.figure()
        plt.imshow(nphotons, vmax=2, origin='lower')

    def plotROI(self, snap_idx, module_index, is_transpose=False):
        calib_data = getCalibData(self.cxi_data,snap_idx, module_index)*self.module_masks[str(module_index)]
        roi_data = calib_data[self.ROI]
        data_indices = np.indices(calib_data.shape)
        row_max = np.max(data_indices[0][self.ROI])
        row_min = np.min(data_indices[0][self.ROI])
        col_max = np.max(data_indices[1][self.ROI])
        col_min = np.min(data_indices[1][self.ROI])
        extent = [col_min, col_max, row_min, row_max]
        nphotons = roi_data / self.adu_per_photon
        nphotons[nphotons < 0.5] = 0
        if is_transpose:
            nphotons = nphotons.transpose()
            extent = [row_min, row_max, col_min, col_max]
        plt.figure()
        plt.imshow(nphotons, vmax=2, origin='lower', extent=extent)



    def getLitPixels(self,
                     module: int,
                     nproc: int = 0,
                     pixel_thresh: float = 25,
                     chunk_size=4096):
        """Get the number of hit pixelsself.

        Args:
            module (int): The detector module index.
            nproc (int, optional): Number of processes to use. Defaults to 0 meaning to use all
            availale CPUs.
            pixel_thresh (float, optional): The threshold for pixels counting. Defaults to 25.
            chunk_size (int, optional): The chunk size of each CPU process. Defaults to 4096.
        """
        print(f'VDS file: {self.cxi_data.fname}')
        nframes_filtered = self.cxi_data.nframes_filtered
        print('Calculating number of lit pixels for %d frames'
              % nframes_filtered)

        if chunk_size % 32 != 0:
            print(
                'WARNING: Performance is best with a multiple of 32 chunk_size'
            )
        if nproc == 0:
            nproc = mp.cpu_count()

        # Total machine memory in MB
        mem = int(virtual_memory().total / 1024 / 1024)
        # Memory needed in MB (roughly)
        mem_needed = nproc * chunk_size
        if mem_needed > mem:
            new_nproc = int(nproc // 2)
            print(
                f'WARNING: Required memory size ({mem_needed:d} MB) exceeds available mem ({mem:d} MB)'
            )
            print(f'WARNING: Will reduce nproc = {nproc} to {new_nproc}')
            nproc = new_nproc
        if self.verbose > 0:
            print('Using %d processes' % nproc)
        self.litpix_params = {
            'module': module,
            'nproc': nproc,
            'pixel_thresh': pixel_thresh,
            'chunk_size': chunk_size
        }

        litpix = mp.Array(ctypes.c_ulong, nframes_filtered)
        jobs = []
        for c in range(nproc):
            p = mp.Process(target=self._LitPixelsWorker,
                           args=(c, module, litpix))
            jobs.append(p)
            p.start()
        for j in tqdm(jobs):
            j.join()
        self.litpix = np.frombuffer(litpix.get_obj(), dtype='u8')
        self.cxi_good_frames = self.cxi_data.good_frames
        nframe = len(self.litpix)
        self.nframe = nframe

    def readLitPixels(self, fname: str, module: int):
        """Read lit pixels from a xx_hits.h5 file.

        Args:
            fname (str): The h5 file name.
            module (int): The detector module index for hitfinding.

        Raises:
            ValueError: The litpix data of the module must exist.
        """
        rname = os.path.basename(fname).split('_')[0]
        self.run = int(re.findall(r'\d+', rname)[0])
        with h5py.File(fname, 'r') as f:
            if 'litpixels_%.2d' % module in f:
                self.litpix_params = {}
                for key, value in f['litpix_params_%.2d' % module].items():
                    self.litpix_params[key] = value[()]
                self.litpix = f['litpixels_%02d' % module][:]
                self.cxi_good_frames = f['cxi_good_frames'][()]
                nframe = len(self.litpix)
                self.nframe = nframe
                print(f'{fname} contains {nframe} frames.')
            else:
                raise ValueError(
                    f'Did not find the module = {module} in this file: {fname}'
                )

    def setLitPixelsThreshold(self,
                              threshold=None,
                              plot: bool = True,
                              upper_threshold=None):
        lp = self.litpix
        if threshold is None:
            module = self.litpix_params['module']
            bin_values, bin_edges = np.histogram(
                lp, bins=int(lp.max() - lp.min() + 1))
            bin_centers = np.array([(bin_edges[i] + bin_edges[i + 1]) / 2
                                    for i in range(len(bin_values))])
            t_min, t_max = 0.22, 0.8  # fraction of data used to calculate threshold
            lp_sorted = np.sort(lp)
            m = np.median(lp_sorted[int(len(lp_sorted) * t_min
                                        ):int(len(lp_sorted) * t_max)])
            s = lp_sorted[int(len(lp_sorted) * t_min
                              ):int(len(lp_sorted) * t_max)].std()
            threshold = m + 10 * s
            if plot:
                self._plotLitPixelsThreshold(lp, module, threshold,
                                             bin_centers, bin_values,
                                             upper_threshold)
        self.litpix_params['threshold'] = threshold
        if upper_threshold is None:
            frames = np.where(lp > threshold)[0]
        else:
            frames = np.where((lp > threshold) & (lp <= upper_threshold))[0]
        self.frame_selection = frames
        hit_rate = len(self.frame_selection) / self.nframe * 100
        self.hit_rate = hit_rate
        print(f'Hit rate = {hit_rate:.2f}%')
        print('Found %d hits above lit-pixel threshold = %.0f ADU'
              % (len(frames), threshold))

    def getHitFrames(self, nproc=0, selection=None):
        cxi_good_frames = self.cxi_good_frames
        if selection and len(selection) <= len(self.frame_selection):
            hit_frame_selection = self.frame_selection[selection]
        else:
            hit_frame_selection = self.frame_selection
        self.hit_frame_selection = hit_frame_selection
        self.assembled = self.cxi_data.getCalibratedFrames(
            hit_frame_selection,
            assemble=True,
            good_frames=cxi_good_frames,
            nproc=nproc)

    def writeHitFrames(self, out_fn=None):
        params = self.litpix_params
        out_fn = self._getOutFname(out_fn)
        if os.path.exists(out_fn):
            print('WARNING: overwriting file: %s' % out_fn)
        f = h5py.File(out_fn, 'a')
        if 'hits' in f:
            del f['hits']
        selction = self.cxi_good_frames[self.hit_frame_selection]
        g = f.create_group("hits")
        g['litpixelThreshold'] = params['threshold']
        g['litpixelModule'] = params['module']
        g['raw'] = self.cxi_data.calib_frame
        g['assembled'] = self.assembled
        g['frameID'] = self.hit_frame_selection
        g['pulseID'] = self.cxi_data.pulseIDs[selction]
        g['trainID'] = self.cxi_data.trainIDs[selction]
        g['cellID'] = self.cxi_data.cellIDs[selction, 0]
        g['hit_rate'] = self.hit_rate
        if self.run is not None:
            print('Saved %d frames from run %d to: %s' % (len(
                self.assembled), self.run, out_fn))
        f.close()
        os.system('chmod ao+rw %s' % out_fn)

    def _plotLitPixelsThreshold(self, lp, module, threshold, bin_centers,
                                bin_values, upper_threshold):
        # plot to determine threshold
        fig = plt.figure(num=None,
                         figsize=(13.5, 5),
                         dpi=100,
                         facecolor='w',
                         edgecolor='k')
        canvas = fig.add_subplot(121)
        canvas.plot(lp, '.')
        plt.hlines(threshold,
                   0,
                   len(lp),
                   colors='r',
                   linestyles='--',
                   zorder=10)
        if upper_threshold is not None:
            plt.hlines(upper_threshold,
                       0,
                       len(lp),
                       colors='r',
                       linestyles='--',
                       zorder=10)
            plt.ylim(0, upper_threshold * 1.1)
        if self.run is not None:
            plt.suptitle('run %d' % self.run)
        plt.title('lit pixels in module %02d - common-mode corrected' % module)
        plt.xlabel('shot number')
        plt.ylabel('number of lit pixels')
        canvas = fig.add_subplot(122)
        canvas.plot(bin_centers, bin_values, 'k')
        plt.vlines(threshold,
                   3e-1,
                   bin_values.max(),
                   colors='r',
                   linestyles='--',
                   zorder=10)
        if upper_threshold is not None:
            plt.vlines(upper_threshold,
                       3e-1,
                       bin_values.max(),
                       colors='r',
                       linestyles='--',
                       zorder=10)
            plt.xlim(0, upper_threshold * 1.1)
        plt.yscale('log')
        plt.title('lit-pixel histogram - threshold=%.0f ADU' % threshold)
        plt.ylabel('frequency')
        plt.xlabel('number of lit pixels')
        if self.run is not None:
            plt.savefig('run%d_lit_pixels.png' % self.run)
            print('Saved figure: run%d_lit_pixels.png' % self.run)

    def writeLitPixels(self, out_fname: str = None):
        out_fname = self._getOutFname(out_fname)
        module = self.litpix_params['module']
        if (self.verbose):
            print(f'Writting to {out_fname}')
        try:
            f = h5py.File(out_fname, 'a')
        except FileExistsError:
            f = h5py.File(out_fname, 'w')

        if 'litpixels_%.2d' % module in f:
            del f['litpixels_%.2d' % module]
        if 'litpix_params_%.2d' % module in f:
            del f['litpix_params_%.2d' % module]
        if 'cxi_good_frames' in f:
            del f['cxi_good_frames']
        assert len(self.cxi_good_frames) == len(self.litpix)
        grp = f.create_group(f"litpix_params_{module:02}")
        for key, value in self.litpix_params.items():
            grp[key] = value
        f['litpixels_%.2d' % module] = self.litpix
        f['cxi_good_frames'] = self.cxi_good_frames

        f.close()
        os.system('chmod ao+rw %s' % out_fname)
        if (self.verbose):
            print(f'Writting to {out_fname} done')

    def _getOutFname(self, out_fname):
        if out_fname is None:
            os_base = os.path.basename(self.cxi_data.fname)
            base_name = re.findall(r'^(\S+)\.\S+$', os_base)[0]
            out_fname = base_name.split('_')[0] + '_hits.h5'
        return out_fname

    def _LitPixelsWorker(self, p, m, litpix):
        np_litpix = np.frombuffer(litpix.get_obj(), dtype='u8')
        good_frames = self.cxi_data.good_frames
        data = self.cxi_data.data
        cellIDs = self.cxi_data.cellIDs

        nframes = self.cxi_data.nframes_filtered
        calib = self.cxi_data.calib
        self.nproc = self.litpix_params['nproc']
        self.chunk_size = self.litpix_params['chunk_size']
        my_start = (nframes // self.nproc) * p
        if (p == self.nproc - 1):
            remainder = nframes % self.nproc
            my_end = min((nframes // self.nproc) * (p + 1) + remainder,
                         nframes)
        else:
            my_end = min((nframes // self.nproc) * (p + 1), nframes)
        num_chunks = int(np.ceil((my_end - my_start) / self.chunk_size))
        if p == 0:
            print('Doing %d chunks of %d frames each'
                  % (num_chunks, self.chunk_size))
        for c in range(num_chunks):
            pmin = my_start + c * self.chunk_size
            pmax = min(my_start + (c + 1) * self.chunk_size, my_end)
            frame_filter = good_frames[pmin:pmax]

            stime = time.time()
            vals = np.float32(data[frame_filter, m])
            filtered_cellIDs = cellIDs[frame_filter, m]
            calib_vals = np.empty(
                (vals.shape[0], vals.shape[2], vals.shape[3]))
            for n in range(vals.shape[0]):
                calib_vals[n] = calibrateModule(vals[n, 0, :, :],
                                                vals[n, 1, :, :], m,
                                                filtered_cellIDs[n], calib)
            etime = time.time()
            if p == 0:
                sys.stderr.write(
                    '(%.4d/%.4d) %d frames in %.4f s (%.2f Hz)\n'
                    % (c + 1, num_chunks, vals.shape[0], etime - stime,
                       vals.shape[0] * self.nproc / (etime - stime)))

            np_litpix[pmin:pmax] = (calib_vals >
                                    self.litpix_params['pixel_thresh']).sum(
                                        (1, 2))

    def _copy_ids(self, fptr: h5py.File):
        """Copy IDs to the ouput h5 file object pointer

        Args:
            fptr (h5py.File): The ouput h5 file object pointer.
        """
        good_frames = self.cxi_data.good_frames
        if 'ID/trainId' in fptr:
            del fptr['ID/trainId']
        fptr['ID/trainId'] = self.cxi_data.trainIDs[good_frames]
        if 'ID/cellId' in fptr:
            del fptr['ID/cellId']
        fptr['ID/cellId'] = self.cxi_data.cellIDs[good_frames, 0]
        if 'ID/pulseId' in fptr:
            del fptr['ID/pulseId']
        fptr['ID/pulseId'] = self.cxi_data.pulseIDs[good_frames]

def check_snapshot(snap_idx, module_index, threshold, module_mask, ROI, adu_per_photon):
    calib_data = getCalibData(cxi, snap_idx, module_index)*module_mask
    roi_data = calib_data[ROI]
    nphotons = roi_data / adu_per_photon
    nphotons[nphotons < 0.5] = 0
    if threshold < np.log(nphotons.sum()):
        return snap_idx

def getCalibData(cxi_data, snap_idx, module_index):
    n = snap_idx
    data = cxi_data.data
    cell_ids = cxi_data.cellIDs

    calib_data = calibrateModule(data[n, module_index, 0, :, :],
                                    data[n, module_index,
                                        1, :, :], module_index,
                                    cell_ids[n, 0], cxi_data.calib)
    return calib_data

def getHits(idx_range: list, module_index,
            intens_thresh: float, module_mask, ROI, adu_per_photon) -> list:
    # Get the index of the hits
    with mp.Pool(mp.cpu_count()) as pool:
        results = pool.starmap_async(
            check_snapshot, [(snap_idx, module_index, intens_thresh, module_mask, ROI, adu_per_photon)
                                    for snap_idx in idx_range]).get()
    hits_indices = list(filter((None).__ne__, results))
    return  hits_indices