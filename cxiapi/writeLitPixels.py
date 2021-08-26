import numpy as np
from pathlib import Path
import sys
from timeit import default_timer as timer
from argparse import ArgumentParser
from cxiapi import cxiData
from litPixelsAnalyzer import litPixelsAnalyzer

default_cxi_path = '/gpfs/exfel/u/scratch/SPB/202130/p900201/spi-comission/vds/'

parser = ArgumentParser(
    prog=sys.argv[0],
    usage='%(prog)s [-h] [-m MODULE] [--nproc NPROC] run1,run2,run3,...')
parser.add_argument('run',
                    help='A run or several runs seperating with comma.',
                    type=str)
parser.add_argument('--nproc',
                    help='Number of processors to use. (Default: use all) ',
                    type=int,
                    default=0)
parser.add_argument(
    '-m',
    '--module',
    help='The detector module index for hitfinding (Default: %(default)s)',
    type=int,
    default=15)
parser.add_argument('--cxi_folder',
                    help='The folder of cxi files',
                    type=int,
                    default=default_cxi_path)
args = parser.parse_args()
runs = [int(item) for item in args.run.split(',')]
module = args.module
nproc = args.nproc
cxi_folder = args.cxi_folder

print(f'runs: {runs}')
print(f'hitfinding with module {module}\n')
for run in runs:
    start = timer()
    cxi_path = Path(cxi_folder, f'r{run:04}.cxi')
    fn = str(cxi_path)
    with cxiData(fn, verbose=1, debug=0) as cxi:
        pulse = np.arange(0, 352)
        base_pulse_filter = np.ones(600, dtype="bool")
        base_pulse_filter[len(pulse):] = False
        base_pulse_filter[0] = False
        base_pulse_filter[18::32] = False
        base_pulse_filter[29::32] = False
        good_cells = pulse[base_pulse_filter[:len(pulse)]]
        cxi.setGoodCells(good_cells)

        cxi.setCalib(
            '/gpfs/exfel/exp/SPB/202130/p900201/usr/Software/calib/r0361-r0362-r0363/'
        )
        cxi.setGeom()

        start_litpixel = timer()
        analyzer = litPixelsAnalyzer(cxi, verbose=1)
        analyzer.readLitPixels(f'./r{run:04}_hits.h5', module)

        start_frame = timer()
        analyzer.setLitPixelsThreshold()
        analyzer.getHitFrames(selection=range(500))
        end = timer()
        print('getHitFrames: {:.2f} s elapsed'.format(end - start_frame))
        analyzer.writeHitFrames()
        end = timer()
        print('getLitPixels: {:.2f} s elapsed'.format(end - start_litpixel))
    end = timer()
    print('Run {:04} done: {:.2f} s elapsed'.format(run, end - start))
