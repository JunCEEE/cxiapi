import numpy as np
import sys
from timeit import default_timer as timer
from argparse import ArgumentParser
from cxiData import cxiData
from litPixelsAnalyzer import litPixelsAnalyzer

parser = ArgumentParser(prog=sys.argv[0], usage='%(prog)s [-h] [-m MODULE] [--nproc NPROC] run1,run2,run3,...')
parser.add_argument('run', help='A run or several runs seperating with comma.', type=str)
parser.add_argument('--nproc', help='Number of processors to use. (Default: use all) ', type=int, default=0)
parser.add_argument('-m','--module', help='The detector module index for hitfinding (Default: %(default)s)', type=int, default=15)
args = parser.parse_args()
runs = [int(item) for item in args.run.split(',')]
module = args.module
nproc = args.nproc

print(f'runs: {runs}')
print(f'hitfinding with module {module}\n')
for run in runs:
    start = timer()
    fn = f'/gpfs/exfel/u/scratch/SPB/202130/p900201/juncheng/spi-comission/cxi/r{run:04}_cxi.h5'
    with cxiData(fn,verbose=1,debug=0) as cxi:
        pulse = np.arange(0, 352)
        base_pulse_filter = np.ones(600, dtype="bool")
        base_pulse_filter[len(pulse):] = False
        base_pulse_filter[0] = False
        base_pulse_filter[18::32] = False
        base_pulse_filter[29::32] = False
        good_cells = pulse[base_pulse_filter[:len(pulse)]]
        cxi.setGoodCells(good_cells)

        cxi.setCalib('/gpfs/exfel/exp/SPB/202130/p900201/usr/Software/calib/r0361-r0362-r0363/')
        cxi.setGeom()

        start_litpixel = timer()
        analyzer = litPixelsAnalyzer(cxi, verbose=1)
        analyzer.getLitPixels(module, nproc=nproc)
        analyzer.writeLitPixels()
        end = timer()
        print('getLitPixels: {:.2f} s elapsed'.format(end - start_litpixel))

        # analyzer.readLitPixels('./r0365_hits.h5', module)

        start_frame = timer()
        analyzer.setLitPixelsThreshold()
        analyzer.getHitFrames(selection=range(500))
        end = timer()
        print('getHitFrames: {:.2f} s elapsed'.format(end - start_frame))
        analyzer.writeHitFrames()
    end = timer()
    print('Run {:04} done: {:.2f} s elapsed'.format(run, end - start))
