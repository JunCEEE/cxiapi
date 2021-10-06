#!/usr/bin/env python
"""Tests for `cxiapi` package."""

import pytest
import multiprocessing as mp

from click.testing import CliRunner

from cxiapi import cxiData
from cxiapi import cli

test_file = '/gpfs/exfel/u/scratch/SPB/202130/p900201/spi-comission/vds/r0364.cxi'
calib_folder = '/gpfs/exfel/exp/SPB/202130/p900201/usr/Software/calib/r0361-r0362-r0363/'
geom_file = '/gpfs/exfel/exp/SPB/202130/p900201/usr/Software/geom/agipd_2696_v5.geom'
module = 15


def test_command_line_interface():
    """Test the CLI."""
    runner = CliRunner()
    result = runner.invoke(cli.main)
    assert result.exit_code == 0
    assert 'cxiapi.cli.main' in result.output
    help_result = runner.invoke(cli.main, ['--help'])
    assert help_result.exit_code == 0
    assert '--help  Show this message and exit.' in help_result.output


def test_construct_cxi_data():
    data = cxiData(test_file, verbose=1)


def test_parallell_cxi_data():
    cxi = cxiData(test_file, verbose=1)
    cxi.setCalib(calib_folder)
    cxi.setGeom(geom_file)
    with mp.Pool(2) as pool:
        results = pool.starmap_async(cxi.getCalibrateModule,
                                     [(i, module) for i in range(2)]).get()
    for res in results:
        assert res is not None


def test_parallell_cxi_data_fixed_gain():
    cxi = cxiData(test_file, verbose=1)
    cxi.setCalib(calib_folder)
    cxi.setGeom(geom_file)
    cxi.setGainMode(0)
    with mp.Pool(2) as pool:
        results = pool.starmap_async(cxi.getCalibrateModule,
                                     [(i, module) for i in range(2)]).get()
    for res in results:
        assert res is not None


def test_wrong_fixed_gain():
    cxi = cxiData(test_file, verbose=1)
    with pytest.raises(Exception):
        cxi.setGainMode(5)