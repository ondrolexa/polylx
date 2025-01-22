#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
test_polylx
----------------------------------

Tests for `polylx` module.
"""

import numpy as np
from polylx import Sample, circstat
import unittest

sample = Sample.example()


class TestPolylx(unittest.TestCase):
    def setUp(self):
        pass

    def test_grains(self):
        self.assertTrue((len(sample.g.polys) == 701) & (len(sample.g.names) == 3))

    def test_boundaries(self):
        self.assertTrue((len(sample.b) == 1901) & (len(sample.b.names) == 6))

    def test_boundaries_lengths(self):
        eq = np.array(
            [
                np.allclose(sample.b[sample.bids(i)].length.sum(), sample.g[i].length)
                for i in range(len(sample.g))
            ]
        )
        self.assertTrue(len(sample.g[~eq]) == 124)

    def test_grain_read_shapefile(self):
        ex = (
            0.77740719118042412,
            0.048953149577128548,
            3.7836452599329293,
            2.0781454198574365,
        )
        self.assertTrue(np.allclose(sample.g.extent, ex))

    def test_grain_axialratio(self):
        self.assertTrue(np.allclose(sample.g.ar.mean(), 1.8623975420114138))

    def test_grain_area(self):
        self.assertTrue(np.allclose(sample.g.area.sum(), 4.6933456711381556))

    def test_grain_ead(self):
        self.assertTrue(np.allclose(sample.g.ead.mean(), 0.072811839410185208))

    def test_circular_mean(self):
        self.assertTrue(
            np.allclose(circstat.mean(sample.g["pl"].lao), 94.19784708988459)
        )

    def test_paror(self):
        self.assertTrue(
            np.allclose(
                sample.g.paror(normalized=False).mean(axis=1).mean(),
                0.09137254165620362,
            )
        )

    def test_surfor(self):
        self.assertTrue(
            np.allclose(
                sample.g.surfor(normalized=False).mean(axis=1).mean(),
                0.10549790260878265,
            )
        )

    def test_generalize_chaikin(self):
        try:
            sample.g.generalize(method="chaikin")
        except Exception:
            self.fail("Chaikin corner-cutting generalization failed.")

    def test_generalize_vw(self):
        try:
            sample.g.generalize(method="vw")
        except Exception:
            self.fail("Visvalingam-Whyatt generalization failed.")

    def test_generalize_taubin(self):
        try:
            sample.g.generalize(method="taubin")
        except Exception:
            self.fail("Taubin generalization failed.")

    def test_regularization(self):
        try:
            sample.g.regularize()
        except Exception:
            self.fail("Regularization failed.")


if __name__ == "__main__":
    unittest.main()
