"""Test static Maxwell operators: dense vs FMM assembler comparison."""

import pytest
import numpy as np
import bempp_cl.api

pytestmark = pytest.mark.usefixtures("default_parameters", "helpers")

TOL = 2e-3
bempp_cl.api.GLOBAL_PARAMETERS.fmm.expansion_order = 10


def test_maxwell_static_boundary_fmm_dense():
    """Compare dense and FMM assembly for static Maxwell operators with RWG/SNC."""
    if not bempp_cl.api.check_for_fmm():
        pytest.skip("ExaFMM not available.")

    rand = np.random.RandomState(0)
    grid = bempp_cl.api.shapes.regular_sphere(2)
    rwg = bempp_cl.api.function_space(grid, "RWG", 0)
    snc = bempp_cl.api.function_space(grid, "SNC", 0)
    vec = rand.rand(rwg.global_dof_count)

    for operator in [
        bempp_cl.api.operators.boundary.maxwell.single_layer,
        bempp_cl.api.operators.boundary.maxwell.double_layer,
        bempp_cl.api.operators.boundary.maxwell.hypersingular,
    ]:
        dense = operator(rwg, rwg, snc, assembler="dense").weak_form()
        fmm = operator(rwg, rwg, snc, assembler="fmm").weak_form()

        np.testing.assert_allclose(dense @ vec, fmm @ vec, rtol=TOL)

    bempp_cl.api.clear_fmm_cache()
