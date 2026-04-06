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


def test_maxwell_static_boundary_fmm_bc_rbc():
    """Compare dense_evaluation and FMM assembly for static Maxwell operators with BC/RBC."""
    if not bempp_cl.api.check_for_fmm():
        pytest.skip("ExaFMM not available.")

    rand = np.random.RandomState(0)
    grid = bempp_cl.api.shapes.regular_sphere(2)
    bc = bempp_cl.api.function_space(grid, "BC", 0)
    rbc = bempp_cl.api.function_space(grid, "RBC", 0)
    vec = rand.rand(bc.global_dof_count)

    for operator in [
        bempp_cl.api.operators.boundary.maxwell.single_layer,
        bempp_cl.api.operators.boundary.maxwell.double_layer,
        bempp_cl.api.operators.boundary.maxwell.hypersingular,
    ]:
        bempp_cl.api.GLOBAL_PARAMETERS.fmm.dense_evaluation = True
        ref = operator(bc, bc, rbc, assembler="fmm").weak_form() @ vec
        bempp_cl.api.clear_fmm_cache()

        bempp_cl.api.GLOBAL_PARAMETERS.fmm.dense_evaluation = False
        fmm = operator(bc, bc, rbc, assembler="fmm").weak_form() @ vec
        bempp_cl.api.clear_fmm_cache()

        np.testing.assert_allclose(ref, fmm, rtol=TOL)


def test_magnetic_field_converges_to_static_double_layer():
    """Verify magnetic field operator converges to static double layer as k->0.

    The magnetic field operator with Helmholtz kernel G_k should converge to
    the static double layer (Laplace kernel) as k->0.
    """
    if not bempp_cl.api.check_for_fmm():
        pytest.skip("ExaFMM not available.")

    rand = np.random.RandomState(0)
    grid = bempp_cl.api.shapes.regular_sphere(3)
    rwg = bempp_cl.api.function_space(grid, "RWG", 0)
    snc = bempp_cl.api.function_space(grid, "SNC", 0)
    vec = rand.rand(rwg.global_dof_count)

    static = bempp_cl.api.operators.boundary.maxwell.double_layer(
        rwg, rwg, snc, assembler="fmm"
    ).weak_form()
    static_result = static @ vec

    for k in [0.1, 0.01, 0.001]:
        magnetic = bempp_cl.api.operators.boundary.maxwell.magnetic_field(
            rwg, rwg, snc, k, assembler="fmm"
        ).weak_form()
        magnetic_result = np.real(magnetic @ vec)
        err = np.linalg.norm(magnetic_result - static_result) / np.linalg.norm(static_result)
        assert err < 2 * k, f"k={k}: relative error {err:.2e} not converging to zero"

    bempp_cl.api.clear_fmm_cache()
