"""
2D Chebyshev wavelength solution (pixel, order) fit to a gas-cell spectrum
against an FTS reference. NumPy/SciPy + numba, hand-written Jacobian.

The wavelength solution uses the same convention as the LFC solution
(see `fit_wavesol_chebyshev2d`): m * lam is a 2D Chebyshev series in the
normalized pixel and normalized order, with coefficients of shape (Ncpix, Ncord).

    lam(m, s)   = (1/m) sum_ij c_ij T_i(s_n) T_j(m_n)        (linear in c), s on the oversampled pixel grid
    F_hi(m, s)  = F_ref(lam(m, s))                           (linear interpolation; F_ref is a flux density)
    conv(m, x)  = (chromatic LSF kernels * F_hi)(m, x*osamp + osamp//2)
    cont(m, x)  = sum_q a_mq T_q(x_n)                        (optional; a per-order Chebyshev in pixel)
    model       = cont * conv
    resid       = (flux - model) / sigma

Parameter vector p = [c.ravel(), a.ravel()]  (a omitted when continuum_deg_intra_order is None), with a of shape (O, Q).

The convolution is linear and its kernels do not depend on p, so
    d conv / d c_ij = (T_j(m) / m) * (kernels * [F'(lam) T_i(s)])
i.e. the same convolution applied to the derivative of the interpolant times T_i.
"""
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from numba import njit, prange
from numpy.polynomial import chebyshev as C
from scipy.optimize import least_squares

from ..lsf.lsf_convolve import convolve_numba_1d_chromatic
from ..utils.array_utils import solve_chebyshev2d_ols


def _to_cheb_domain(v, lo, hi):
    return 2.0 * (v - lo) / (hi - lo) - 1.0


# ---------------------------------------------------------------------------
# numba kernels
# ---------------------------------------------------------------------------
@njit(parallel=True, cache=True)
def _interp_reference_numba(A, Txs, xp, fp, F_hi, G_hi, want_G):
    """
    lam[o, s] = sum_i A[o, i] Txs[s, i];  F_hi[o, s] = F(lam)  (linear interpolation, edge-clamped)
    If want_G: G_hi[o, i, s] = F'(lam) * Txs[s, i]   (F' = slope of the linear interpolant)
    """
    O, Ncpix = A.shape
    S = Txs.shape[0]
    n = xp.shape[0]
    for o in prange(O):
        for s in range(S):
            lam = 0.0
            for i in range(Ncpix):
                lam += A[o, i] * Txs[s, i]
            if lam <= xp[0]:
                f = fp[0]
                slope = 0.0
            elif lam >= xp[n - 1]:
                f = fp[n - 1]
                slope = 0.0
            else:
                k = np.searchsorted(xp, lam) - 1
                slope = (fp[k + 1] - fp[k]) / (xp[k + 1] - xp[k])
                f = fp[k] + slope * (lam - xp[k])
            F_hi[o, s] = f
            if want_G:
                for i in range(Ncpix):
                    G_hi[o, i, s] = slope * Txs[s, i]


@njit(parallel=True, cache=True)
def _fill_jacobian_numba(J, B, Tm, cont, conv, isig, Tc, nc, Q):
    """Fill J (O*X, nc + O*Q) in place. J must be zero-initialised."""
    O, X, Ncpix = B.shape
    Ncord = Tm.shape[1]
    for o in prange(O):
        for x in range(X):
            r = o * X + x
            sc = -isig[o, x] * cont[o, x]
            for i in range(Ncpix):
                for j in range(Ncord):
                    J[r, i * Ncord + j] = sc * Tm[o, j] * B[o, x, i]
            sa = -isig[o, x] * conv[o, x]
            for q in range(Q):
                J[r, nc + o * Q + q] = sa * Tc[x, q]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def make_fit_context(orders, flux, sigma, lam_ref, F_ref, kernels, osamp,
                     deg_intra_order, deg_inter_order, dq=None, continuum_deg_intra_order=None):
    """Precompute everything that does not depend on the parameters.

    orders  : (O,)    echelle order numbers
    flux    : (O, X)  observed gas-cell spectrum (may be NaN where dq > 0)
    sigma   : (O, X)  1-sigma uncertainties (may be 0 or NaN where dq > 0)
    lam_ref : (R,)    FTS wavelength grid in microns, strictly increasing
    F_ref   : (R,)    FTS transmission on that grid, a flux density (per unit wavelength),
                      so interpolating it at arbitrary wavelengths is valid
    kernels : (O, X*osamp, Ks)  chromatic LSF, one kernel row per oversampled sample (Ks odd),
                      in the layout expected by convolve_numba_1d_chromatic
    osamp   : int     oversampling factor of the pixel grid (odd). Oversampled sample s sits at pixel
                      coordinate (s - osamp//2) / osamp, so sample p*osamp + osamp//2 is the centre of pixel p.
    deg_intra_order : degree of the Chebyshev series in pixel
    deg_inter_order : degree of the Chebyshev series in order
    dq      : (O, X)  data quality, only dq == 0 is used; None means every pixel is good.
                      Pixels with non-finite flux or sigma, or sigma <= 0, are also masked.
    continuum_deg_intra_order : None for no continuum, else degree of the continuum polynomial in pixel,
                      an independent polynomial for each order

    The normalization bounds follow the LFC solution: the data range padded by 10 on each side,
    i.e. pixel_bounds = (-10, X - 1 + 10) and order_bounds = (min(orders) - 10, max(orders) + 10).
    """
    orders = np.asarray(orders)
    O, X = flux.shape
    if osamp % 2 == 0:
        raise ValueError(f"osamp must be odd so a sample sits on each pixel centre; got {osamp}")
    S = X * osamp
    if kernels.shape[:2] != (O, S):
        raise ValueError(f"kernels must have shape (O, X*osamp, Ks) = ({O}, {S}, Ks); got {kernels.shape}")

    good = np.isfinite(flux) & np.isfinite(sigma) & (sigma > 0)
    if dq is not None:
        good &= dq == 0
    isig = np.zeros((O, X))
    np.divide(1.0, sigma, out=isig, where=good)  # 1/sigma on good pixels, exactly 0 elsewhere
    flux = np.where(good, flux, 0.0)             # a NaN in a masked pixel would give NaN * 0 = NaN

    pix = np.arange(X)
    pix_s = (np.arange(S) - osamp // 2) / osamp
    pixel_bounds = (-10.0, X - 1 + 10.0)
    order_bounds = (int(orders.min()) - 10, int(orders.max()) + 10)
    Q = 0 if continuum_deg_intra_order is None else continuum_deg_intra_order + 1
    return dict(
        O=O, X=X, S=S, osamp=osamp, shape=(deg_intra_order + 1, deg_inter_order + 1),
        nc=(deg_intra_order + 1) * (deg_inter_order + 1), Q=Q,
        cont_shape=(O, Q),
        orders=orders, pixel_bounds=pixel_bounds, order_bounds=order_bounds,
        flux=flux, isig=isig, n_valid=np.count_nonzero(isig),
        xp=lam_ref, fp=F_ref, kernels=kernels,
        dec=slice(osamp // 2, None, osamp),                                          # oversampled -> pixel centres
        pool=ThreadPoolExecutor(),                                                   # convolve is nogil
        Tm=C.chebvander(_to_cheb_domain(orders, *order_bounds), deg_inter_order) / orders[:, None],  # (O, Ncord), includes 1/m
        Txs=C.chebvander(_to_cheb_domain(pix_s, *pixel_bounds), deg_intra_order),    # (S, Ncpix)
        Tx0=C.chebvander(_to_cheb_domain(pix, *pixel_bounds), deg_intra_order),      # (X, Ncpix)
        Tc=C.chebvander(_to_cheb_domain(pix, *pixel_bounds), continuum_deg_intra_order) if Q else np.zeros((X, 0)),
    )


def coeffs_from_wavelength_grid(wave, ctx):
    """Least-squares Chebyshev coefficients (Ncpix, Ncord) of an (O, X) wavelength grid, in the basis of ctx.

    Non-finite wavelengths are ignored. Use this to turn an initial wavelength solution into c0.
    """
    order_grid = np.broadcast_to(ctx["orders"][:, None], wave.shape)
    pix_grid = np.broadcast_to(np.arange(ctx["X"])[None, :], wave.shape)
    ok = np.isfinite(wave)
    return solve_chebyshev2d_ols(
        _to_cheb_domain(pix_grid[ok], *ctx["pixel_bounds"]),
        _to_cheb_domain(order_grid[ok], *ctx["order_bounds"]),
        order_grid[ok] * wave[ok],
        ctx["shape"][0] - 1, ctx["shape"][1] - 1,
    )


def _unpack_params(p, ctx):
    c = p[:ctx["nc"]].reshape(ctx["shape"])
    a = p[ctx["nc"]:].reshape(ctx["cont_shape"])
    return c, a


def evaluate_wavelength_solution(cmat, ctx):
    """lam(m, x) on the full (O, X) pixel grid."""
    return (ctx["Tm"] @ cmat.T) @ ctx["Tx0"].T


def build_continuum_1d(amat, pix, pixel_bounds):
    """Best-fit continuum (Norders, Npix) from its Chebyshev coefficients, without a fit context.

    amat         : (Norders, continuum_deg_intra_order + 1)  per-order continuum coefficients, e.g. fit_results['continuum_coeffs']
    pix          : (Npix,)  pixel coordinates to evaluate at, e.g. np.arange(Ncols)
    pixel_bounds : (min, max)  pixel normalization of the fit, e.g. fit_results['pixel_bounds']

    Each order's continuum is its own Chebyshev series in the normalized pixel.
    """
    amat = np.asarray(amat, dtype=np.float64)
    Tc = C.chebvander(_to_cheb_domain(np.asarray(pix, dtype=np.float64), *pixel_bounds), amat.shape[1] - 1)
    return amat @ Tc.T


def evaluate_continuum(amat, ctx):
    """Continuum (O, X) on the fit's pixel grid; ones without a continuum."""
    if not ctx["Q"]:
        return np.ones((ctx["O"], ctx["X"]))
    return amat @ ctx["Tc"].T


def evaluate_forward_model(cmat, amat, ctx):
    """cont * conv on the full (O, X) pixel grid. amat is ignored without a continuum."""
    return evaluate_continuum(amat, ctx) * _lsf_convolved_reference(cmat, ctx)[0]


def _lsf_convolved_reference(cmat, ctx, want_B=False):
    """Returns conv (O, X) and, if want_B, B (O, X, Ncpix) = d conv / d (lam coefficients, without the T_j(m)/m factor)."""
    O, X, S, Ncpix = ctx["O"], ctx["X"], ctx["S"], ctx["shape"][0]
    kernels, dec = ctx["kernels"], ctx["dec"]

    F_hi = np.empty((O, S))
    G_hi = np.empty((O, Ncpix, S)) if want_B else np.empty((1, 1, 1))
    _interp_reference_numba(ctx["Tm"] @ cmat.T, ctx["Txs"], ctx["xp"], ctx["fp"], F_hi, G_hi, want_B)

    conv = np.empty((O, X))
    B = np.empty((O, X, Ncpix)) if want_B else np.empty((1, 1, 1))

    def work(o):
        conv[o] = convolve_numba_1d_chromatic(F_hi[o], kernels[o])[dec]
        if want_B:
            for i in range(Ncpix):
                B[o, :, i] = convolve_numba_1d_chromatic(G_hi[o, i], kernels[o])[dec]

    list(ctx["pool"].map(work, range(O)))
    return conv, B


def _initial_continuum(cmat, ctx):
    """Weighted linear least-squares continuum (O, Q) for fixed lam coefficients, one fit per order."""
    conv = _lsf_convolved_reference(cmat, ctx)[0]
    amat = np.zeros((ctx["O"], ctx["Q"]))
    for o in range(ctx["O"]):
        w = ctx["isig"][o]
        if not np.any(w):
            amat[o, 0] = 1.0
            continue
        V = ctx["Tc"] * (conv[o] * w)[:, None]
        amat[o], *_ = np.linalg.lstsq(V, ctx["flux"][o] * w, rcond=None)
    return amat


def _forward_model(p, ctx):
    c, a = _unpack_params(p, ctx)
    return evaluate_forward_model(c, a, ctx)


def _normalized_residuals(p, ctx):
    return ((ctx["flux"] - _forward_model(p, ctx)) * ctx["isig"]).ravel()


def _residuals_jacobian(p, ctx):
    c, a = _unpack_params(p, ctx)
    conv, B = _lsf_convolved_reference(c, ctx, want_B=True)
    J = np.zeros((ctx["O"] * ctx["X"], ctx["nc"] + a.size))
    _fill_jacobian_numba(J, B, ctx["Tm"], evaluate_continuum(a, ctx), conv, ctx["isig"], ctx["Tc"], ctx["nc"], ctx["Q"])
    return J


def fit_wavelength_solution(ctx, c0, a0=None, max_nfev=50, ftol=1e-9):
    """Levenberg-Marquardt fit with scipy.optimize.least_squares (method='lm') and the analytic Jacobian.

    Parameters are scaled by the Jacobian column norms (x_scale='jac'), as in MINPACK.
    If a0 is None, the continuum starts from a linear least-squares fit given c0.
    max_nfev and ftol are passed to least_squares.
    Returns (cmat, amat, cov, chi2_red, info). amat is None without a continuum.
    """
    p0 = c0.ravel()
    if ctx["Q"]:
        if a0 is None:
            a0 = _initial_continuum(c0, ctx)
        p0 = np.concatenate([p0, a0.ravel()])

    res = least_squares(
        _normalized_residuals, p0, jac=_residuals_jacobian, args=(ctx,),
        method="lm", x_scale="jac", ftol=ftol, max_nfev=max_nfev,
    )
    cov = np.linalg.inv(res.jac.T @ res.jac)
    chi2_red = 2.0 * res.cost / (ctx["n_valid"] - res.x.size)
    c, a = _unpack_params(res.x, ctx)
    info = {
        "nfev": res.nfev, "njev": res.njev, "cost": res.cost,
        "status": res.status, "message": res.message, "converged": res.success,
    }
    return c, (a if ctx["Q"] else None), cov * chi2_red, chi2_red, info
