
import numpy as np
import scipy
import scipy.ndimage as ndi
from scipy import fftpack
import logging

def detrend_df_f(A, b, C, f, YrA=None, quantileMin=8, frames_window=500, 
                 flag_auto=True, use_fast=False, detrend_only=False):
    """ Compute DF/F signal without using the original data.
    In general much faster than extract_DF_F

    Args:
        A: scipy.sparse.csc_matrix
            spatial components (from cnmf cnm.A)

        b: ndarray
            spatial background components

        C: ndarray
            temporal components (from cnmf cnm.C)

        f: ndarray
            temporal background components

        YrA: ndarray
            residual signals

        quantile_min: float
            quantile used to estimate the baseline (values in [0,100])
            used only if 'flag_auto' is False, i.e. ignored by default

        frames_window: int
            number of frames for computing running quantile

        flag_auto: bool
            flag for determining quantile automatically

        use_fast: bool
            flag for using approximate fast percentile filtering

        detrend_only: bool (False)
            flag for only subtracting baseline and not normalizing by it.
            Used in 1p data processing where baseline fluorescence cannot be
            determined.

    Returns:
        F_df:
            the computed Calcium activity to the derivative of f
    """

    if C is None:
        logging.warning("There are no components for DF/F extraction!")
        return None
    
    if b is None or f is None:
        b = np.zeros((A.shape[0], 1))
        f = np.zeros((1, C.shape[1]))
        logging.warning("Background components not present. Results should" +
                        " not be interpreted as DF/F normalized but only" +
                        " as detrended.")
        detrend_only = True
    if 'csc_matrix' not in str(type(A)):
        A = scipy.sparse.csc_matrix(A)
    if 'array' not in str(type(b)):
        b = b.toarray()
    if 'array' not in str(type(C)):
        C = C.toarray()
    if 'array' not in str(type(f)):
        f = f.toarray()

    nA = np.sqrt(np.ravel(A.power(2).sum(axis=0)))
    nA_mat = scipy.sparse.spdiags(nA, 0, nA.shape[0], nA.shape[0])
    nA_inv_mat = scipy.sparse.spdiags(1. / nA, 0, nA.shape[0], nA.shape[0])
    A = A * nA_inv_mat
    C = nA_mat * C
    if YrA is not None:
        YrA = nA_mat * YrA

    F = C + YrA if YrA is not None else C
    B = A.T.dot(b).dot(f)
    T = C.shape[-1]

    if flag_auto:
        data_prct, val = df_percentile(F[:, :frames_window], axis=1)
        if frames_window is None or frames_window > T:
            Fd = np.stack([np.percentile(f, prctileMin) for f, prctileMin in
                           zip(F, data_prct)])
            Df = np.stack([np.percentile(f, prctileMin) for f, prctileMin in
                           zip(B, data_prct)])
            if not detrend_only:
                F_df = (F - Fd[:, None]) / (Df[:, None] + Fd[:, None])
            else:
                F_df = F - Fd[:, None]
        else:
            if use_fast:
                Fd = np.stack([fast_prct_filt(f, level=prctileMin,
                                              frames_window=frames_window) for
                               f, prctileMin in zip(F, data_prct)])
                Df = np.stack([fast_prct_filt(f, level=prctileMin,
                                              frames_window=frames_window) for
                               f, prctileMin in zip(B, data_prct)])
            else:
                Fd = np.stack([ndi.percentile_filter(
                    f, prctileMin, (frames_window)) for f, prctileMin in
                    zip(F, data_prct)])
                Df = np.stack([ndi.percentile_filter(
                    f, prctileMin, (frames_window)) for f, prctileMin in
                    zip(B, data_prct)])
            if not detrend_only:
                F_df = (F - Fd) / (Df + Fd)
            else:
                F_df = F - Fd
    else:
        if frames_window is None or frames_window > T:
            Fd = np.percentile(F, quantileMin, axis=1)
            Df = np.percentile(B, quantileMin, axis=1)
            if not detrend_only:
                F_df = (F - Fd[:, None]) / (Df[:, None] + Fd[:, None])
            else:
                F_df = F - Fd[:, None]
        else:
            Fd = ndi.percentile_filter(
                F, quantileMin, (1, frames_window))
            Df = ndi.percentile_filter(
                B, quantileMin, (1, frames_window))
            if not detrend_only:
                F_df = (F - Fd) / (Df + Fd)
            else:
                F_df = F - Fd

    return F_df

def fast_prct_filt(input_data, level=8, frames_window=1000):
    """
    Fast approximate percentage filtering
    """

    data = np.atleast_2d(input_data).copy()
    T = np.shape(data)[-1]
    downsampfact = frames_window

    elm_missing = int(np.ceil(T * 1.0 / downsampfact)
                      * downsampfact - T)
    padbefore = int(np.floor(elm_missing / 2.))
    padafter = int(np.ceil(elm_missing / 2.))
    tr_tmp = np.pad(data.T, ((padbefore, padafter), (0, 0)), mode='reflect')
    numFramesNew, num_traces = np.shape(tr_tmp)
    #% compute baseline quickly

    tr_BL = np.reshape(tr_tmp, (downsampfact, int(numFramesNew / downsampfact),
                                num_traces), order='F')

    tr_BL = np.percentile(tr_BL, level, axis=0)
    tr_BL = ndi.zoom(np.array(tr_BL, dtype=np.float32),
                               [downsampfact, 1], order=3, mode='nearest',
                               cval=0.0, prefilter=True)

    if padafter == 0:
        data -= tr_BL.T
    else:
        data -= tr_BL[padbefore:-padafter].T

    return data.squeeze()


def df_percentile(inputData, axis=None):
    """
    Extracting the percentile of the data where the mode occurs and its value.
    Used to determine the filtering level for DF/F extraction. Note that
    computation can be innacurate for short traces.
    """
    if axis is not None:

        def fnc(x):
            return df_percentile(x)

        result = np.apply_along_axis(fnc, axis, inputData)
        data_prct = result[:, 0]
        val = result[:, 1]
    else:
        # Create the function that we can use for the half-sample mode
        err = True
        while err:
            try:
                bandwidth, mesh, density, cdf = kde(inputData)
                err = False
            except:
                logging.warning('Percentile computation failed. Duplicating ' + 'and trying again.')
                if not isinstance(inputData, list):
                    inputData = inputData.tolist()
                inputData += inputData

        data_prct = cdf[np.argmax(density)] * 100
        val = mesh[np.argmax(density)]
        if data_prct >= 100 or data_prct < 0:
            logging.warning('Invalid percentile computed possibly due ' + 'short trace. Duplicating and recomuputing.')
            if not isinstance(inputData, list):
                inputData = inputData.tolist()
            inputData *= 2
            err = True
        if np.isnan(data_prct):
            logging.warning('NaN percentile computed. Reverting to median.')
            data_prct = 50
            val = np.median(np.array(inputData))

    return data_prct, val


def kde(data, N=None, MIN=None, MAX=None):

    # Parameters to set up the mesh on which to calculate
    N = 2**12 if N is None else int(2**scipy.ceil(scipy.log2(N)))
    if MIN is None or MAX is None:
        minimum = min(data)
        maximum = max(data)
        Range = maximum - minimum
        MIN = minimum - Range / 10 if MIN is None else MIN
        MAX = maximum + Range / 10 if MAX is None else MAX

    # Range of the data
    R = MAX - MIN

    # Histogram the data to get a crude first approximation of the density
    M = len(data)
    DataHist, bins = scipy.histogram(data, bins=N, range=(MIN, MAX))
    DataHist = DataHist / M
    DCTData = fftpack.dct(DataHist, norm=None)

    I = [iN * iN for iN in range(1, N)]
    SqDCTData = (DCTData[1:] / 2)**2

    # The fixed point calculation finds the bandwidth = t_star
    guess = 0.1
    try:
        t_star = scipy.optimize.brentq(fixed_point, 0, guess, args=(M, I, SqDCTData))
    except ValueError:
        print('Oops!')
        return None

    # Smooth the DCTransformed data using t_star
    SmDCTData = DCTData * scipy.exp(-scipy.arange(N)**2 * scipy.pi**2 * t_star / 2)
    # Inverse DCT to get density
    density = fftpack.idct(SmDCTData, norm=None) * N / R
    mesh = [(bins[i] + bins[i + 1]) / 2 for i in range(N)]
    bandwidth = scipy.sqrt(t_star) * R

    density = density / scipy.trapz(density, mesh)
    cdf = np.cumsum(density) * (mesh[1] - mesh[0])

    return bandwidth, mesh, density, cdf


def fixed_point(t, M, I, a2):
    l = 7
    I = scipy.float64(I)
    M = scipy.float64(M)
    a2 = scipy.float64(a2)
    f = 2 * scipy.pi**(2 * l) * scipy.sum(I**l * a2 * scipy.exp(-I * scipy.pi**2 * t))
    for s in range(l, 1, -1):
        K0 = scipy.prod(range(1, 2 * s, 2)) / scipy.sqrt(2 * scipy.pi)
        const = (1 + (1 / 2)**(s + 1 / 2)) / 3
        time = (2 * const * K0 / M / f)**(2 / (3 + 2 * s))
        f = 2 * scipy.pi**(2 * s) * scipy.sum(I**s * a2 * scipy.exp(-I * scipy.pi**2 * time))
    return t - (2 * M * scipy.sqrt(scipy.pi) * f)**(-2 / 5)


def minian_to_caiman(A, b, C, f, Yra) -> tuple:
        """
        Convert the minian data to caiman format

        Parameters
        ----------
        A : xr.DataArray
            Spatial footprints.
        b : xr.DataArray
            Background footprint.
        C : xr.DataArray
            Temporal Calcium Signal.
        f : xr.DataArray
            Temporal Fluorescence Signal.
        Yra : xr.DataArray
            Residuals.
        """
        # Minian data is in the format (unit_id, height, width) -> (height x width, unit_id) for caiman
        A = A.values
        A = np.reshape(A, (A.shape[0], -1), order="F").T
        A = scipy.sparse.csc_matrix(A) # Needs to be converted to csc_matrix for caiman

        C = C.values # Needs no change (unit_id, time) is the same in both

        # Minian data has one estimate of the background, whilst caiman can have several. To make it compatible, we
        # add a new axis to the background and make it the last axis, so the conversion is (height, width) -> (height x width, 1)
        b = b.values
        b = np.expand_dims(b.flatten(order="F"), axis=1)

        # With the aforementioned change, we just need to insert a new axis at the beginning
        f = f.values
        f = np.expand_dims(f, axis=0)

        Yra = Yra.values # Needs no change

        return A, b, C, f, Yra