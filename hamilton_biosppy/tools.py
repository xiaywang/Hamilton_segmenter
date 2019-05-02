import numpy as np
import utils as utils
import scipy.signal as ss
import six

def find_extrema(signal=None, mode='both'):
    """Locate local extrema points in a signal.

    Based on Fermat's Theorem [Ferm]_.

    Parameters
    ----------
    signal : array
        Input signal.
    mode : str, optional
        Whether to find maxima ('max'), minima ('min'), or both ('both').

    Returns
    -------
    extrema : array
        Indices of the extrama points.
    values : array
        Signal values at the extrema points.

    References
    ----------
    .. [Ferm] Wikipedia, "Fermat's theorem (stationary points)",
       https://en.wikipedia.org/wiki/Fermat%27s_theorem_(stationary_points)

    """

    # check inputs
    if signal is None:
        raise TypeError("Please specify an input signal.")

    if mode not in ['max', 'min', 'both']:
        raise ValueError("Unknwon mode %r." % mode)

    aux = np.diff(np.sign(np.diff(signal)))

    if mode == 'both':
        aux = np.abs(aux)
        extrema = np.nonzero(aux > 0)[0] + 1
    elif mode == 'max':
        extrema = np.nonzero(aux < 0)[0] + 1
    elif mode == 'min':
        extrema = np.nonzero(aux > 0)[0] + 1

    values = signal[extrema]

    return utils.ReturnTuple((extrema, values), ('extrema', 'values'))


def _get_window(kernel, size, **kwargs):
    """Return a window with the specified parameters.

    Parameters
    ----------
    kernel : str
        Type of window to create.
    size : int
        Size of the window.
    ``**kwargs`` : dict, optional
        Additional keyword arguments are passed to the underlying
        scipy.signal.windows function.

    Returns
    -------
    window : array
        Created window.

    """

    # mimics scipy.signal.get_window
    if kernel in ['blackman', 'black', 'blk']:
        winfunc = ss.blackman
    elif kernel in ['triangle', 'triang', 'tri']:
        winfunc = ss.triang
    elif kernel in ['hamming', 'hamm', 'ham']:
        winfunc = ss.hamming
    elif kernel in ['bartlett', 'bart', 'brt']:
        winfunc = ss.bartlett
    elif kernel in ['hanning', 'hann', 'han']:
        winfunc = ss.hann
    elif kernel in ['blackmanharris', 'blackharr', 'bkh']:
        winfunc = ss.blackmanharris
    elif kernel in ['parzen', 'parz', 'par']:
        winfunc = ss.parzen
    elif kernel in ['bohman', 'bman', 'bmn']:
        winfunc = ss.bohman
    elif kernel in ['nuttall', 'nutl', 'nut']:
        winfunc = ss.nuttall
    elif kernel in ['barthann', 'brthan', 'bth']:
        winfunc = ss.barthann
    elif kernel in ['flattop', 'flat', 'flt']:
        winfunc = ss.flattop
    elif kernel in ['kaiser', 'ksr']:
        winfunc = ss.kaiser
    elif kernel in ['gaussian', 'gauss', 'gss']:
        winfunc = ss.gaussian
    elif kernel in ['general gaussian', 'general_gaussian', 'general gauss',
                    'general_gauss', 'ggs']:
        winfunc = ss.general_gaussian
    elif kernel in ['boxcar', 'box', 'ones', 'rect', 'rectangular']:
        winfunc = ss.boxcar
    elif kernel in ['slepian', 'slep', 'optimal', 'dpss', 'dss']:
        winfunc = ss.slepian
    elif kernel in ['cosine', 'halfcosine']:
        winfunc = ss.cosine
    elif kernel in ['chebwin', 'cheb']:
        winfunc = ss.chebwin
    else:
        raise ValueError("Unknown window type.")

    try:
        window = winfunc(size, **kwargs)
    except TypeError as e:
        raise TypeError("Invalid window arguments: %s." % e)

    return window


def smoother(signal=None, kernel='boxzen', size=10, mirror=True, **kwargs):
    """Smooth a signal using an N-point moving average [MAvg]_ filter.

    This implementation uses the convolution of a filter kernel with the input
    signal to compute the smoothed signal [Smit97]_.

    Availabel kernels: median, boxzen, boxcar, triang, blackman, hamming, hann,
    bartlett, flattop, parzen, bohman, blackmanharris, nuttall, barthann,
    kaiser (needs beta), gaussian (needs std), general_gaussian (needs power,
    width), slepian (needs width), chebwin (needs attenuation).

    Parameters
    ----------
    signal : array
        Signal to smooth.
    kernel : str, array, optional
        Type of kernel to use; if array, use directly as the kernel.
    size : int, optional
        Size of the kernel; ignored if kernel is an array.
    mirror : bool, optional
        If True, signal edges are extended to avoid boundary effects.
    ``**kwargs`` : dict, optional
        Additional keyword arguments are passed to the underlying
        scipy.signal.windows function.

    Returns
    -------
    signal : array
        Smoothed signal.
    params : dict
        Smoother parameters.

    Notes
    -----
    * When the kernel is 'median', mirror is ignored.

    References
    ----------
    .. [MAvg] Wikipedia, "Moving Average",
       http://en.wikipedia.org/wiki/Moving_average
    .. [Smit97] S. W. Smith, "Moving Average Filters - Implementation by
       Convolution", http://www.dspguide.com/ch15/1.htm, 1997

    """

    # check inputs
    if signal is None:
        raise TypeError("Please specify a signal to smooth.")

    length = len(signal)

    if isinstance(kernel, six.string_types):
        # check length
        if size > length:
            size = length - 1

        if size < 1:
            size = 1

        if kernel == 'boxzen':
            # hybrid method
            # 1st pass - boxcar kernel
            aux, _ = smoother(signal,
                              kernel='boxcar',
                              size=size,
                              mirror=mirror)

            # 2nd pass - parzen kernel
            smoothed, _ = smoother(aux,
                                   kernel='parzen',
                                   size=size,
                                   mirror=mirror)

            params = {'kernel': kernel, 'size': size, 'mirror': mirror}

            args = (smoothed, params)
            names = ('signal', 'params')

            return utils.ReturnTuple(args, names)

        elif kernel == 'median':
            # median filter
            if size % 2 == 0:
                raise ValueError(
                    "When the kernel is 'median', size must be odd.")

            smoothed = ss.medfilt(signal, kernel_size=size)

            params = {'kernel': kernel, 'size': size, 'mirror': mirror}

            args = (smoothed, params)
            names = ('signal', 'params')

            return utils.ReturnTuple(args, names)

        else:
            win = _get_window(kernel, size, **kwargs)

    elif isinstance(kernel, np.ndarray):
        win = kernel
        size = len(win)

        # check length
        if size > length:
            raise ValueError("Kernel size is bigger than signal length.")

        if size < 1:
            raise ValueError("Kernel size is smaller than 1.")

    else:
        raise TypeError("Unknown kernel type.")

    # convolve
    w = win / win.sum()
    if mirror:
        aux = np.concatenate(
            (signal[0] * np.ones(size), signal, signal[-1] * np.ones(size)))
        smoothed = np.convolve(w, aux, mode='same')
        smoothed = smoothed[size:-size]
    else:
        smoothed = np.convolve(w, signal, mode='same')

    # output
    params = {'kernel': kernel, 'size': size, 'mirror': mirror}
    params.update(kwargs)

    args = (smoothed, params)
    names = ('signal', 'params')

    return utils.ReturnTuple(args, names)


def _filter_signal(b, a, signal, zi=None, check_phase=True, **kwargs):
    """Filter a signal with given coefficients.

    Parameters
    ----------
    b : array
        Numerator coefficients.
    a : array
        Denominator coefficients.
    signal : array
        Signal to filter.
    zi : array, optional
        Initial filter state.
    check_phase : bool, optional
        If True, use the forward-backward technique.
    ``**kwargs`` : dict, optional
        Additional keyword arguments are passed to the underlying filtering
        function.

    Returns
    -------
    filtered : array
        Filtered signal.
    zf : array
        Final filter state.

    Notes
    -----
    * If check_phase is True, zi cannot be set.

    """

    # check inputs
    if check_phase and zi is not None:
        raise ValueError(
            "Incompatible arguments: initial filter state cannot be set when \
            check_phase is True.")

    if zi is None:
        zf = None
        if check_phase:
            filtered = ss.filtfilt(b, a, signal, **kwargs)
        else:
            filtered = ss.lfilter(b, a, signal, **kwargs)
    else:
        filtered, zf = ss.lfilter(b, a, signal, zi=zi, **kwargs)

    return filtered, zf


def _norm_freq(frequency=None, sampling_rate=1000.):
    """Normalize frequency to Nyquist Frequency (Fs/2).

    Parameters
    ----------
    frequency : int, float, list, array
        Frequencies to normalize.
    sampling_rate : int, float, optional
        Sampling frequency (Hz).

    Returns
    -------
    wn : float, array
        Normalized frequencies.

    """

    # check inputs
    if frequency is None:
        raise TypeError("Please specify a frequency to normalize.")

    # convert inputs to correct representation
    try:
        frequency = float(frequency)
    except TypeError:
        # maybe frequency is a list or array
        frequency = np.array(frequency, dtype='float')

    Fs = float(sampling_rate)

    wn = 2. * frequency / Fs

    return wn


def get_filter(ftype='FIR',
               band='lowpass',
               order=None,
               frequency=None,
               sampling_rate=1000., **kwargs):
    """Compute digital (FIR or IIR) filter coefficients with the given
    parameters.

    Parameters
    ----------
    ftype : str
        Filter type:
            * Finite Impulse Response filter ('FIR');
            * Butterworth filter ('butter');
            * Chebyshev filters ('cheby1', 'cheby2');
            * Elliptic filter ('ellip');
            * Bessel filter ('bessel').
    band : str
        Band type:
            * Low-pass filter ('lowpass');
            * High-pass filter ('highpass');
            * Band-pass filter ('bandpass');
            * Band-stop filter ('bandstop').
    order : int
        Order of the filter.
    frequency : int, float, list, array
        Cutoff frequencies; format depends on type of band:
            * 'lowpass' or 'bandpass': single frequency;
            * 'bandpass' or 'bandstop': pair of frequencies.
    sampling_rate : int, float, optional
        Sampling frequency (Hz).
    ``**kwargs`` : dict, optional
        Additional keyword arguments are passed to the underlying
        scipy.signal function.

    Returns
    -------
    b : array
        Numerator coefficients.
    a : array
        Denominator coefficients.

    See Also:
        scipy.signal

    """

    # check inputs
    if order is None:
        raise TypeError("Please specify the filter order.")
    if frequency is None:
        raise TypeError("Please specify the cutoff frequency.")
    if band not in ['lowpass', 'highpass', 'bandpass', 'bandstop']:
        raise ValueError(
            "Unknown filter type '%r'; choose 'lowpass', 'highpass', \
            'bandpass', or 'bandstop'."
            % band)

    # convert frequencies
    frequency = _norm_freq(frequency, sampling_rate)

    # get coeffs
    b, a = [], []
    if ftype == 'FIR':
        # FIR filter
        if order % 2 == 0:
            order += 1
        a = np.array([1])
        if band in ['lowpass', 'bandstop']:
            b = ss.firwin(numtaps=order,
                          cutoff=frequency,
                          pass_zero=True, **kwargs)
        elif band in ['highpass', 'bandpass']:
            b = ss.firwin(numtaps=order,
                          cutoff=frequency,
                          pass_zero=False, **kwargs)
    elif ftype == 'butter':
        # Butterworth filter
        b, a = ss.butter(N=order,
                         Wn=frequency,
                         btype=band,
                         analog=False,
                         output='ba', **kwargs)
    elif ftype == 'cheby1':
        # Chebyshev type I filter
        b, a = ss.cheby1(N=order,
                         Wn=frequency,
                         btype=band,
                         analog=False,
                         output='ba', **kwargs)
    elif ftype == 'cheby2':
        # chevyshev type II filter
        b, a = ss.cheby2(N=order,
                         Wn=frequency,
                         btype=band,
                         analog=False,
                         output='ba', **kwargs)
    elif ftype == 'ellip':
        # Elliptic filter
        b, a = ss.ellip(N=order,
                        Wn=frequency,
                        btype=band,
                        analog=False,
                        output='ba', **kwargs)
    elif ftype == 'bessel':
        # Bessel filter
        b, a = ss.bessel(N=order,
                         Wn=frequency,
                         btype=band,
                         analog=False,
                         output='ba', **kwargs)

    return utils.ReturnTuple((b, a), ('b', 'a'))


def filter_signal(signal=None,
                  ftype='FIR',
                  band='lowpass',
                  order=None,
                  frequency=None,
                  sampling_rate=1000., **kwargs):
    """Filter a signal according to the given parameters.

    Parameters
    ----------
    signal : array
        Signal to filter.
    ftype : str
        Filter type:
            * Finite Impulse Response filter ('FIR');
            * Butterworth filter ('butter');
            * Chebyshev filters ('cheby1', 'cheby2');
            * Elliptic filter ('ellip');
            * Bessel filter ('bessel').
    band : str
        Band type:
            * Low-pass filter ('lowpass');
            * High-pass filter ('highpass');
            * Band-pass filter ('bandpass');
            * Band-stop filter ('bandstop').
    order : int
        Order of the filter.
    frequency : int, float, list, array
        Cutoff frequencies; format depends on type of band:
            * 'lowpass' or 'bandpass': single frequency;
            * 'bandpass' or 'bandstop': pair of frequencies.
    sampling_rate : int, float, optional
        Sampling frequency (Hz).
    ``**kwargs`` : dict, optional
        Additional keyword arguments are passed to the underlying
        scipy.signal function.

    Returns
    -------
    signal : array
        Filtered signal.
    sampling_rate : float
        Sampling frequency (Hz).
    params : dict
        Filter parameters.

    Notes
    -----
    * Uses a forward-backward filter implementation. Therefore, the combined
      filter has linear phase.

    """

    # check inputs
    if signal is None:
        raise TypeError("Please specify a signal to filter.")

    # get filter
    b, a = get_filter(ftype=ftype,
                      order=order,
                      frequency=frequency,
                      sampling_rate=sampling_rate,
                      band=band, **kwargs)

    # filter
    filtered, _ = _filter_signal(b, a, signal, check_phase=True)

    # output
    params = {
        'ftype': ftype,
        'order': order,
        'frequency': frequency,
        'band': band,
    }
    params.update(kwargs)

    args = (filtered, sampling_rate, params)
    names = ('signal', 'sampling_rate', 'params')

    return utils.ReturnTuple(args, names)
