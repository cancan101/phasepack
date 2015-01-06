from numpy import exp, log, real

from phasepack.filtergrid import filtergrid 
from phasepack.tools import fft2, ifft2


def monofilt(im, nscale, minWaveLength, mult, sigmaOnf, orientWrap=None):
    if im.dtype not in ['float32', 'float64']:
        im = np.float64(im)
        imgdtype = 'float64'
    else:
        imgdtype = im.dtype

    # Generate horizontal and vertical frequency grids that vary from
    # -0.5 to 0.5 
    radius, u1, u2 = filtergrid(*im.shape)

    radius = radius.astype(imgdtype)
    u1 = u1.astype(imgdtype)
    u2 = u2.astype(imgdtype)

    # Get rid of the 0 radius value in the middle (at top left corner after
    # fftshifting) so that taking the log of the radius, or dividing by the
    # radius, will not cause trouble.
    radius[0, 0] = 1

    # The two monogenic filters in the frequency domain
    H1 = (1j) * u1 / radius
    H2 = (1j) * u2 / radius
    
    IM = fft2(im)

    # The two monogenic filters H1 and H2 are oriented in frequency space
    # but are not selective in terms of the magnitudes of the
    # frequencies.  The code below generates bandpass log-Gabor filters
    # which are point-wise multiplied by H1 and H2 to produce different
    # bandpass versions of H1 and H2

    f = [None] * nscale
    h1f = [None] * nscale
    h2f = [None] * nscale
    A = [None] * nscale
    for s in xrange(nscale):
        wavelength = minWaveLength*mult**(s)
        fo = 1.0/wavelength                  # Centre frequency of filter.
        logGabor = exp((-(log(radius/fo))**2) / (2 * log(sigmaOnf)**2))
        logGabor[0,0] = 0                    # undo the radius fudge.

        # Generate bandpass versions of H1 and H2 at this scale
        H1s = H1 * logGabor
        H2s = H2 * logGabor 

        # Apply filters to image in the frequency domain and get spatial
        # results 
        f[s] = real(ifft2(IM * logGabor))   
        h1f[s] = real(ifft2(IM * H1s))
        h2f[s] = real(ifft2(IM * H2s))

        A[s] = (f[s]**2 + h1f[s]**2 + h2f[s]**2)**0.5  # Magnitude of Energy.
    return f, h1f, h2f, A#, theta, psi
