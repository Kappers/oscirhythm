'''
Pytorch based single-layer GrFNN, i.e. gradient frequency neural oscillator array.

The most useful reference for this was the flagship GrFNNToolbox (MATLAB):
    https://github.com/MusicDynamicsLab/GrFNNToolbox

For some arithmetic, this was a very useful reference using the largely inactive pybrain project:
    https://github.com/andyr0id/PyGFNN/tree/master/pygfnn

@author T. Kaplan
'''
import torch
import numpy as np
import attr
import functools
import enum

def rk4(z0, h, fn_dz):
    ''' 4th order Runge-Kutta solution to differential '''
    k1 = h * fn_dz(z0)
    k2 = h * fn_dz(z0+(k1/2))
    k3 = h * fn_dz(z0+(k2/2))
    k4 = h * fn_dz(z0+k3)
    k0 = z0 + (k1 + 2*k2 + 2*k3 + k4)/6
    return k0

def spont_amp(a, b1, b2, e):
    ''' Spontaneuous amplitude of the canonical model '''

    def _slope(r, a, b1, b2, e):
        return a + 3*b1*np.power(r, 2) +\
            (5*e*b2*np.power(r, 4) - 3*(e**2)*b2*np.power(r, 6))\
                                        / (np.power((1 - e*np.power(r, 2)), 2))


    if b2 == 0 and e != 0:
        e = 0
    r = np.roots([e*(b2 - b1), 0, b1 - e*a, 0, a, 0])
    r = [x for x in r if np.abs(np.imag(x)) < np.spacing(1)]
    r = np.real(np.unique(r))
    r = [x for x in r if x >= 0]
    if b2 != 0:
        r = [x for x in r if x < 1/np.sqrt(e)]
    sl1 = _slope(r, a, b1, b2, e)
    ind1 = [i for i in range(len(sl1)) if sl1[i] < 0]
    ind2a = [i for i in range(len(sl1)) if sl1[i] == 0]
    sl2b = _slope(r-np.spacing(1), a, b1, b2, e)
    ind2b = [i for i in range(len(sl2b)) if sl2b[i] < 0]
    sl2c = _slope(r+np.spacing(1), a, b1, b2, e)
    ind2c = [i for i in range(len(sl2b)) if sl2c[i] < 0]
    ind2 = np.intersect1d(ind2a, np.intersect1d(ind2b, ind2c))
    ind = np.concatenate((ind1, ind2)).astype('int')
    r = np.array(r)
    r = r[ind]
    return r.sort()[::-1] if len(ind) > 1 else np.array([r])

class FrequencyType(enum.Enum):
    ''' Types of frequency gradients that can be used for FrequencyDist '''
    LINEAR = 'linear'
    LOG = 'log'

@attr.s(slots=True)
class FrequencyDist(object):
    ''' Distribution/Gradient of frequencies, encapsulating useful state for plotting/analytics,
    and simplifying initialisation of GrFNN '''
    min_freq = attr.ib()
    max_freq = attr.ib()
    dim = attr.ib()
    ftype = attr.ib()
    dist = attr.ib(default=None)

    def __attrs_post_init__(self):
        # This crude hook allows us to run post initialisation code, depsite attr syntax sugar -
        # and this simply expands the gradient using chosen distribution.
        if self.ftype == FrequencyType.LINEAR:
            ftype_fn = np.linspace
        elif self.ftype == FrequencyType.LOG:
            ftype_fn = lambda minf, maxf, dim: np.logspace(np.log10(minf), np.log10(maxf), dim)
        else:
            raise NotImplementedError('Unhandled frequency type: {}'.format(self.ftype))
        self.dist = ftype_fn(self.min_freq, self.max_freq, self.dim)

@attr.s(slots=True)
class ZParams(object):
    ''' Various parameters for our GrFNN model '''
    alpha = attr.ib(default=0.0) # Dampening
    beta1 = attr.ib(default=-1.0) # Amplitude compression factor(s)
    beta2 = attr.ib(default=-1.0)
    delta1 = attr.ib(default=0.0)
    delta2 = attr.ib(default=0.0)
    epsilon = attr.ib(default=1.0) # Scale factor/coupling strength
    w = attr.ib(default=None) # Natural oscillation frequency

    @property
    def epsilon_c(self):
        return np.complex64(self.epsilon)

    @property
    def root_e(self):
        return np.sqrt(self.epsilon_c)

    def tune_to_freq(self, freq_dist):
        ''' Initialises parameters according to a given frequency distribution in use '''
        if freq_dist.ftype == FrequencyType.LINEAR:
            self.alpha += (1j * 2 * np.pi * freq_dist.dist)
            self.beta1 += (1j * self.delta1)
            self.beta2 += (1j * self.delta2)
            self.w = 1.
        elif freq_dist.ftype == FrequencyType.LOG:
            self.alpha += (1j * 2 * np.pi)
            self.alpha *= freq_dist.dist
            self.beta1 += (1j * self.delta1)
            self.beta1 *= freq_dist.dist
            self.beta2 += (1j * self.delta2)
            self.beta2 *= freq_dist.dist
            self.w = freq_dist.dist
        else:
            raise NotImplementedError('Unhandled frequency type: {}'.format(freq_dist.ftype))

    def ext_stimulus(self, z, w, extin):
        return w * (extin/(1-self.root_e*extin)) * (1./(1-(self.root_e*np.conj(z))))

    def zdot(self, external, z):
        nl1 = self.beta1 * np.abs(z) ** 2
        nl2 = self.beta2 * self.epsilon_c * (np.abs(z) ** 4)
        nl2 /= (1 - self.epsilon_c * (np.abs(z) ** 2))
        z_new = z * (self.alpha + nl1 + nl2) + external
        return z_new


class GrFNN(torch.nn.Module):
    ''' Single-layer GrFNN that can be used alongside other torch NN modules '''

    def __init__(self, freq_dist, zparams, fs=40.0, w=1.0):
        super(GrFNN, self).__init__()
        self.set_state(freq_dist, zparams, fs, w)

    def set_state(self, freq_dist, zparams, fs=40.0, w=1.0):
        self.dt = 1.0/fs
        self.freq_dist = freq_dist

        self.zparams = zparams
        self.zparams.tune_to_freq(freq_dist)

        # Initially set model spontaneuous amplitudes randomly
        self.z = np.zeros(self.freq_dist.dim, dtype=np.complex64)
        self.set_random_spont_amp()

    def set_random_spont_amp(self):

        def _try_get_first(param):
            # However zparams are scaled, make sure something is returned
            try:
                return param[0]
            except TypeError:
                return param*self.freq_dist.dist[0]

        r = spont_amp(np.real(_try_get_first(self.zparams.alpha)),
                      np.real(_try_get_first(self.zparams.beta1)),
                      np.real(_try_get_first(self.zparams.beta2)),
                      self.zparams.epsilon_c)
        self.z = r[-1] + self.z
        self.z = self.z + .01 * np.random.randn(self.freq_dist.dim)
        phi0 = 2 * np.pi * np.random.randn(self.freq_dist.dim)
        self.z = np.complex128(self.z * np.exp(1j * 2 * np.pi * phi0))

    def forward(self, x):
        ''' This is the single-step propagation function '''
        ext_in = self.zparams.ext_stimulus(self.z, self.zparams.w, x)
        zdot_ext = functools.partial(self.zparams.zdot, ext_in)
        self.z = rk4(self.z, self.dt, zdot_ext)
        return self.z
