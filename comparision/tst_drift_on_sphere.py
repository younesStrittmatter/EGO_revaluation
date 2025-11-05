import psyneulink as pnl
import numpy as np

def main():
    """Compare against reference implementation in nback-paper model (https://github.com/andrebeu/nback-paper)."""

    # PNL DriftOnASphere
    DoS = pnl.DriftOnASphereIntegrator(dimension=5, initializer=np.array([.2] * (4)), noise=0.0)
    results_dos = []
    for i in range(3):
        results_dos.append(DoS(.1))

    # nback-paper implementation
    def spherical_drift(n_steps=3, dim=5, var=0, mean=.1):
        def convert_spherical_to_angular(dim, ros):
            ct = np.zeros(dim)
            ct[0] = np.cos(ros[0])
            prod = np.prod([np.sin(ros[k]) for k in range(1, dim - 1)])
            n_prod = prod
            for j in range(dim - 2):
                n_prod /= np.sin(ros[j + 1])
                amt = n_prod * np.cos(ros[j + 1])
                ct[j + 1] = amt
            ct[dim - 1] = prod
            return ct
        # initialize the spherical coordinates to ensure each context run begins in a new random location on the unit sphere
        ros = np.array([.2] *(dim - 1))
        slen = n_steps
        ctxt = np.zeros((slen, dim))
        for i in range(slen):
            noise = np.random.normal(mean, var, size=(dim - 1)) # add a separately-drawn Gaussian to each spherical coord
            ros += noise
            ctxt[i] = convert_spherical_to_angular(dim, ros)
        return ctxt
    results_sd = spherical_drift()

    np.testing.assert_allclose(np.array(results_dos), np.array(results_sd))

main()