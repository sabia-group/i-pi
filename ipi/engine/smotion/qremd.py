import numpy as np
import time

from ipi.engine.smotion import Smotion
from ipi.engine.ensembles import ensemble_swap
from ipi.utils.depend import dstrip
from ipi.utils.messages import verbosity, info
from ipi.utils.units import Constants
import itertools

__all__ = ["QReplicaExchange"]

def thermo_scale(thermo, scale):
    if hasattr(thermo, "tlist"):
        for t in thermo.tlist:
            thermo_scale(t, scale)
    if hasattr(thermo, "s"):
        thermo.s *= scale

def motion_scale(motion, scale):
    if hasattr(motion, "mlist"):
        for m in motion.mlist:
            motion_scale(m, scale)
    if hasattr(motion, "thermostat"):
        thermo_scale(motion.thermostat, scale)
    if hasattr(motion, "barostat"):
        thermo_scale(motion.barostat.thermostat, scale)

def gle_scale(sys, scale):
    motion_scale(sys.motion, scale)

class QReplicaExchange(Smotion):
    """quantum replica exchange (QREMD)."""

    def __init__(self, stride=1.0, repindex=None, krescale=True, swapfile="PARATEMP"):
        super().__init__()
        self.swapfile = swapfile
        self.rescalekin = krescale
        self.stride = int(stride)
        self._cached_V = None

        if repindex is None:
            self.repindex = np.zeros(0, int)
        else:
            self.repindex = np.asarray(repindex, int).copy()

        self.mode = "qremd"

    def bind(self, syslist, prng, omaker):
        super().bind(syslist, prng, omaker)

        if self.repindex is None or len(self.repindex) == 0:
            self.repindex = np.asarray(list(range(len(self.syslist))))
        else:
            if len(self.syslist) != len(self.repindex):
                raise ValueError("Replica index size does not match number of systems.")

        self.sf = self.output_maker.get_output(self.swapfile)

    def step(self, step=None):
        if self.stride <= 0.0:
            return

        u = self.prng.u


        #if step % self.stride != 0 or step == 0:
        #    return
        info(f"\nTrying to exchange replicas on STEP {step}", verbosity.debug)

        t_start = time.time()
        fxc = False
        sl = self.syslist
        self._cached_V = [s.forces.pot for s in sl]

        #offset = 0 if (step // self.stride) % 2 == 0 else 1
        #pairs = [(k, k+1) for k in range(offset, len(sl)-1, 2)]
        for i in range(len(sl)):
            for j in range(i):
                if 1.0 / self.stride < u:
                    continue  # tries a swap with probability 1/stride

                ti = sl[i].ensemble.temp
                tj = sl[j].ensemble.temp

                q1_orig = sl[i].beads.q.copy()
                q2_orig = sl[j].beads.q.copy()

                q1_centroid = np.mean(q1_orig, axis=0)
                q2_centroid = np.mean(q2_orig, axis=0)

                f_12 = np.sqrt(tj / ti)
                f_21 = np.sqrt(ti / tj)

                q1p = q1_centroid + f_12 * (q1_orig - q1_centroid)
                q2p = q2_centroid + f_21 * (q2_orig - q2_centroid)

                sl[i].beads.q[:] = q2p
                V_q2p = sl[i].forces.pot
                sl[j].beads.q[:] = q1p
                V_q1p = sl[j].forces.pot

                sl[i].beads.q[:] = q1_orig
                sl[j].beads.q[:] = q2_orig
                Vi = self._cached_V[i]
                Vj = self._cached_V[j]
                #Vi = sl[i].forces.pot
                #Vj = sl[j].forces.pot

                beta_i = 1.0 / ti
                beta_j = 1.0 / tj
                info(f"ti = {ti} K, beta_i = {beta_i} 1/Ha", verbosity.debug)
                info(f"ti = {tj} K, beta_i = {beta_j} 1/Ha", verbosity.debug)
                Delta1 = V_q2p - Vi
                Delta2 = V_q1p - Vj
                pxc = np.exp(-beta_i * Delta1 - beta_j * Delta2)
                info(f" @ QREMD: Acceptance criterium:{pxc:.2e} and delta1: {Delta1:.2e} and delta2: {Delta2:.2e}", verbosity.medium)
                if pxc > self.prng.u:
                    info(f" @ QREMD: SWAPPING replicas {i:5d} and {j:5d}.", verbosity.high)

                    ensemble_swap(sl[i].ensemble, sl[j].ensemble)

                    if self.rescalekin:
                        sl[i].beads.p *= np.sqrt(tj / ti)
                        sl[j].beads.p *= np.sqrt(ti / tj)
                        try:
                            sl[i].motion.barostat.p *= tj / ti
                            sl[j].motion.barostat.p *= ti / tj
                        except AttributeError:
                            pass

                    try:
                        bjh = dstrip(sl[j].motion.barostat.h0.h).copy()
                        sl[j].motion.barostat.h0.h[:] = sl[i].motion.barostat.h0.h[:]
                        sl[i].motion.barostat.h0.h[:] = bjh
                    except AttributeError:
                        pass

                    gle_scale(sl[i], tj / ti)
                    gle_scale(sl[j], ti / tj)

                    self.repindex[i], self.repindex[j] = self.repindex[j], self.repindex[i]
                    self._cached_V[i], self._cached_V[j] = self._cached_V[j], self._cached_V[i]
                    fxc = True
                else:
                    info(f" @ QREMD: SWAP REJECTED BETWEEN replicas {i:5d} and {j:5d}.", verbosity.high)

        if fxc:

            self.sf.write(f"{step:10d}")
            for idx in self.repindex:
                self.sf.write(f" {idx:5d}")
            self.sf.write("\n")
            self.sf.force_flush()

        info(f"# QREMD step evaluated in {time.time() - t_start:.6f} sec.", verbosity.debug)













