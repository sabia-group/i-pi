# pylint: disable=all

import numpy as np
import time
from ipi.utils.units import *
from ipi.engine.smotion import Smotion
from ipi.engine.ensembles import ensemble_swap
from ipi.utils.depend import dstrip
from ipi.utils.messages import verbosity, info
from time import perf_counter
from ipi.utils.units import Constants
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

def ring_polymer_rg(beads):
    q  = dstrip(beads.q)
    qc = dstrip(beads.qc)
    dq = q - qc
    return np.sqrt(np.mean(np.sum(dq*dq, axis=1)))

from ipi.utils.messages import info, verbosity
import numpy as np

class QReplicaExchange(Smotion):
    """quantum replica exchange (QREMD)."""

    def __init__(self, stride=1.0, repindex=None, krescale=True, swapfile="PARATEMP", rand_mix=0.0, sim_mode="nn", nnn_mix=0):  
        super(QReplicaExchange, self).__init__()
        self.swapfile = swapfile
        self.rescalekin = krescale
        self.stride = int(stride)
        self.rand_mix = rand_mix # probability for random pairing (instead of nearest neighbor pairing)
        self.sim_mode = sim_mode #"nn" for NN Swapping or random pairing, "all" for all-pairs swapping, window for NNN Swapping depending on window size
        self.nnn_mix = nnn_mix
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
        t_dump = 0.0
        t_load = 0.0
        t_force = 0.0
        
        if self.stride <= 0:
            return

        info(f"\nTrying to exchange replicas on STEP {step}", verbosity.debug)

        t_start = time.time()
        fxc = False
        sl = self.syslist
        N = len(sl)
        mode = (self.sim_mode or "nn").strip().lower()

        pairs = []

        if mode == "all":
            pairs = [(i, j) for i in range(N) for j in range(i + 1, N)]
            self.prng.shuffle(pairs)

        elif mode == "nn":
            # alternierender even/odd shift
            if self.prng.u < 0.5:
                pairs = [(i, i + 1) for i in range(0, N - 1, 2)]
            else:
                pairs = [(i, i + 1) for i in range(1, N - 1, 2)]

        elif mode == "nnn":
            if self.prng.u < 0.5:
                pairs = [(i, i + 2) for i in range(0, N - 2, 4)]
            else:
                pairs = [(i, i + 2) for i in range(1, N - 2, 4)]

        elif mode == "mix":
            # Wahrscheinlichkeiten sauber behandeln
            p_random = float(self.rand_mix)
            p_nnn = float(self.nnn_mix)

            if p_random < 0 or p_nnn < 0 or (p_random + p_nnn) > 1.0:
                raise ValueError("rand_mix and nnn_mix must satisfy 0<=p and p_random+p_nnn<=1")

            u = self.prng.u

            if u < p_random:
                # random pairing
                idx = np.arange(N)
                self.prng.shuffle(idx)
                pairs = [(idx[k], idx[k + 1]) for k in range(0, N - 1, 2)]

            elif u < p_random + p_nnn:
                # distance-2 pairing
                if self.prng.u < 0.5:
                    pairs = [(i, i + 2) for i in range(0, N - 2, 4)]
                else:
                    pairs = [(i, i + 2) for i in range(1, N - 2, 4)]

            else:
                # NN pairing
                if self.prng.u < 0.5:
                    pairs = [(i, i + 1) for i in range(0, N - 1, 2)]
                else:
                    pairs = [(i, i + 1) for i in range(1, N - 1, 2)]

        else:
            raise ValueError(f"Unknown sim_mode '{self.sim_mode}'. Use: nn, nnn, mix, all.")

        #loop over all created pairs
        for (i, j) in pairs:
            if 1.0 / self.stride < self.prng.u:
                continue  # tries a swap with probability 1/stride

            #info(f"{i, j}",verbosity.low)
            ##########-backup data-##########
            dbeadsi = sl[i].beads.clone()
            dcelli = sl[i].cell.clone()
            dbeadsj = sl[j].beads.clone()
            dcellj = sl[j].cell.clone()
            t0 = perf_counter()
            oldfi = sl[i].forces.dump_state()
            oldfj = sl[j].forces.dump_state()

            ##############-time measure-##############
            t_dump += perf_counter() - t0
            ##########-backup data-##########

            ti = sl[i].ensemble.temp
            tj = sl[j].ensemble.temp
            eci = sl[i].ensemble.econs
            ecj = sl[j].ensemble.econs
            lpensi = sl[i].ensemble.lpens
            lpensj = sl[j].ensemble.lpens

            #coordinates not containing dependency, 
            qi = dstrip(sl[i].beads.q).copy()
            qj = dstrip(sl[j].beads.q).copy()
            qi_centroid = dstrip(sl[i].beads.qc).copy()
            qj_centroid = dstrip(sl[j].beads.qc).copy()
            qi_scaled = qi_centroid + (ti / tj)**0.5 * (qi - qi_centroid) #qi auf temp j 
            qj_scaled = qj_centroid + (tj / ti)**0.5 * (qj - qj_centroid) #qj auf temp i

            sl[i].beads.q = qi_scaled
            sl[j].beads.q = qj_scaled
            _ = sl[i].nm.qnm 
            _ = sl[j].nm.qnm
            pots_after = sl[i].forces.pots
            #swap ensemble
            ensemble_swap(sl[i].ensemble, sl[j].ensemble)
            _ = sl[i].nm.qnm 
            _ = sl[j].nm.qnm
            _ = sl[i].nm.omegak
            _ = sl[j].nm.omegak
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

            #ensure that lpens calls a new force calculation
            t0 = perf_counter()
            newpensi = sl[i].ensemble.lpens #Vi(qi')
            newpensj = sl[j].ensemble.lpens #Vj(qj')
            t_force += perf_counter() - t0
            pxc = ((newpensi + newpensj) - (lpensi + lpensj))
            #info(f"pxc value {pxc}", verbosity.low)
            if pxc > np.log(self.prng.u):
                #info(f"pxc value {pxc}", verbosity.low)
                info(
                    f" @ QREMD: SWAPPING replicas {i:5d} and {j:5d}.",
                    verbosity.high,
                )

                gle_scale(sl[i], tj / ti)
                gle_scale(sl[j], ti / tj)

                #update conserved energies
                sl[i].ensemble.eens += eci - sl[i].ensemble.econs
                sl[j].ensemble.eens += ecj - sl[j].ensemble.econs
                #update replica indices
                self.repindex[i], self.repindex[j] = (
                    self.repindex[j],
                    self.repindex[i],
                )

                fxc = True
            else:
                #undoes the changes before acceptance
                ensemble_swap(sl[i].ensemble, sl[j].ensemble)

                #undoes the kinetic energy rescaling
                if self.rescalekin:
                        sl[i].beads.p *= np.sqrt(ti / tj)
                        sl[j].beads.p *= np.sqrt(tj / ti)
        
                        try:
                            sl[i].motion.barostat.p *= ti / tj
                            sl[j].motion.barostat.p *= tj / ti
                        except AttributeError:
                            pass
                try:
                    bjh = dstrip(sl[j].motion.barostat.h0.h).copy()
                    sl[j].motion.barostat.h0.h[:] = sl[i].motion.barostat.h0.h[:]
                    sl[i].motion.barostat.h0.h[:] = bjh
                except AttributeError:
                    pass

                t0 = perf_counter()

                #return to original state, including forces etc. 
                sl[i].beads.q = dbeadsi.q
                sl[j].beads.q = dbeadsj.q
                sl[i].cell.h = dcelli.h
                sl[j].cell.h = dcellj.h
                sl[i].forces.load_state(oldfi)
                sl[j].forces.load_state(oldfj)

                t_load += perf_counter() - t0

                info(
                    f" @ QREMD: SWAP REJECTED BETWEEN replicas {i:5d} and {j:5d}.",
                    verbosity.high,
                )

        if fxc:
            self.sf.write(f"{step:10d}")
            for idx in self.repindex:
                self.sf.write(f" {idx:5d}")
            self.sf.write("\n")
            self.sf.force_flush()

        info(
            f"# QREMD step evaluated in {time.time() - t_start:.6f} sec.",
            verbosity.debug,
        )

        info(f"t_dump, t_load, t_extra_force {t_dump}, {t_load}, {t_force}", verbosity.high)

