# pylint: disable=all
# Authors: Jan-Niklas Mohr, Prof. Mariana Rossi
# 2026
# This smotion class implements fluctuation-rescaled replica exchange path-integral molecular dynamics (frREPIMD). The main idea is to rescale the bead fluctuations around the centroid according to the new temperatures before the acceptance test
# to remove the spring-term in the acceptance criterion, as well as the kinetic contribution due to kinetic rescaling. This ensures that the acceptance criterion solely depends on the 
# potential energy differences between the scaled and original states. 


import numpy as np
from ipi.utils.units import *
from ipi.engine.smotion import Smotion
from ipi.engine.ensembles import ensemble_swap
from ipi.utils.depend import dstrip
from ipi.utils.messages import verbosity, info
from time import perf_counter
from ipi.utils.units import *

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

    def __init__(self, stride=1.0, repindex=None, krescale=True, swapfile="PARATEMP", rand_mix=0.0, sim_mode="nn", nnn_mix=0):  
        super(QReplicaExchange, self).__init__()
        self.swapfile = swapfile #file for storing the replica indices of each system. will be written at every successful exchange step.
        self.rescalekin = krescale #whether to rescale the momenta of the systems upon exchange. should be True for temperature REMD.
        self.stride = int(stride) #how often to attempt exchanges (in steps) on average.
        self.rand_mix = rand_mix #@frREPIMD: probability of random pairing in "mix" mode. Should be between 0 and 1, and rand_mix + nnn_mix should be <= 1.
        self.sim_mode = sim_mode #choose between next-neighbor (nn), next-next-neighbor (nnn), random (rand), mixed or all
        self.nnn_mix = nnn_mix # @frREPIMD: probability of next-next-neighbor pairing in "mix" mode. Should be between 0 and 1, and rand_mix + nnn_mix should be <= 1.
        
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
        ###TEST
        #self.af = self.output_maker.get_output("QREMD_ACCEPTANCE")
        #self.af.write("# step temp_index_1 temp_index_2 pxc\n")
        #self.af.force_flush()
        #####

    def step(self, step=None):
        
        if self.stride <= 0:
            return

        info(f"\nTrying to exchange replicas on STEP {step}", verbosity.debug)

        
        t_start = perf_counter()    
        t_pairs  = 0.0   # Pair-List Construction
        t_prep   = 0.0   # Coordinate and Kinetic Energy Scaling and Ensemble Swap
        t_queue  = 0.0   # Force-Queue (Phase 2)
        t_eval   = 0.0   # Force-Evaluation + Acceptance (Phase 3)
        t_swap   = 0.0   # Restore if rejected

        fxc = False
        sl = self.syslist
        N = len(sl)
        mode = (self.sim_mode or "nn").strip().lower()

        # create list of pairs in temperature-index space
        t_pairs -= perf_counter()
        temp_to_sys = np.argsort(self.repindex)
        pairs = []

        if mode == "all":
            #be careful, mode "all" creates all possible pairs (N*(N-1)/2), which is not recommended due to the computational costs.
            temp_pairs = [(a, b) for a in range(N) for b in range(a + 1, N)]
            self.prng.shuffle(temp_pairs)

            pairs = [
                (temp_to_sys[a], temp_to_sys[b])
                for a, b in temp_pairs
            ]

        elif mode == "nn":
            # nearest neighbors in temperature index space:
            # (0,1), (2,3), ... or (1,2), (3,4), ...
            if self.prng.u < 0.5:
                temp_pairs = [(k, k + 1) for k in range(0, N - 1, 2)]
            else:
                temp_pairs = [(k, k + 1) for k in range(1, N - 1, 2)]

            pairs = [
                (temp_to_sys[a], temp_to_sys[b])
                for a, b in temp_pairs
            ]

        elif mode == "nnn":
            # next-nearest neighbors in temperature index space:
            # (0,2), (4,6), ... or (1,3), (5,7), ....
            if self.prng.u < 0.5:
                temp_pairs = [(k, k + 2) for k in range(0, N - 2, 4)]
            else:
                temp_pairs = [(k, k + 2) for k in range(1, N - 2, 4)]

            pairs = [
                (temp_to_sys[a], temp_to_sys[b])
                for a, b in temp_pairs
            ]

        elif mode == "mix":
            #create NN pairs with a certain probability, NNN pairs with a certain probability, and random pairs with the remaining probability. This allows to have more variety in the attempted exchanges, which can be beneficial for sampling, especially if the temperature ladder is not optimal or if there are bottlenecks in the exchange between certain temperature pairs. The probabilities can be adjusted to find a good balance between exploration and acceptance rates.
            p_random = float(self.rand_mix)
            p_nnn = float(self.nnn_mix)

            if p_random < 0 or p_nnn < 0 or (p_random + p_nnn) > 1.0:
                raise ValueError(
                    "rand_mix and nnn_mix must satisfy 0<=p and p_random+p_nnn<=1"
                )

            u = self.prng.u

            if u < p_random:
                temp_idx = np.arange(N)
                self.prng.shuffle(temp_idx)

                temp_pairs = [
                    (temp_idx[k], temp_idx[k + 1])
                    for k in range(0, N - 1, 2)
                ]

                pairs = [
                    (temp_to_sys[a], temp_to_sys[b])
                    for a, b in temp_pairs
                ]

            elif u < p_random + p_nnn:
                # NNN in temperature index space
                if self.prng.u < 0.5:
                    temp_pairs = [(k, k + 2) for k in range(0, N - 2, 4)]
                else:
                    temp_pairs = [(k, k + 2) for k in range(1, N - 2, 4)]

                pairs = [
                    (temp_to_sys[a], temp_to_sys[b])
                    for a, b in temp_pairs
                ]

            else:
                # NN in temperature index space
                if self.prng.u < 0.5:
                    temp_pairs = [(k, k + 1) for k in range(0, N - 1, 2)]
                else:
                    temp_pairs = [(k, k + 1) for k in range(1, N - 1, 2)]

                pairs = [
                    (temp_to_sys[a], temp_to_sys[b])
                    for a, b in temp_pairs
                ]

        else:
            raise ValueError(f"Unknown sim_mode '{self.sim_mode}'. Use: nn, nnn, mix, all.")
        t_pairs += perf_counter()

        # Phase 1: Prepare all pairs, dump states, and apply scaling to the coordinates and momenta
        t_prep -= perf_counter()
        
        pair_data = []

        #loop over all created pairs
        for (i, j) in pairs:
            if 1.0 / self.stride < self.prng.u:
                continue

            #clone original state including beads, cells and forces
            dbeadsi = sl[i].beads.clone()
            dcelli = sl[i].cell.clone()
            dbeadsj = sl[j].beads.clone()
            dcellj = sl[j].cell.clone()
            oldfi = sl[i].forces.dump_state()
            oldfj = sl[j].forces.dump_state()

            #save originale temperatures, energies and lpens for later use in acceptance criterion
            ti = sl[i].ensemble.temp
            tj = sl[j].ensemble.temp
            eci = sl[i].ensemble.econs
            ecj = sl[j].ensemble.econs
            lpensi = sl[i].ensemble.lpens
            lpensj = sl[j].ensemble.lpens

            #copy coordinates and centroids, w/o depencency
            qi = dstrip(sl[i].beads.q).copy()
            qj = dstrip(sl[j].beads.q).copy()
            qi_centroid = dstrip(sl[i].beads.qc).copy()
            qj_centroid = dstrip(sl[j].beads.qc).copy()

            #rescale bead fluctuations around the centroid according to the new temperatures. This is important to avoid huge energy differences and thus very low acceptance rates, especially for large systems and/or large temperature differences.
            qi_scaled = qi_centroid + (ti / tj)**0.5 * (qi - qi_centroid)
            qj_scaled = qj_centroid + (tj / ti)**0.5 * (qj - qj_centroid)

            #set rescaled coordinates
            sl[i].beads.q = qi_scaled
            sl[j].beads.q = qj_scaled

            #access nm.qnm to have correct normalmodes, theoretically not necessary since dependent on beads.q, but just to be sure that everything is updated after the coordinate changes and before the swap. Also access omegak to be sure that they are updated as well.
            _ = sl[i].nm.qnm
            _ = sl[j].nm.qnm

            #swap ensembles 
            ensemble_swap(sl[i].ensemble, sl[j].ensemble)

            #access nm.qnm and omegak again just to be sure everything is up to date. Theoretically not necessary. 
            _ = sl[i].nm.qnm
            _ = sl[j].nm.qnm
            _ = sl[i].nm.omegak
            _ = sl[j].nm.omegak

            #rescale momenta to match the new temperatures if desired
            if self.rescalekin:
                sl[i].beads.p *= np.sqrt(tj / ti)
                sl[j].beads.p *= np.sqrt(ti / tj)
                try:
                    sl[i].motion.barostat.p *= tj / ti
                    sl[j].motion.barostat.p *= ti / tj
                except AttributeError:
                    pass
            #handle also barostats for NPT 
            try:
                bjh = dstrip(sl[j].motion.barostat.h0.h).copy()
                sl[j].motion.barostat.h0.h[:] = sl[i].motion.barostat.h0.h[:]
                sl[i].motion.barostat.h0.h[:] = bjh
            except AttributeError:
                pass

            #store all relevant data for the pair for later use in the acceptance criterion and for restoring the original states in case of rejection. This allows to decouple the preparation of the pairs and the force evaluations from the acceptance/rejection step, which can be beneficial for parallelization and load balancing, especially if the force evaluations are expensive and the number of pairs is large.
            pair_data.append((i, j, ti, tj, eci, ecj, lpensi, lpensj,
                               dbeadsi, dcelli, dbeadsj, dcellj, oldfi, oldfj))
            t_prep += perf_counter()
        t_queue -= perf_counter()
        # Phase 2: Queue force evaluations for all pairs --> this allows for better parallelization and load balancing, especially if the force evaluations are expensive and the number of pairs is large.
        for (i, j, *_) in pair_data:
            sl[i].forces.queue()
            sl[j].forces.queue()
        t_queue += perf_counter()
        # Phase 3: Gather forces, compute acceptance probabilities, and finalize swaps
        for (i, j, ti, tj, eci, ecj, lpensi, lpensj,
             dbeadsi, dcelli, dbeadsj, dcellj, oldfi, oldfj) in pair_data:

            t_eval -= perf_counter()
            newpensi = sl[i].ensemble.lpens
            newpensj = sl[j].ensemble.lpens


            pxc = ((newpensi + newpensj) - (lpensi + lpensj))

            ### TEST
            #temp_idx_i = int(self.repindex[i])
            #temp_idx_j = int(self.repindex[j])

            #t_low = min(temp_idx_i, temp_idx_j)
            #t_high = max(temp_idx_i, temp_idx_j)
            ###
            #self.af.write(
            #    f"{step:10d} "
            #    f"{t_low:5d} "
            #    f"{t_high:5d} "
            #    f"{pxc:20.12e}\n"
            #)
            #self.af.force_flush()





            t_eval += perf_counter()

            #accept if metropolis criterion is satisfied
            if pxc > np.log(self.prng.u):
                t_eval -= perf_counter()
                info(
                    f" @ QREMD: SWAP ACCEPTED BETWEEN replicas {i:5d} and {j:5d}.",
                    verbosity.high,
                )

                # if we have GLE thermostats, we also have to exchange rescale the s
                gle_scale(sl[i], tj / ti)
                gle_scale(sl[j], ti / tj)

                # we just have to carry on with the swapped ensembles, but we also keep track of the changes in econs
                sl[i].ensemble.eens += eci - sl[i].ensemble.econs
                sl[j].ensemble.eens += ecj - sl[j].ensemble.econs


                #swap indices in replica index list to keep track of the swaps
                self.repindex[i], self.repindex[j] = (
                    self.repindex[j],
                    self.repindex[i],
                )
                fxc = True
                t_eval += perf_counter()
            #reject if not and restore the original states
            else:
                t_swap -= perf_counter()
                info(
                    f" @ QREMD: SWAP REJECTED BETWEEN replicas {i:5d} and {j:5d}.",
                    verbosity.high,
                )

                #swap ensembles back to restore original temperatures and other parameters
                ensemble_swap(sl[i].ensemble, sl[j].ensemble)

                #rescale momenta back
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

                #restore original states, incluing beads, cells and forces
                sl[i].beads.q = dbeadsi.q
                sl[j].beads.q = dbeadsj.q
                #prevent that MD integrator sees something tainted, w/o there would be additional force calls that are not necessary
                _ = sl[i].nm.qnm
                _ = sl[j].nm.qnm
                _ = sl[i].beads.qc
                _ = sl[j].beads.qc
                sl[i].cell.h = dcelli.h
                sl[j].cell.h = dcellj.h
                sl[i].forces.load_state(oldfi)
                sl[j].forces.load_state(oldfj)
                t_swap += perf_counter()
        
        ### TEST
        #self.af.force_flush()
        ####
        if fxc: #write replica indices to PARATEMP file if at least one exchange has been made
            self.sf.write(f"{step:10d}")
            for idx in self.repindex:
                self.sf.write(f" {idx:5d}")
            self.sf.write("\n")
            self.sf.force_flush()


        info(
        "# QREMD step evaluated in %f (%f pairs, %f prep, %f queue, %f eval, %f swap) sec."
        % (perf_counter() - t_start, t_pairs, t_prep, t_queue, t_eval, t_swap),
        verbosity.debug,
        )
