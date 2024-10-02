"""Holds the algorithms to perform replica exchange.

Algorithms implemented by Robert Meissner and Riccardo Petraglia, 2016
"""

# This file is part of i-PI.
# i-PI Copyright (C) 2014-2016 i-PI developers
# See the "licenses" directory for full license information.


import numpy as np
import time

from ipi.engine.smotion import Smotion
from ipi.engine.ensembles import ensemble_swap
from ipi.utils.depend import *
from ipi.utils.messages import verbosity, info


__all__ = ["ReplicaExchange"]


# TODO: Do not shout :-)
#       (1) Exchange of Hamiltonians is missing


# utility functions to traverse systems to rescale all of the s momenta of
# GLE thermostats that might be around. should look also inside multi-motion
# and multi-thermo classes
def thermo_scale(thermo, scale):
    if hasattr(thermo, "tlist"):
        for t in thermo.tlist:
            thermo_scale(t, scale)
    if hasattr(thermo, "s"):  # scale the GLE degrees of freedom
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


class ReplicaExchange(Smotion):
    """Replica exchange routine.

    Attributes:
        every: on which steps REMD should be performed
        exchange:
            temperature: activate temperature replica exchange
            hamiltonian: activate hamiltonian replica exchange
            bias: activate hamiltonian replica exchange ***not yet implemented
    """

    def __init__(
        self,
        stride=1.0,
        repindex=None,
        krescale=True,
        srescale=False,
        swapfile="PARATEMP",
    ):
        """Initialises REMD.

        Args:

        """

        super(ReplicaExchange, self).__init__()

        self.swapfile = swapfile
        self.rescalekin = krescale
        self.rescalespring = srescale

        # replica exchange options
        self.stride = stride

        if repindex is None:
            self.repindex = np.zeros(0, int)
        else:
            self.repindex = np.asarray(repindex, int).copy()

        self.mode = "remd"

    def bind(self, syslist, prng, omaker):
        super(ReplicaExchange, self).bind(syslist, prng, omaker)

        if self.repindex is None or len(self.repindex) == 0:
            self.repindex = np.asarray(list(range(len(self.syslist))))
        else:
            if len(self.syslist) != len(self.repindex):
                raise ValueError(
                    "Size of replica index does not match number of systems replicas"
                )

        self.sf = self.output_maker.get_output(self.swapfile)

    def step(self, step=None):
        """Tries to exchange replica."""

        if self.stride <= 0.0:
            return

        info("\nTrying to exchange replicas on STEP %d" % step, verbosity.debug)

        t_start = time.time()
        fxc = False
        sl = self.syslist

        t_eval = 0
        t_swap = 0
        for i in range(len(sl)):
            for j in range(i):
                if 1.0 / self.stride < self.prng.u:
                    continue  # tries a swap with probability 1/stride

                t_eval -= time.time()
                ti = sl[i].ensemble.temp
                tj = sl[j].ensemble.temp
                eci = sl[i].ensemble.econs
                ecj = sl[j].ensemble.econs
                pensi = sl[i].ensemble.lpens
                pensj = sl[j].ensemble.lpens
                #                print("PENS FIRST", pensi, pensj)
                t_eval += time.time()

                t_swap -= time.time()
                #                print("potsorig", sl[i].forces.pot, sl[j].forces.pot, sl[i].ensemble.temp, sl[j].ensemble.temp)
                #                print("factororig", sl[i].forces.pot*sl[i].ensemble.temp, sl[j].forces.pot*sl[j].ensemble.temp)
                print("springorig", sl[i].nm.vspring, sl[j].nm.vspring)
                print("kinorig", sl[i].nm.kin, sl[j].nm.kin)
                ensemble_swap(
                    sl[i].ensemble, sl[j].ensemble
                )  # tries to swap the ensembles!
                #               print("potsswap", sl[i].forces.pot, sl[j].forces.pot,sl[i].ensemble.temp, sl[j].ensemble.temp)
                #               print("factorswap", sl[i].forces.pot * sl[i].ensemble.temp, sl[j].forces.pot * sl[j].ensemble.temp)
                if self.rescalespring:
                    # keep some information about positions and forces
                    dbeadsi = sl[i].beads.clone()
                    dcelli = sl[i].cell.clone()
                    dbeadsj = sl[j].beads.clone()
                    dcellj = sl[j].cell.clone()
                    oldforcesi = sl[i].forces.dump_state()
                    oldforcesj = sl[j].forces.dump_state()

                # it is generally a good idea to rescale the kinetic energies,
                # which means that the exchange is done only relative to the potential energy part.
                if self.rescalekin:
                    # also rescales the velocities -- should do the same with cell velocities
                    sl[i].beads.p *= np.sqrt(tj / ti)
                    sl[j].beads.p *= np.sqrt(ti / tj)
                    try:  # if motion has a barostat, and barostat has a momentum, does the swap
                        # also note that the barostat has a hidden T dependence inside the mass, so
                        # as a matter of fact <p^2> \propto T^2
                        sl[i].motion.barostat.p *= tj / ti
                        sl[j].motion.barostat.p *= ti / tj
                    except AttributeError:
                        pass
                print("springswap", sl[i].nm.vspring, sl[j].nm.vspring)
                print("kinswapscale", sl[i].nm.kin, sl[j].nm.kin)
                try:  # if motion has a barostat, and the barostat has a reference cell, does the swap
                    # as that when there are very different pressures, the cell should reflect the
                    # pressure/temperature dependence. this also changes the barostat conserved quantities
                    bjh = dstrip(sl[j].motion.barostat.h0.h).copy()
                    sl[j].motion.barostat.h0.h[:] = sl[i].motion.barostat.h0.h[:]
                    sl[i].motion.barostat.h0.h[:] = bjh
                except AttributeError:
                    pass
                # try to get rid of the spring term contributions and leave exchange mostly on the centroid
                # needs extra potential evaluations
                if self.rescalespring:
                    # needs centroid position
                    # pens of the initial potentials are saved
                    # print("positionsi before", positionsi)
                    # print("rescaled before", sl[i].beads.q)
                    centroidi = dstrip(sl[i].beads.qc)
                    centroidj = dstrip(sl[j].beads.qc)
                    deltai = sl[i].beads.q - centroidi
                    deltaj = sl[j].beads.q - centroidj
                    # print("TYPES bead cent i", type(sl[i].beads.q),  type(centroidi))
                    # print("TYPES bead cent j", type(sl[j].beads.q), type(centroidj))
                    sl[i].beads.q = centroidi + (ti / tj) ** (1.0 / 2) * deltai
                    # print("TYPES BEADS", type(sl[i].beads.q))
                    sl[j].beads.q = centroidj + (tj / ti) ** (1.0 / 2) * deltaj
                    # print("positionsi after", positionsi)
                    # print("rescaled after, cent", sl[j].beads.q, centroidj, np.sqrt(ti / tj), (positionsj - centroidj))
                    # I think the barostat should be fine, no need to do anything

                t_swap += time.time()

                t_eval -= time.time()
                # if we rescaled the positions, this should trigger a new call to the drivers
                newpensi = sl[i].ensemble.lpens
                newpensj = sl[j].ensemble.lpens
                #                print("potsnew", sl[i].forces.pot, sl[j].forces.pot, sl[i].ensemble.temp, sl[j].ensemble.temp)
                print("springnew", sl[i].nm.vspring, sl[j].nm.vspring)
                print("kinnew", sl[i].nm.kin, sl[j].nm.kin)
                #                print("factornew", sl[i].forces.pot * sl[i].ensemble.temp, sl[j].forces.pot * sl[j].ensemble.temp)

                # it seems to me that this should be the same
                pxc = np.exp((newpensi + newpensj) - (pensi + pensj))
                #                print("PENS LAST", pensi, pensj)
                #                print("NEWPENS", newpensi, newpensj)
                t_eval += time.time()

                if pxc > self.prng.u:  # really does the exchange
                    info(
                        " @ PT:  SWAPPING replicas % 5d and % 5d." % (i, j),
                        verbosity.medium,
                    )

                    # if we have GLE thermostats, we also have to exchange rescale the s!!!
                    gle_scale(sl[i], (tj / ti))
                    gle_scale(sl[j], (ti / tj))

                    t_eval -= time.time()
                    # we just have to carry on with the swapped ensembles, but we also keep track of the changes in econs
                    # I will also carry on with the rescaled geometries. It is a good question whether I should unscale them? I think not.
                    sl[i].ensemble.eens += eci - sl[i].ensemble.econs
                    sl[j].ensemble.eens += ecj - sl[j].ensemble.econs
                    t_eval += time.time()

                    self.repindex[i], self.repindex[j] = (
                        self.repindex[j],
                        self.repindex[i],
                    )  # keeps track of the swap

                    fxc = True  # signal that an exchange has been made!

                else:  # undoes the swap - note that the original positions and forces were never changed, so they do not need to be undone
                    t_swap -= time.time()
                    ensemble_swap(sl[i].ensemble, sl[j].ensemble)

                    # undoes the kinetic scaling
                    if self.rescalekin:
                        sl[i].beads.p *= np.sqrt(ti / tj)
                        sl[j].beads.p *= np.sqrt(tj / ti)
                        try:
                            sl[i].motion.barostat.p *= ti / tj
                            sl[j].motion.barostat.p *= tj / ti
                        except AttributeError:
                            pass

                    if self.rescalespring:
                        ## HERE NEEDS TO UNDO THE RESCALING OF POSITIONS BUT RESTORE THE ENERGIES AND FORCES WITHOUT RECALCULATING
                        sl[i].beads.q = dbeadsi.q
                        sl[j].beads.q = dbeadsj.q
                        sl[i].cell.h = dcelli.h
                        sl[j].cell.h = dcellj.h
                        sl[i].forces.load_state(oldforcesi)
                        sl[i].forces.load_state(oldforcesj)

                    try:
                        bjh = dstrip(sl[j].motion.barostat.h0.h).copy()
                        sl[j].motion.barostat.h0.h[:] = sl[i].motion.barostat.h0.h[:]
                        sl[i].motion.barostat.h0.h[:] = bjh
                    except AttributeError:
                        pass

                    t_swap += time.time()
                    info(
                        " @ PT:  SWAP REJECTED BETWEEN replicas % 5d and % 5d."
                        % (i, j),
                        verbosity.medium,
                    )

                # tempi = copy(self.syslist[i].ensemble.temp)

                # self.syslist[i].ensemble.temp = copy(self.syslist[j].ensemble.temp)
                # velocities have to be adjusted according to the new temperature

        if fxc:  # writes out the new status
            self.sf.write("% 10d" % (step))
            for i in self.repindex:
                self.sf.write(" % 5d" % (i))
            self.sf.write("\n")
            self.sf.force_flush()

        info(
            "# REMD step evaluated in %f (%f eval, %f swap) sec."
            % (time.time() - t_start, t_eval, t_swap),
            verbosity.debug,
        )
