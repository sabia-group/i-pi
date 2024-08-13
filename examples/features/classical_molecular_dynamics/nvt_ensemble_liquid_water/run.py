#!/usr/bin/env python
# -*-coding:utf-8 -*-open(filename, "r")
'''
@File    :   run.py
@Time    :   2024/06/07 17:05:46
@Author  :   George Trenins
@Contact :   gstrenin@gmail.com
@Desc    :   None
'''


from __future__ import print_function, division, absolute_import

def main():
    from ipi.utils.softexit import softexit
    from ipi.engine.simulation import Simulation
    from pathlib import Path

    cwd = Path(__file__).parent

    simulation = Simulation.load_from_xml(
        cwd / 'input.xml',
        request_banner=True,
        custom_verbosity='medium')

    simulation.run()
    softexit.trigger(status="success", message=" @ SIMULATION: Exiting cleanly.")

if __name__ == "__main__":
    main()