"""generate and cache the double spring pendulum dataset used by run_dsp.py (upstream settings, seed 2021)"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from oil.utils.utils import FixedNumpySeed, FixedPytorchSeed
from trainer.hamiltonian_dynamics import DoubleSpringPendulum

if __name__ == "__main__":
    with FixedNumpySeed(2021), FixedPytorchSeed(2021):
        ds = DoubleSpringPendulum(n_systems=5000, chunk_len=5, dt=0.2, integration_time=30, regen=False)
    print(np.asarray(ds.Zs).shape, float(np.asarray(ds.Zs).std()))
