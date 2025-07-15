import sys, os

if os.path.abspath(os.path.join(os.getcwd(), "..")) not in sys.path:
    sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..")))


from scalaremlp.nn import EquivarianceLayer_objax, compute_scalars, radial_basis_transform
from trainer.hamiltonian_dynamics import IntegratedODETrainer, DoubleSpringPendulum, odeScalars_trial
from torch.utils.data import DataLoader
from oil.utils.utils import FixedNumpySeed, FixedPytorchSeed
from trainer.utils import LoaderTo
from oil.datasetup.datasets import split_dataset
from oil.tuning.args import argupdated_config
import logging
import objax

from neuralode_scalars import makeTrainerScalars

levels = {'critical': logging.CRITICAL, 'error': logging.ERROR,
          'warn': logging.WARNING, 'warning': logging.WARNING,
          'info': logging.INFO, 'debug': logging.DEBUG}





if __name__ == "__main__":
    Trial = odeScalars_trial(makeTrainerScalars)
    cfg, outcome = Trial(argupdated_config(makeTrainerScalars.__kwdefaults__))
    print(outcome)
