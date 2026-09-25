import sys, os

if os.path.abspath(os.path.join(os.getcwd(), "..")) not in sys.path:
    sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..")))

from scalaremlp.nn.objax import InvarianceLayer_objax
from trainer.hamiltonian_dynamics import IntegratedDynamicsTrainer, DoubleSpringPendulum, hnnScalars_trial
from torch.utils.data import DataLoader
from oil.utils.utils import FixedNumpySeed, FixedPytorchSeed
from trainer.utils import LoaderTo
from oil.datasetup.datasets import split_dataset
from oil.tuning.args import argupdated_config
import torch.nn as nn
import logging
import scalaremlp
import scalaremlp.reps
import objax

from hnn_scalars import makeTrainerScalars

levels = {'critical': logging.CRITICAL, 'error': logging.ERROR,
          'warn': logging.WARNING, 'warning': logging.WARNING,
          'info': logging.INFO, 'debug': logging.DEBUG}



if __name__ == "__main__":
    Trial = hnnScalars_trial(makeTrainerScalars)
    cfg, outcome = Trial(argupdated_config(makeTrainerScalars.__kwdefaults__))
    # print(outcome)

