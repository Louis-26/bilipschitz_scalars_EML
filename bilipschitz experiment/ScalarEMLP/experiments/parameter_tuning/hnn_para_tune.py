import sys, os

if os.path.abspath(os.path.join(os.getcwd(), "..")) not in sys.path:
    sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..")))

from scalaremlp.nn import InvarianceLayer_objax
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
import itertools
from hnn_scalars import makeTrainerScalars

layer_num_li = [3, 5, 7]
hidden_layer_num_li = [100, 150, 200]
lr_li = [1e-2, 5e-3, 3e-3]

levels = {'critical': logging.CRITICAL, 'error': logging.ERROR,
          'warn': logging.WARNING, 'warning': logging.WARNING,
          'info': logging.INFO, 'debug': logging.DEBUG}

if __name__ == '__main__':
    parameter_comb = list(itertools.product(layer_num_li, hidden_layer_num_li, lr_li))
    for parameter in parameter_comb:
        layer_num, hidden_layer_num, lr = parameter
        print(f"Running with parameters: layers={layer_num}, hidden_layers={hidden_layer_num}, lr={lr}")
        Trial = hnnScalars_trial(makeTrainerScalars)

        # makeTrainerScalars.__kwdefaults__: a dictionary of default arguments for the function
        # print(makeTrainerScalars.__kwdefaults__)
        # kw = makeTrainerScalars.__kwdefaults__.copy()
        # kw["net_config"]["n_layers"] = layer_num
        # kw["net_config"]["n_hidden"] = hidden_layer_num
        # kw["lr"] = lr

        makeTrainerScalars.__kwdefaults__["net_config"]["n_layers"] = layer_num
        makeTrainerScalars.__kwdefaults__["net_config"]["n_hidden"] = hidden_layer_num
        makeTrainerScalars.__kwdefaults__["lr"] = lr

        # cfg, outcome = Trial(argupdated_config(kw))
        cfg, outcome = Trial(argupdated_config(makeTrainerScalars.__kwdefaults__))

        # print(cfg,outcome)
