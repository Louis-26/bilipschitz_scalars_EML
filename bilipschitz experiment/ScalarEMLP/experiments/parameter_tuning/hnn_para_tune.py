import sys, os

# if os.path.abspath(os.path.join(os.getcwd(), "..")) not in sys.path:
#     sys.path.insert(0,os.path.abspath(os.path.join(os.getcwd(), "..")))

print(sys.path)

from scalaremlp.nn.objax import InvarianceLayer_objax
# import trainer
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
    already_run=[
        (3, 100, 0.01), (3, 100, 5e-3), (3, 100, 3e-3),
        (3, 150, 0.01)
    ]
    for parameter in parameter_comb:
        layer_num, hidden_layer_num, lr = parameter
        if parameter in already_run:
            print(f"Skipping already run parameters: layers={layer_num}, hidden_layers={hidden_layer_num}, lr={lr}")
            continue
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

        # for test purpose
        makeTrainerScalars.__kwdefaults__["num_epochs"]=1
        makeTrainerScalars.__kwdefaults__["save"] = False
        # cfg, outcome = Trial(argupdated_config(kw))
        # cfg, outcome = Trial(argupdated_config(makeTrainerScalars.__kwdefaults__))
        a=argupdated_config(makeTrainerScalars.__kwdefaults__)
        b=makeTrainerScalars.__kwdefaults__.copy()
        b["save"] = False
        # cfg, outcome = Trial(makeTrainerScalars.__kwdefaults__)
        cfg, outcome = Trial(b)

        print("cfg:", cfg)
        print("outcome:",outcome)
        print(type(outcome))
