import sys, os

# uncomment these two lines in rockish
# if os.path.abspath(os.path.join(os.getcwd(), "..")) not in sys.path:
#     sys.path.insert(0,os.path.abspath(os.path.join(os.getcwd(), "..")))
# while not os.getcwd().endswith("ScalarEMLP_bilipschitz"):
#     os.chdir("..")
# print(sys.path)
# print(sys.argv)

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
    # print(sys.argv)
    if len(sys.argv) != 1:
        sys.argv = [sys.argv[0]]  # Remove IPython kernel argument
    parameter_comb = list(itertools.product(layer_num_li, hidden_layer_num_li, lr_li))
    # parameter_comb = [
    #     (5,100,0.01)
    # ]

    already_run = [
        # (3, 100, 0.01), (3, 100, 5e-3), (3, 100, 3e-3),
        # (3, 150, 0.01)
    ]
    with open(file="parameter_tune_result_hnn.txt", mode="w") as f:
        f.write("hnnScalars parameter tuning results\n")
        f.write("=" * 50 + "\n")
    for parameter in parameter_comb:
        layer_num, hidden_layer_num, lr = parameter
        if parameter in already_run:
            print(f"Skipping already run parameters: layers={layer_num}, hidden_layers={hidden_layer_num}, lr={lr}")
            continue
        print(f"Running with parameters: layers={layer_num}, hidden_layers={hidden_layer_num}, lr={lr}")
        Trial = hnnScalars_trial(makeTrainerScalars)

        # change the parameter setting
        makeTrainerScalars.__kwdefaults__["net_config"]["n_layers"] = layer_num
        makeTrainerScalars.__kwdefaults__["net_config"]["n_hidden"] = hidden_layer_num
        makeTrainerScalars.__kwdefaults__["lr"] = lr
        # makeTrainerScalars.__kwdefaults__["num_epochs"] = 1
        makeTrainerScalars.__kwdefaults__["save"] = False

        # 1
        cfg, outcome = Trial(argupdated_config(makeTrainerScalars.__kwdefaults__))

        # 2
        # kwds = makeTrainerScalars.__kwdefaults__.copy()
        # cfg, outcome = Trial(kwds)

        with open(file="parameter_tune_result_hnn.txt", mode="a") as f:
            parameter_w = "-".join(map(str, parameter))
            f.write(f"layer number-hidden layer number-learning rate: {parameter_w}\n")
            outcome_str = [s.strip() for s in str(outcome).split("\n")]
            f.write(f"{outcome_str[0]}: {'-'.join(outcome_str[1].split()[1:])}\n")
            f.write("-" * 50)
            f.write("\n")
