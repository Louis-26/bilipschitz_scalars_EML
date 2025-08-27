import sys, os
# if os.path.abspath(os.path.join(os.getcwd(), "..")) not in sys.path:
#     sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..")))

while not os.getcwd().endswith("ScalarEMLP_bilipschitz"):
    os.chdir("..")

from scalaremlp.nn.objax import EquivarianceLayer_objax,compute_scalars,radial_basis_transform
from trainer.hamiltonian_dynamics import IntegratedODETrainer,DoubleSpringPendulum,odeScalars_trial
from torch.utils.data import DataLoader
from oil.utils.utils import FixedNumpySeed,FixedPytorchSeed
from trainer.utils import LoaderTo
from oil.datasetup.datasets import split_dataset
from oil.tuning.args import argupdated_config
import logging
import objax

import itertools
from neuralode_scalars import makeTrainerScalars


layer_num_li=[3,5,7]
hidden_layer_num_li=[100,150,200]
lr_li=[1e-2,5e-3,3e-3]

levels = {'critical': logging.CRITICAL,'error': logging.ERROR,
          'warn': logging.WARNING,'warning': logging.WARNING,
          'info': logging.INFO,'debug': logging.DEBUG}

if __name__ == '__main__':
    if len(sys.argv)!=1:
        sys.argv=[sys.argv[0]]  # Remove IPython kernel argument
    parameter_comb=list(itertools.product(layer_num_li,hidden_layer_num_li,lr_li))
    with open(file="parameter_tune_result_node.txt", mode="w") as f:
        f.write("NODE scalars parameter tuning results\n")
        f.write("="*50+"\n")
    already_run=[]
    for parameter in parameter_comb:
        layer_num, hidden_layer_num, lr = parameter
        Trial = odeScalars_trial(makeTrainerScalars)
        if parameter in already_run:
            print(f"Skipping already run parameters: layers={layer_num}, hidden_layers={hidden_layer_num}, lr={lr}")
            continue
        print(f"Running with parameters: layers={layer_num}, hidden_layers={hidden_layer_num}, lr={lr}")
        Trial = odeScalars_trial(makeTrainerScalars)

        # change the parameter setting
        makeTrainerScalars.__kwdefaults__["net_config"]["n_layers"] = layer_num
        makeTrainerScalars.__kwdefaults__["net_config"]["n_hidden"] = hidden_layer_num
        makeTrainerScalars.__kwdefaults__["lr"] = lr
        # makeTrainerScalars.__kwdefaults__["num_epochs"]=1
        makeTrainerScalars.__kwdefaults__["save"]=False
        
        # 1, doesn't work
        cfg, outcome = Trial(argupdated_config(makeTrainerScalars.__kwdefaults__))
        
        # 2, only this works
        # kwds = makeTrainerScalars.__kwdefaults__.copy()
        # kwds = copy.deepcopy(makeTrainerScalars.__kwdefaults__)
        # cfg, outcome = Trial(kwds)

        with open(file="parameter_tune_result_node.txt", mode="a") as f:
            parameter_w="-".join(map(str, parameter))
            f.write(f"layer number-hidden layer number-learning rate: {parameter_w}\n")
            outcome_str = [s.strip() for s in str(outcome).split("\n")]
            f.write(f"{outcome_str[0]}: {'-'.join(outcome_str[1].split()[1:])}\n")
            f.write("-"*50)
            f.write("\n") 

