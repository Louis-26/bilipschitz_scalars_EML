hamiltonian_dynamics.py
```python
from torch.utils.data import Dataset
class HamiltonianDataset(Dataset):
    def __getitem__(self, i):
        # make sure tensors are writable, replace the following
        # return (self.Zs[i, 0], self.T), self.Zs[i]
        return (self.Zs[i, 0].copy(), self.T.copy()), self.Zs[i].copy()

```

```python
import copy
class hnnScalars_trial(object):
    def __call__(self,cfg):
        try:
            cfg_cpy = copy.deepcopy(cfg)
            cfg_cpy["dataset"] = cfg["dataset"].__name__
            # trainer.logger.add_scalars('config',flatten_dict(cfg))
            trainer.logger.add_scalars('config', flatten_dict(cfg_cpy))

class odeScalars_trial(object):
    def __call__(self,cfg):
        try:
            cfg_cpy = copy.deepcopy(cfg)
            cfg_cpy["dataset"] = cfg["dataset"].__name__
            # trainer.logger.add_scalars('config',flatten_dict(cfg))
            trainer.logger.add_scalars('config', flatten_dict(cfg_cpy))

```

