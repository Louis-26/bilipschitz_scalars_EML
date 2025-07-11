import jax 
from jax import grad, jit, vmap, jacfwd, jvp, vjp, random
from jax.experimental.ode import odeint
import jax.numpy as jnp
import objax
from oil.tuning.configGenerator import flatten_dict
from oil.utils.utils import export
 
import os
import torch  
from torch.utils.data import Dataset
import numpy as np 
from functools import partial 

from scalaremlp.groups import SO2eR3,O2eR3,DkeR3,Trivial
from scalaremlp.reps import T,Scalar
from .classifier import Regressor,Classifier

## Code to rollout a Hamiltonian system

def unpack(z):
    D = jnp.shape(z)[-1]
    assert D % 2 == 0
    d = D//2
    q, p_or_v = z[..., :d], z[..., d:]
    return q, p_or_v

def pack(q, p_or_v):
    return jnp.concatenate([q, p_or_v], axis=-1)

def symplectic_form(z):
    """ Equivalent to multiplying z by the matrix J=[[0,I],[-I,0]]"""
    q, p = unpack(z)
    return pack(p, -q)

def hamiltonian_dynamics(hamiltonian, z,t):
    """ Takes a Hamiltonian function, a state vector z, and an unused time t
        to compute the hamiltonian dynamics J∇H"""
    grad_h = grad(hamiltonian) # ∇H
    gh = grad_h(z) # ∇H(z)
    return symplectic_form(gh) # J∇H(z)

def HamiltonianFlow(H,z0,T):
    """ Converts a Hamiltonian H and initial conditions z0
         to rolled out trajectory at time points T.
         z0 shape (state_dim,) and T shape (t,) yields (t,state_dim) rollout."""
    dynamics = lambda z,t: hamiltonian_dynamics(H,z,t)
    return odeint(dynamics, z0, T, rtol=1e-4, atol=1e-4)#.transpose((1,0,2))

def BHamiltonianFlow(H,z0,T,tol=1e-4):
    """ Batched version of HamiltonianFlow, essentially equivalent to vmap(HamiltonianFlow),
        z0 of shape (bs,state_dim) and T of shape (t,) yields (bs,t,state_dim) rollouts """
    dynamics = jit(vmap(jit(partial(hamiltonian_dynamics,H)),(0,None)))
    return odeint(dynamics, z0, T, rtol=tol).transpose((1,0,2))

def BOdeFlow(dynamics,z0,T,tol=1e-4):
    """ Batched integration of ODE dynamics into rollout trajectories.
        Given dynamics (state_dim->state_dim) and z0 of shape (bs,state_dim)
        and T of shape (t,) outputs trajectories (bs,t,state_dim) """
    dynamics = vmap(dynamics,(0,None))
    return odeint(dynamics, z0, T, rtol=tol).transpose((1,0,2))

class HamiltonianDataset(Dataset):
    """ A dataset that generates trajectory chunks from integrating the Hamiltonian dynamics
        from a given Hamiltonian system and initial condition distribution.
        Each element ds[i] = ((ic,T),z_target) where ic (state_dim,) are the initial conditions,
        T are the evaluation timepoints, and z_target (T,state_dim) is the ground truth trajectory chunk.
        Here state_dim includes both the position q and canonical momentum p concatenated together.
        Args:
            n_systems (int): total number of trajectory chunks that makeup the dataset.
            chunk_len (int): the number of timepoints at which each chunk is evaluated
            dt (float): the spacing of the evaluation points (not the integrator step size which is set by tol=1e-4)
            integration_time (float): The integration time for evaluation rollouts and also
                the total integration time from which each trajectory chunk is randomly sampled
            regen (bool): whether or not to regenerate and overwrite any datasets cached to disk
                with the same arguments. If false, will use trajectories saved at {filename}
        Returns:
            Dataset: A (torch style) dataset.  """
    def __init__(self,n_systems=100,chunk_len=5,dt=0.2,integration_time=30,regen=False):
        super().__init__()
        # root_dir = os.path.expanduser(f"~/datasets/ODEDynamics/{self.__class__}/")
        root_dir = os.path.expanduser(f"~\\datasets\\ODEDynamics\\{self.__class__.__name__}")
        filename = os.path.join(root_dir, f"trajectories_{n_systems}_{chunk_len}_{dt}_{integration_time}.pz")

        if os.path.exists(filename) and not regen:
            Zs = torch.load(filename, weights_only=False)
        else:
            zs = self.generate_trajectory_data(n_systems, dt, integration_time)
            Zs = np.asarray(self.chunk_training_data(zs, chunk_len))
            os.makedirs(root_dir, exist_ok=True)
            torch.save(Zs, filename)
        
        self.Zs = Zs
        self.T = np.asarray(jnp.arange(0, chunk_len*dt, dt))
        self.T_long = np.asarray(jnp.arange(0,integration_time,dt))

    def __len__(self):
        return self.Zs.shape[0]

    def __getitem__(self, i):
        return (self.Zs[i, 0], self.T), self.Zs[i]

    def integrate(self,z0s,ts):
        return HamiltonianFlow(self.H,z0s, ts)
    
    def generate_trajectory_data(self, n_systems, dt, integration_time, bs=100):
        """ Returns ts: (n_systems, traj_len) zs: (n_systems, traj_len, z_dim) """
        n_gen = 0; bs = min(bs, n_systems)
        t_batches, z_batches = [], []
        while n_gen < n_systems:
            z0s = self.sample_initial_conditions(bs)
            ts = jnp.arange(0, integration_time, dt)
            new_zs = BHamiltonianFlow(self.H,z0s, ts)
            z_batches.append(new_zs)
            n_gen += bs
        zs = jnp.concatenate(z_batches, axis=0)[:n_systems]
        return zs

    def chunk_training_data(self, zs, chunk_len):
        batch_size, traj_len, *z_dim = zs.shape
        n_chunks = traj_len // chunk_len
        chunk_idx = np.random.randint(0, n_chunks, (batch_size,))
        chunked_zs = np.stack(np.split(zs,n_chunks, axis=1))
        chosen_zs = chunked_zs[chunk_idx, np.arange(batch_size)]
        return chosen_zs

    def H(self,z):
        """ The Hamiltonian function, depending on z=pack(q,p)"""
        raise NotImplementedError

    def sample_initial_conditions(self,bs):
        """ Initial condition distribution """
        raise NotImplementedError

    def animate(self, zt=None):
        """ Visualize the dynamical system, or given input trajectories.
            Usage:  from IPython.display import HTML
                    HTML(dataset.animate())"""
        if zt is None:
            zt = np.asarray(self.integrate(self.sample_initial_conditions(10)[0],self.T_long))
        # bs, T, 2nd
        if len(zt.shape) == 3:
            j = np.random.randint(zt.shape[0])
            zt = zt[j]
        xt,pt = unpack(zt)
        xt = xt.reshape((xt.shape[0],-1,3))
        anim = self.animator(xt)
        return anim.animate()

class SHO(HamiltonianDataset):
    """ A basic simple harmonic oscillator"""
    def H(self,z):
        ke = (z[...,1]**2).sum()/2
        pe = (z[...,0]**2).sum()/2
        return ke+pe
    def sample_initial_conditions(self,bs):
        return np.random.randn(bs,2)
    
class DoubleSpringPendulum(HamiltonianDataset):
    """ The double spring pendulum dataset described in the paper."""
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.rep_in = 4*T(1)#Vector
        self.rep_out = T(0)#Scalar
        self.symmetry = O2eR3()
        self.stats = (0,1,0,1)
    def H(self,z):
        g=1
        m1,m2,k1,k2,l1,l2 = 1,1,1,1,1,1
        x,p = unpack(z)
        p1,p2 = unpack(p)
        x1,x2 = unpack(x)
        ke = .5*(p1**2).sum(-1)/m1 + .5*(p2**2).sum(-1)/m2
        pe = .5*k1*(jnp.sqrt((x1**2).sum(-1))-l1)**2 
        pe += k2*(jnp.sqrt(((x1-x2)**2).sum(-1))-l2)**2
        pe += m1*g*x1[...,2]+m2*g*x2[...,2]
        return (ke + pe).sum()
    def sample_initial_conditions(self,bs):
        x1 = np.array([0,0,-1.5]) +.2*np.random.randn(bs,3)
        x2= np.array([0,0,-3.]) +.2*np.random.randn(bs,3)
        p = .4*np.random.randn(bs,6)
        z0 = np.concatenate([x1,x2,p],axis=-1)
        return z0
    @property
    def animator(self):
        return CoupledPendulumAnimation
 

class IntegratedDynamicsTrainer(Regressor):
    """ A trainer for training the Hamiltonian Neural Networks. Feel free to use your own instead."""
    def __init__(self,model,*args,**kwargs):
        super().__init__(model,*args,**kwargs)
        self.loss = objax.Jit(self.loss,model.vars())
        self.gradvals = objax.Jit(objax.GradValues(self.loss,model.vars()))

    def loss(self, minibatch):
        """ Standard cross-entropy loss """
        (z0, ts), true_zs = minibatch
        pred_zs = BHamiltonianFlow(self.model,z0,ts[0])
        return jnp.mean((pred_zs - true_zs)**2)

    def metrics(self, loader):
        mse = lambda mb: np.asarray(self.loss(mb))
        return {"MSE": self.evalAverageMetrics(loader, mse)}
    
    def logStuff(self, step, minibatch=None):
        loader = self.dataloaders['test']
        metrics = {'test_Rollout': np.exp(self.evalAverageMetrics(loader,partial(log_rollout_error,loader.dataset,self.model)))}
        print(step, metrics)
        self.logger.add_scalars('metrics', metrics, step)
        super().logStuff(step,minibatch)

class IntegratedODETrainer(Regressor):
    """ A trainer for training the Neural ODEs. Feel free to use your own instead."""
    def __init__(self,model,*args,**kwargs):
        super().__init__(model,*args,**kwargs)
        self.loss = objax.Jit(self.loss,model.vars())
        #self.model = objax.Jit(self.model)
        self.gradvals = objax.Jit(objax.GradValues(self.loss,model.vars()))#objax.Jit(objax.GradValues(fastloss,model.vars()),model.vars())
        #self.model.predict = objax.Jit(objax.ForceArgs(model.__call__,training=False),model.vars())

    def loss(self, minibatch):
        """ Standard cross-entropy loss """
        (z0, ts), true_zs = minibatch
        pred_zs = BOdeFlow(self.model,z0,ts[0])
        return jnp.mean((pred_zs - true_zs)**2)

    def metrics(self, loader):
        mse = lambda mb: np.asarray(self.loss(mb))
        return {"MSE": self.evalAverageMetrics(loader, mse)}
        
    def logStuff(self, step, minibatch=None):
        loader = self.dataloaders['test']
        metrics = {'test_Rollout': np.exp(self.evalAverageMetrics(loader,partial(log_rollout_error_ode,loader.dataset,self.model)))}
        print(step, metrics)
        self.logger.add_scalars('metrics', metrics, step)
        super().logStuff(step,minibatch)
  
def rel_err(a,b):
    """ Relative error |a-b|/|a+b|"""
    return jnp.sqrt(((a-b)**2).mean())/(jnp.sqrt((a**2).mean())+jnp.sqrt((b**2).mean()))#

def log_rollout_error(ds,model,minibatch):
    """ Computes the log of the geometric mean of the rollout
        error computed between the dataset ds and HNN model
        on the initial condition in the minibatch."""
    (z0, _), _ = minibatch
    pred_zs = BHamiltonianFlow(model,z0,ds.T_long)
    gt_zs  = BHamiltonianFlow(ds.H,z0,ds.T_long)
    errs = vmap(vmap(rel_err))(pred_zs,gt_zs) # (bs,T,)
    clamped_errs = jax.lax.clamp(1e-7,errs,np.inf)
    log_geo_mean = jnp.log(clamped_errs).mean()
    return log_geo_mean
   


def pred_and_gt(ds,model,minibatch):
    (z0, _), _ = minibatch
    pred_zs = BHamiltonianFlow(model,z0,ds.T_long,tol=2e-6)
    gt_zs  = BHamiltonianFlow(ds.H,z0,ds.T_long,tol=2e-6)
    return np.stack([pred_zs,gt_zs],axis=-1)

def pred_and_gt_ode(ds,model,minibatch):
    (z0, _), _ = minibatch
    pred_zs = BOdeFlow(model,z0,ds.T_long,tol=2e-6)
    gt_zs  = BHamiltonianFlow(ds.H,z0,ds.T_long,tol=2e-6)
    return np.stack([pred_zs,gt_zs],axis=-1)
 
def log_rollout_error_ode(ds,model,minibatch):
    """ Computes the log of the geometric mean of the rollout
        error computed between the dataset ds and NeuralODE model
        on the initial condition in the minibatch."""
    (z0, _), _ = minibatch
    pred_zs = BOdeFlow(model,z0,ds.T_long)
    gt_zs  = BHamiltonianFlow(ds.H,z0,ds.T_long)
    errs = vmap(vmap(rel_err))(pred_zs,gt_zs) # (bs,T,)
    clamped_errs = jax.lax.clamp(1e-7,errs,np.inf)
    log_geo_mean = jnp.log(clamped_errs).mean()
    return log_geo_mean
 




 





### Some extra code to make pretty visualizations for the given system

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
import numpy as np

class Animation(object):
    def __init__(self, qt,lims=None,traj_lw=1,figkwargs={}):
        """ [qt (T,n,d)"""
        self.qt = qt
        T,n,d = qt.shape
        assert d in (2,3), "too many dimensions for animation"
        self.fig = plt.figure(**figkwargs)
        self.ax = self.fig.add_axes([0, 0, 1, 1],projection='3d') if d==3 else self.fig.add_axes([0, 0, 1, 1])
        
        #self.ax.axis('equal')
        xyzmin = self.qt.min(0).min(0)#.min(dim=0)[0].min(dim=0)[0]
        xyzmax = self.qt.max(0).max(0)#.max(dim=0)[0].max(dim=0)[0]
        delta = xyzmax-xyzmin
        lower = xyzmin-.1*delta; upper = xyzmax+.1*delta
        if lims is None:
            lims = (min(lower),max(upper)),(min(lower),max(upper)),(min(lower),max(upper))
        self.ax.set_xlim(lims[0])
        self.ax.set_ylim(lims[1])
        if d==3: self.ax.set_zlim(lims[2])
        if d!=3: self.ax.set_aspect("equal")
        #elf.ax.auto_scale_xyz()
        empty = d*[[]]
        self.colors = np.random.choice([f"C{i}" for i in range(10)],size=n,replace=False)
        self.objects = {
            'pts':sum([self.ax.plot(*empty, "o", ms=6,color=self.colors[i]) for i in range(n)], []),
            'traj_lines':sum([self.ax.plot(*empty, "-",color=self.colors[i],lw=traj_lw) for i in range(n)], []),
        }
        
    def init(self):
        empty = 2*[[]]
        for obj in self.objects.values():
            for elem in obj:
                elem.set_data(*empty)
                #if self.qt.shape[-1]==3: elem.set_3d_properties([])
        return sum(self.objects.values(),[])

    def update(self, i=0):
        T,n,d = self.qt.shape
        trail_len = 150
        for j in range(n):
            # trails
            xyz = self.qt[max(i - trail_len,0): i + 1,j,:]
            #chunks = xyz.shape[0]//10
            #xyz_chunks = torch.chunk(xyz,chunks)
            #for i,xyz in enumerate(xyz_chunks):
            self.objects['traj_lines'][j].set_data(*xyz[...,:2].T)
            if d==3: self.objects['traj_lines'][j].set_3d_properties(xyz[...,2].T)
            self.objects['pts'][j].set_data(*xyz[-1:,...,:2].T)
            if d==3: self.objects['pts'][j].set_3d_properties(xyz[-1:,...,2].T)
        #self.fig.canvas.draw()
        return sum(self.objects.values(),[])

    def animate(self):
        return animation.FuncAnimation(self.fig,self.update,frames=self.qt.shape[0],
                    interval=33,init_func=self.init,blit=True).to_html5_video()

class PendulumAnimation(Animation):
    def __init__(self, qt,*args,**kwargs):
        super().__init__(qt,*args,**kwargs)
        empty = self.qt.shape[-1] * [[]]
        self.objects["pts"] = sum([self.ax.plot(*empty, "o", ms=10,c=self.colors[i]) for i in range(self.qt.shape[1])], [])

    def update(self, i=0):
        return super().update(i)

def helix(Ns=1000,radius=.05,turns=25):
    t = np.linspace(0,1,Ns)
    xyz = np.zeros((Ns,3))
    xyz[:,0] = np.cos(2*np.pi*Ns*t*turns)*radius
    xyz[:,1] = np.sin(2*np.pi*Ns*t*turns)*radius
    xyz[:,2] = t
    xyz[:,:2][(t>.9)|(t<.1)]=0
    return xyz

def align2ref(refs,vecs):
    """ inputs [refs (n,3), vecs (N,3)]
        outputs [aligned (n,N,3)]
    assumes vecs are pointing along z axis"""
    n,_ = refs.shape
    N,_ = vecs.shape
    norm = np.sqrt((refs**2).sum(-1))
    v = refs/norm[:,None]
    A = np.zeros((n,3,3))
    A[:,:,2] += v
    A[:,2,:] -= v
    M = (np.eye(3)+A+(A@A)/(1+v[:,2,None,None]))
    scaled_vecs = vecs[None]+0*norm[:,None,None] #broadcast to right shape
    scaled_vecs[:,:,2] *= norm[:,None]#[:,None,None]
    return (M[:,None]@scaled_vecs[...,None]).squeeze(-1)

    
class CoupledPendulumAnimation(PendulumAnimation):
    
    def __init__(self, *args, spring_lw=.6,spring_r=.2,**kwargs):
        super().__init__(*args, **kwargs)
        empty = self.qt.shape[-1]*[[]]
        self.objects["springs"] = self.ax.plot(*empty,c='k',lw=spring_lw)#
        #self.objects["springs"] = sum([self.ax.plot(*empty,c='k',lw=2) for _ in range(self.n-1)],[])
        self.helix = helix(200,radius=spring_r,turns=10)
        
    def update(self,i=0):
        qt_padded = np.concatenate([0*self.qt[i,:1],self.qt[i,:]],axis=0)
        diffs = qt_padded[1:]-qt_padded[:-1]
        x,y,z = (align2ref(diffs,self.helix)+qt_padded[:-1][:,None]).reshape(-1,3).T
        self.objects['springs'][0].set_data(x,y)
        self.objects['springs'][0].set_3d_properties(z)
        return super().update(i)

from collections.abc import Iterable

@export
class hnn_trial(object):
    """ A training trial for the HNNs, contains lots of boiler plate which is not necessary.
        Feel free to use your own."""
    def __init__(self,make_trainer,strict=True):
        self.make_trainer = make_trainer
        self.strict=strict
    def __call__(self,cfg,i=None):
        try:
            cfg.pop('local_rank',None) #TODO: properly handle distributed
            resume = cfg.pop('resume',False)
            save = cfg.pop('save',False)
            if i is not None:
                orig_suffix = cfg.setdefault('trainer_config',{}).get('log_suffix','')
                cfg['trainer_config']['log_suffix'] = os.path.join(orig_suffix,f'trial{i}/')
            trainer = self.make_trainer(**cfg)
            trainer.logger.add_scalars('config',flatten_dict(cfg))
            trainer.train(cfg['num_epochs'])
            if save: cfg['saved_at']=trainer.save_checkpoint()
            outcome = trainer.ckpt['outcome']
            trajectories = []
            for mb in trainer.dataloaders['test']:
                trajectories.append(pred_and_gt(trainer.dataloaders['test'].dataset,trainer.model,mb))
            torch.save(np.concatenate(trajectories),f"./{cfg['network']}_{cfg['net_config']['group']}_{i}.t")
        except Exception as e:
            if self.strict: raise
            outcome = e
        del trainer
        return cfg, outcome


@export
class ode_trial(object):
    """ A training trial for the Neural ODEs, contains lots of boiler plate which is not necessary.
        Feel free to use your own."""
    def __init__(self,make_trainer,strict=True):
        self.make_trainer = make_trainer
        self.strict=strict
    def __call__(self,cfg,i=None):
        try:
            cfg.pop('local_rank',None) #TODO: properly handle distributed
            resume = cfg.pop('resume',False)
            save = cfg.pop('save',False)
            if i is not None:
                orig_suffix = cfg.setdefault('trainer_config',{}).get('log_suffix','')
                cfg['trainer_config']['log_suffix'] = os.path.join(orig_suffix,f'trial{i}/')
            trainer = self.make_trainer(**cfg)
            trainer.logger.add_scalars('config',flatten_dict(cfg))
            trainer.train(cfg['num_epochs'])
            if save: cfg['saved_at']=trainer.save_checkpoint()
            outcome = trainer.ckpt['outcome']
            trajectories = []
            for mb in trainer.dataloaders['test']:
                trajectories.append(pred_and_gt_ode(trainer.dataloaders['test'].dataset,trainer.model,mb))
            torch.save(np.concatenate(trajectories),f"./{cfg['network']}_{cfg['net_config']['group']}_{i}.t") 
        except Exception as e:
            if self.strict: raise
            outcome = e
        del trainer
        return cfg, outcome
    
@export
class odeScalars_trial(object):
    """ A training trial for the Neural ODEs, contains lots of boiler plate which is not necessary.
        Feel free to use your own."""
    def __init__(self,make_trainer,strict=True):
        self.make_trainer = make_trainer
        self.strict=strict
    def __call__(self,cfg):
        try:
            cfg.pop('local_rank',None) #TODO: properly handle distributed
            resume = cfg.pop('resume',False)
            save = cfg.pop('save',False)
            i = cfg['trial']
            orig_suffix = cfg.setdefault('trainer_config',{}).get('log_suffix','')
            cfg['trainer_config']['log_suffix'] = os.path.join(orig_suffix,f'trial{i}/')
            trainer = self.make_trainer(**cfg)
            trainer.logger.add_scalars('config',flatten_dict(cfg))
            trainer.train(cfg['num_epochs'])
            if save: cfg['saved_at']=trainer.save_checkpoint()
            outcome = trainer.ckpt['outcome']
            trajectories = []
            for mb in trainer.dataloaders['test']:
                trajectories.append(pred_and_gt_ode(trainer.dataloaders['test'].dataset,trainer.model,mb)) 
            torch.save(np.concatenate(trajectories),f"{cfg['trainer_config']['log_dir']}/{'scalars_NODEs_modified'}_{i}.t")
        except Exception as e:
            if self.strict: raise
            outcome = e
        del trainer
        return cfg, outcome
    
    
@export
class hnnScalars_trial(object):
    """ A training trial for the HNNs, contains lots of boiler plate which is not necessary.
        Feel free to use your own."""
    def __init__(self,make_trainer,strict=True):
        self.make_trainer = make_trainer
        self.strict=strict
    def __call__(self,cfg):
        try:
            cfg.pop('local_rank',None) #TODO: properly handle distributed
            resume = cfg.pop('resume',False)
            save = cfg.pop('save',False)
            i = cfg['trial']
            orig_suffix = cfg.setdefault('trainer_config',{}).get('log_suffix','')
            cfg['trainer_config']['log_suffix'] = os.path.join(orig_suffix,f'trial{i}/')
            trainer = self.make_trainer(**cfg)
            trainer.logger.add_scalars('config',flatten_dict(cfg))
            trainer.train(cfg['num_epochs'])
            if save: cfg['saved_at']=trainer.save_checkpoint()
            outcome = trainer.ckpt['outcome']
            trajectories = []
            for mb in trainer.dataloaders['test']:
                trajectories.append(pred_and_gt(trainer.dataloaders['test'].dataset,trainer.model,mb))
            print(f"{cfg['trainer_config']['log_dir']}/{'scalars_HNNs_modified'}_{i}.t")
            torch.save(np.concatenate(trajectories), f"{cfg['trainer_config']['log_dir']}/{'scalars_HNNs_modified'}_{i}.t")
        except Exception as e:
            if self.strict: raise
            outcome = e
        del trainer
        return cfg, outcome
 
