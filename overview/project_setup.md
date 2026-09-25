# Project setup: reproducing every number in `output/results.md`

All commands are run from the repository root on Rockfish. GPU jobs use the account `enalisn1_gpu`, QoS `qos_gpu`,
partition **`l40s`** (the cheapest GPU and, for this small model, not slower than an A100; see "Compute cost" below).

## 1. Environment (once)
```bash
bash overview/env_setup.sh
```
Key versions: python 3.13, jax/jaxlib 0.7.1 (cuda12), objax 1.8.0, torch 2.8, olive-oil-ml 0.1.1, numpy 2.3.

## 2. Dataset (once)
```bash
bash overview/data_prepare.sh            # or: cd SLURM_execution/SLURM_script && sbatch gpu_check.sh
```
`gpu_check.sh` also generates the dataset and times a 50-epoch run of each of the four model variants.

## 3. A single run
```bash
source /data/svillar3/ylu174/Anaconda3/etc/profile.d/conda.sh && conda activate scalar_mlp_gpu
cd ScalarEMLP_bilipschitz
export SCALAREMLP_DATA_DIR=$PWD/datasets
python experiments/run_dsp.py --model hnn --bilip 1 --n_layers 3 --n_hidden 100 --lr 5e-3 \
       --seed 0 --train_noise 0.03 --out_dir ../experiment/debug/hnn_bilip_example
```
Arguments: `--model {hnn,node}`, `--bilip {0,1}` (inner product `XX^T` vs bilipschitz `(XX^T)^{1/2}`),
`--n_layers --n_hidden --lr --n_rad --epochs(2000)`, `--seed` (model init / batch order; the dataset and split
are always seed 2021), `--train_noise` (relative Gaussian noise on training data), `--eval_noise` (relative noise
levels on test initial conditions for the robustness evaluation).

## 4. The full study (SLURM job arrays)
Each stage writes a task file (one argument line per run) and submits `run_tasks.sh`, which runs `PER_GPU`
lines concurrently on one GPU and skips runs whose `result.json` already exists (safe to resubmit).
```bash
PY="python ScalarEMLP_bilipschitz/experiments"
cd SLURM_execution/SLURM_script
submit () {  # $1 = stage, $2 = number of lines
  sbatch -J dsp_$1 --array=0-$(( ($2 + 3) / 4 - 1 )) \
         --export=ALL,TASK_FILE=$(realpath ../SLURM_outcome/tasks/$1.txt),PER_GPU=4,OVERWRITE_LOGS=false run_tasks.sh
}
cd ../..
$PY/make_tasks.py paper  && (cd SLURM_execution/SLURM_script && submit paper 12)   # stage 1: paper setting x 3 seeds
$PY/make_tasks.py tune   && (cd SLURM_execution/SLURM_script && submit tune 108)   # stage 2: 3x3x3 grid, seed 0
$PY/make_tasks.py probe  && (cd SLURM_execution/SLURM_script && submit probe 20)   # stage 2b: activations / RBF width
$PY/make_tasks.py tune2  && (cd SLURM_execution/SLURM_script && submit tune2 48)   # stage 2c: HNN smooth activations
$PY/make_tasks.py tune3  && (cd SLURM_execution/SLURM_script && submit tune3 38)   # stage 2d: larger lr / width
# after stage 2 finished (selection by validation rollout, never by test):
$PY/make_tasks.py final  && (cd SLURM_execution/SLURM_script && submit final 8)    # stage 3: seeds 1-2 of best configs
$PY/make_tasks.py noise  && (cd SLURM_execution/SLURM_script && submit noise <N>)  # stage 4: training noise
$PY/summarize.py                                                                   # -> output/results.md, figures
```
Walltime: one 2000-epoch run takes 8-20 min on an L40S with 4-6 runs sharing the GPU (the bilipschitz
models are ~2x slower than the inner-product ones: the SVD sits inside the vector field).

## Compute cost (Rockfish billing)
SU/hour of a job = max(#CPU × 1, mem_GB × 0.25, #GPU × w) with w = 8 (l40s), 12 (a100), 16 (ica100).
`run_tasks.sh` asks for 1 GPU, 8 CPU, 36 GB (4 runs × ~7 GB peak RSS) → 9 SU/h on l40s.
Measured median minutes per run (6 per GPU): bilip HNN 15 (l40s) / 23 (a100) / 56 (ica100);
inner N-ODE 8 (l40s) / 21 (ica100). The first stages ran with 12 CPU / 64 GB on all three partitions
(16 SU/h, memory-bound); from tune3 on only l40s is used.

## 5. Where things end up
- `experiment/<stage>/<run>/result.json`: final metrics (last logged row, as upstream), `val_Rollout`,
  `rollout_vs_ic_noise`, full metric history, timings; `model.npz`: trained objax variables; `log.txt`: stdout.
- `SLURM_execution/SLURM_output/dsp_<stage>_slurm_<job>_<task>.{out,err}` and `SLURM_outcome/time_*.txt`.
- `output/results.md`: all tables of the study; `output/noise_robustness.png`; `output/best_configs.json`.
