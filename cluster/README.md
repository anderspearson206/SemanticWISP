# Training on Hyak (klone)

The laptop is where code, notebooks and results live. Hyak only runs jobs: it gets a copy of the code and the
data, and results are copied back. Hyak storage is scratch with no backup, so nothing should exist only there.

```
laptop (source of truth)                         Hyak: /gscratch/<group>/amp206/SemanticWISP
  semwisp/ experiments/ cluster/   -- push -->     same files
  data/ (conv_2B.zip)              -- data -->     data/      (uploaded once)
  results/  logs/                  <-- pull --     results/  logs/   (written by jobs)
  .venv (torch +cpu)                               env/  (conda, torch CUDA)   cache/
```

Not tested against the cluster yet (Hyak was down when this was written on 2026-11-10). The scripts were tested
locally against a stand-in for `ssh` and `sbatch`; the cluster facts come from the Hyak docs. See
[First-run checklist](#first-run-checklist).

## Files

| File | Runs on | Purpose |
|---|---|---|
| `config.env` | both | Host, project directory on Hyak, Slurm defaults. The only file to edit. |
| `hyak.ps1` | laptop | `push`, `submit`, `status`, `pull`, `data`, `run`, `ssh`. |
| `setup_env.sh` | Hyak | Builds the conda environment in `<HYAK_DIR>/env` from `requirements-hyak.txt`. |
| `submit.sh` | Hyak | Queues `python <script> <args>` as a Slurm job. |
| `job.sbatch` | Hyak | The job body that `submit.sh` submits. |
| `smoke_test.py` | both | Tiny training run that checks the whole chain; also the template for training scripts. |

## One-time setup

1. On Hyak, run `hyakalloc` to see your account(s) and partitions, and `hyakstorage` for your gscratch path.
   Put them in `config.env` (`HYAK_DIR`, `SLURM_ACCOUNT`).
2. From the laptop (each command asks for your password and Duo once):

   ```powershell
   cluster\hyak.ps1 push      # code
   cluster\hyak.ps1 data      # conv_2B.zip, 1.3 GB; prints 9332 when the JPEGs are unpacked
   ```
3. Build the environment on a compute node (klone login nodes have no `module` command):

   ```bash
   cluster\hyak.ps1 ssh                 # on the laptop; the rest is on Hyak
   salloc --account=<account> --partition=ckpt-all --cpus-per-task=4 --mem=16G --time=1:00:00
   bash cluster/setup_env.sh            # about 10 minutes; ends by printing the torch version
   exit
   ```
4. Check the chain end to end:

   ```powershell
   cluster\hyak.ps1 submit TIME=10:00 cluster/smoke_test.py
   cluster\hyak.ps1 status
   cluster\hyak.ps1 pull      # then read logs\smoke_test_<jobid>.out and results\_smoke\result.json
   ```
   The log should say `device cuda`. Delete `results\_smoke` on both sides afterwards.

## Everyday use

```powershell
# 1. develop and test locally on a small configuration
.venv\Scripts\python.exe experiments\train_x.py --epochs 1 --stride 20

# 2. send the same script to a GPU (pushes the code first)
cluster\hyak.ps1 submit experiments/train_x.py --epochs 200
cluster\hyak.ps1 submit TIME=1-00:00:00 GPUS=a40:1 MEM=64G experiments/train_x.py --epochs 200

# 3. watch, then fetch
cluster\hyak.ps1 status
cluster\hyak.ps1 pull
```

- `submit` arguments may only contain letters, digits and `_ . / : = , + @ -`. For anything else, use
  `cluster\hyak.ps1 ssh` and run `bash cluster/submit.sh ...` there.
- Job overrides (before the script name): `ACCOUNT PARTITION GPUS CPUS MEM TIME NAME`. `GPUS=0` is a CPU job;
  `GPUS=a40:1` asks for a GPU type (48 GB: `a40`, `l40`, `l40s`, `rtx6k`; 11 GB: `2080ti`).
- Logs go to `logs/<name>_<jobid>.out`. For a live view: `cluster\hyak.ps1 run tail -f logs/<file>`.
- `pull` keeps a local file if it is newer than the copy on Hyak.
- Jobs run the code that is on Hyak **when they start**, not when they were submitted. Avoid pushing changes to
  a script while a job that uses it is still queued.
- `$env:HYAK_DRYRUN = 1` makes `hyak.ps1` print its ssh command instead of running it.

## Rules for scripts that run on Hyak

The existing convention (resumable script in `experiments/`, one output file per configuration in
`results/<experiment>/`, skip if it exists) is what the cluster needs. In addition:

1. **Resume from a checkpoint.** Jobs on the `ckpt` partitions use other groups' idle GPUs. They are stopped and
   requeued every 8-9 hours for GPU jobs, or at any moment when the owner needs the node, and the script then
   starts again from the top. Save a checkpoint (model, optimizer, epoch) at least once per epoch, write it to a
   temporary name and rename it, and load it at start-up if it exists. `smoke_test.py` shows the pattern.
2. **No absolute paths.** Take paths from `semwisp.piropo.REPO` or from arguments. Data sits at the same relative
   path on both machines, so the defaults work unchanged.
3. **Device-agnostic**: `device = 'cuda' if torch.cuda.is_available() else 'cpu'`.
4. **Print with `flush=True`**, or progress will not show in the log until the job ends.
5. One job per configuration is easier to requeue and to read than one long job looping over many.

## Storage

| Where | Limit | Use |
|---|---|---|
| `/mmfs1/home/amp206` | 10 GB, about 250,000 files, fixed | Nothing from this project. Caches are redirected to `<HYAK_DIR>/cache`. |
| `/gscratch/<group>/` | group quota (1 TB per slice), shared with the group | The project directory. |
| `/gscratch/scrubbed/amp206` | 10 TB, files deleted after 21 days without access, world-readable by default | Large temporary files only. |

Check usage with `hyakstorage` (updated hourly). The project needs about 3 GB for data plus roughly 8 GB for
the environment.

## Changing packages

Edit `requirements-hyak.txt`, `push`, then rerun `bash cluster/setup_env.sh` on a compute node. Keep the versions
equal to the laptop `.venv` (`.venv\Scripts\python.exe -m pip freeze`) so results match on both machines.

## First-run checklist

Things that could not be checked while Hyak was down:

- [ ] `HYAK_DIR` and `SLURM_ACCOUNT` filled in from `hyakalloc`. If you have no lab group, the Research
      Computing Club's `stf` account is the usual route for students; confirm that you are a member.
- [ ] `module load conda` works on a compute node and can create a Python 3.13 environment. If 3.13 is not
      offered, use the newest available and note it here.
- [ ] `setup_env.sh` prints a CUDA version (`built with CUDA 12.x`), which confirms that PyPI served the GPU
      build of torch.
- [ ] Compute nodes can reach the internet. `coco_detector_eval.py` downloads pretrained weights on first use.
      If that fails in a job, run it once on the login node with `TORCH_HOME=<HYAK_DIR>/cache/torch` set, just far
      enough to download the weights.
- [ ] `--gpus=1` is accepted on `ckpt-all`. The docs' example uses `--gpus-per-node=2080ti:1`; if the plain form
      is rejected, set `SLURM_GPUS` to a typed value such as `a40:1`.
- [ ] The smoke test log says `device cuda`.

## Later

- **Version control.** `git init` here and a private GitHub repository would give history and a second copy of
  the code. `push` already skips `.git`; add `data/`, `results/`, `logs/`, `.venv/` and `papers/` to `.gitignore`.
- **Faster data loading.** Decoding 9,332 JPEGs each epoch will dominate small-model training. Cache the 160x120
  grayscale frames once as a single array (about 180 MB) and train from that.
- **Sweeps.** For many configurations, a Slurm job array (`sbatch --array`) over a list of argument lines is
  the next step from `submit.sh`.
