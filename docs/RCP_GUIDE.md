# CS-552 Project — RCP Setup & Usage Guide

Interactive Jupyter Lab on an A100 with PyTorch, vLLM, and the Hugging
Face stack already installed. No Docker building required for most teams.
The included `Dockerfile` and `build.sh` are only a base for teams that
genuinely need a custom image.

For the official project scope, grading weights, rubrics, report
requirements, and deadlines, refer to your [project description](https://docs.google.com/document/d/1TECHv4q_eR0X-HIyW10vHFbcU2bHLSph/edit?usp=sharing&ouid=117565422542731737285&rtpof=true&sd=true). This guide focuses on the compute environment and
the files needed to run your code.

Use `/scratch` for caches, datasets, checkpoints, and large generated
files — not as the only place where notebooks or code live (commit those
to your repo).

---

## One-time setup

### 1. Connect to the EPFL VPN

You must be on the EPFL VPN to talk to the cluster. EPFL uses **Cisco
Secure Client** (formerly AnyConnect):

1. Download and install the client for your OS from
   [VPN clients available](https://www.epfl.ch/campus/services/en/it-services/network-services/remote-intranet-access/vpn-clients-available/).
2. Open it, set the gateway to `vpn.epfl.ch`, click *Connect*.
3. Sign in with your **GASPAR** credentials (same as Tequila/SSO).

If you hit issues, see [Remote Intranet Access](https://www.epfl.ch/campus/services/en/it-services/network-services/remote-intranet-access/).

### 2. Install the Run:AI CLI and log in

1. Sign in to <https://rcpepfl.run.ai/> with **Sign in with SSO** (your
   EPFL credentials). This also confirms your account has cluster
   access.
2. Download the CLI: top-right **help icon (?) → Researcher command
   line interface → your OS**. Direct binaries:
   - macOS: `https://rcp-caas-prod.rcp.epfl.ch/cli/darwin`
   - Linux: `https://rcp-caas-prod.rcp.epfl.ch/cli/linux`
   - Windows: `https://rcp-caas-prod.rcp.epfl.ch/cli/windows`
3. Make it executable and put it on your PATH (macOS/Linux):
   ```bash
   chmod +x ./runai
   sudo mv ./runai /usr/local/bin/runai
   ```
   On Windows, place `runai.exe` somewhere on your `PATH` (or call it
   by full path).
4. Download the kubeconfig from
   <https://wiki.rcp.epfl.ch/public/files/kube-config.yaml> and save it
   as `~/.kube/config` (create the `.kube` folder if it doesn't exist;
   on Windows that's `%USERPROFILE%\.kube\config`).
5. Configure the cluster and log in:
   ```bash
   runai config cluster rcp-caas-prod
   runai login        # opens a URL in your browser, paste the code shown
   ```

If anything above behaves differently on your setup, the canonical
reference is the [RCP Quick Start](https://wiki.rcp.epfl.ch/home/CaaS/Quick_Start)
(troubleshooting/Windows specifics live there).

### 3. Set your project context

Replace `<gaspar>` with your username:

```bash
runai config project course-cs-552-<gaspar>
```

`submit.sh` also passes this project explicitly when submitting the job,
so it does not depend on any other default Run:AI project.

### 4. Edit `submit.sh`

Set `GASPAR="gaspar"` to your EPFL username (e.g. `jdupont`) and
`GROUP="gXX"` to your team number (e.g. `g07`). **Required** — the
script refuses to run with either placeholder.

### 5. *(Optional)* Export tokens in your shell so jobs pick them up

```bash
export HF_TOKEN=hf_xxx
export WANDB_API_KEY=xxx
```

## Launch a job

Each job runs on **1 GPU (40GB A100)** — the course cap for this setup.
Asking for more leaves the job stuck `Pending`.

```bash
./submit.sh           # default
./submit.sh train     # custom job suffix
```

Wait for `Running`:
```bash
runai describe job <job-name>
```

In a second terminal, forward the port:
```bash
runai port-forward <job-name> --port 8888:8888
# Open http://localhost:8888 — token is "cs552"
```

This port-forward command is run after the job exists. Do not add
`--service-type portforward` to `runai submit`; some Run:AI CLI versions
reject it because `port-forward` is a client-side command.

> If you see `address already in use` on `8888`, change the **left**
> side to any free local port: `--port 9000:8888`, then open
> `http://localhost:9000`. The same `--port LOCAL:REMOTE` form works
> for any service running in the pod (vLLM OpenAI server on `8000`,
> tensorboard on `6006`, etc.).

When you're done **(read this please)**:
```bash
runai delete job <job-name>
```

### My job is `Pending` forever / crashed — what now?

Two commands cover 95% of cases:

```bash
runai describe job <job-name>   # why it's Pending, last events, current state
runai logs <job-name>           # stdout/stderr from the container
runai logs -f <job-name>        # follow live
```

If `runai logs` shows a Python traceback, the container started fine —
fix the code and resubmit.

## Connecting to your pod

You have three ways to interact with a running pod. Use whichever fits
the task.

### 1. Jupyter Lab (the default)

Already covered above — wait until the job is `Running`, run
`runai port-forward`, then open `http://localhost:8888`. Best for
notebook-driven exploration and plots.

### 2. A shell in the pod (`runai bash`)

For quick CLI work — running scripts, checking GPU usage, installing
packages, debugging — you don't need Jupyter at all:

```bash
runai bash <job-name>
```

You're now inside the container with a normal shell. Useful examples:

```bash
nvidia-smi                              # check GPU state
df -h /scratch                          # how much scratch space is left
python my_script.py                     # run something quickly
pip install some-extra-package          # ad-hoc install for this session
```

You can have a Jupyter port-forward running in one terminal *and* a
`runai bash` open in another, on the same pod.

### 3. VS Code attached to the pod

If you prefer VS Code over Jupyter for editing code, you can attach VS
Code directly to your running pod and edit files inside it as if they
were local. Full setup guide from RCP:
<https://wiki.rcp.epfl.ch/home/CaaS/FAQ/how-to-vscode>

Short version:

1. **Install VS Code** from <https://code.visualstudio.com>.
   The official Microsoft build is required — VSCodium does **not**
   work with the Kubernetes attachment flow.
2. **Install two extensions** from the VS Code Marketplace, both from
   Microsoft:
   - [Kubernetes](https://marketplace.visualstudio.com/items?itemName=ms-kubernetes-tools.vscode-kubernetes-tools)
   - [Remote Development](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.vscode-remote-extensionpack)
   Be careful — there are several Kubernetes extensions on the
   Marketplace; only the official Microsoft one is supported.
3. **Set up your kubeconfig** at `~/.kube/config` — same file you
   downloaded in *One-time setup* step 2. The Kubernetes extension
   reads it automatically.
4. **Attach to your pod.** Click the Kubernetes icon in the left
   sidebar, expand your namespace under the cluster, find your running
   pod (named `<job-name>-0-0`), right-click → **Attach Visual
   Studio Code**. A new VS Code window opens, connected to the pod.
   The bottom-left status bar shows which pod you're attached to.
5. **Open files**: File → Open Folder, then type a path inside the pod
   (e.g. `/scratch`). You can edit files in place.
6. **Open a terminal**: Terminal → New Terminal opens a shell in the
   pod, same as `runai bash`.

VS Code is the most ergonomic option for serious code editing during
the project. Jupyter is still convenient for quick exploration and
plots.

## I need a package that isn't in the image

Three options, in order of preference:

1. **`pip install` from a notebook cell or CLI** — works for the session, takes
   seconds. Fine for one-off experiments.
2. **`requirements.txt` in your repo** — keep a `requirements.txt` with all the requirements in each line and install it at the beginning of each session using CLI or a jupyter cell using `pip install -r requirements.txt`. 
3. **Build your own image.** If your project genuinely needs something
   that can't be pip-installed (custom CUDA kernels, weird system libs),
   you can use the included `Dockerfile` and `build.sh` as a base in the `docker` folder. The
   image lives on **your own public Docker Hub repo** so the cluster
   can pull it without credentials.
   - Create a free Docker Hub account and a **public** repo: see
     [Docker Hub Quickstart](https://docs.docker.com/docker-hub/quickstart/)
     and [Creating repositories](https://docs.docker.com/docker-hub/repos/create/).
   - One-time, on your laptop: `docker login` with your Docker Hub
     credentials.
   - Edit `build.sh` — set `DOCKERHUB_USER` to your Docker Hub
     username, `IMAGE` to your repo name, and `DOCKERFILE` if your
     Dockerfile lives at a non-default path.
   - Build & push:
     ```bash
     ./build.sh           # builds & pushes :v1
     ./build.sh v2        # builds & pushes :v2
     ```
   - Copy the printed image name into `submit.sh` as
     `IMAGE="docker.io/<dockerhub-user>/<repo>:<tag>"`. Keep the
     Docker Hub repo **public** so the cluster can pull without
     credentials.
   - The initial build can take ~30 minutes, so plan for that.

Stick with the course image unless you have a concrete reason to build a
custom one. Also, remember that the evaluation pipeline runs on the base image, and does not have access to your updated packages.

## Storage layout inside the pod

| Path | What it is | Access |
|---|---|---|
| `/scratch` | Team scratch | shared with your group, RW |
| `/shared-ro/datasets` | Course datasets | read-only, all students |
| `/shared-ro/models` | Course base models | read-only, all students |
| `/shared-rw` | Course-wide writable scratch | RW for **everyone** — careful |

**Use `/scratch` for everything heavy** — clone your repo there, save
model checkpoints, store the HF cache, log wandb runs. The HF cache is
already pointed at `/scratch/hf_cache`, so `from_pretrained()` will
download there automatically and your teammates will see the cached
files.

Keep code and notebooks in your repo, not as loose files in `/scratch`.
It is fine to edit a repo clone that lives under `/scratch`, but commit
your work — `/scratch` is not a substitute for version control.

> ⚠️ `/shared-rw` is writable by **all 285 students**. Don't put anything
> sensitive there, and don't rely on files in it persisting — anyone can
> overwrite or delete them.

> ℹ️ Anything in `/scratch` will be wiped end of July 2026.

## GPU etiquette (please read)

The course has **75 A100s shared across ~285 students**. The scheduler
caps each allocation at 1 GPU at a time, but doesn't otherwise prevent
you from holding it indefinitely. A few habits keep things working for
everyone, especially around the May 24 and June 7 deadlines:

- **Delete idle Jupyter jobs.** If you walk away from your laptop for
  more than ~30 minutes, `runai delete job <name>`. You can resubmit in
  ~5 seconds when you come back.
- **Use `--interactive` for exploration, not for long training.** This
  is what `submit.sh` does by default. Interactive jobs are preemptible,
  so the scheduler can reclaim them when capacity is tight — that's the
  right behavior for a notebook session.
- **For long final training runs, submit a non-interactive training
  job.** Those are non-preemptible (won't be killed mid-epoch) but you
  can't sit on them while idle. Same flags as `submit.sh`, minus
  `--interactive`, and with your training command instead of
  `jupyter lab`:
  ```bash
  runai submit \
    --name "cs552-${GASPAR}-${GROUP}-train-$(date +%H%M%S)" \
    -p "course-cs-552-${GASPAR}" \
    --image "${IMAGE}" \
    --gpu 1 --large-shm --node-pools a100-40g \
    --working-dir /scratch \
    --environment HF_HOME=/scratch/hf_cache \
    --environment HF_TOKEN="${HF_TOKEN:-}" \
    --environment WANDB_API_KEY="${WANDB_API_KEY:-}" \
    --existing-pvc "claimname=course-cs-552-scratch-${GROUP},path=/scratch" \
    --existing-pvc "claimname=course-cs-552-shared-ro,path=/shared-ro" \
    --existing-pvc "claimname=course-cs-552-shared-rw,path=/shared-rw" \
    --command -- /bin/bash -lc "\
      ln -sf \"\$(command -v python3)\" /usr/local/bin/python && \
      cd /scratch/<your-repo> && python train.py"
  ```
  Stream logs with `runai logs -f <job-name>`. The job exits on its own
  when the script finishes — still run `runai delete job` if you abort.
- **Expect queues during deadline week.** Plan compute-heavy work
  earlier, not the night before.
