"""Paths, result caching and the common command line of the experiment scripts.

Every experiment script has two stages, run (compute and cache results) and plot (draw the figures
and write the tables from the cache), and two profiles (the ``CONFIGS`` of each script): ``paper``, the
runs reported in the paper, and ``fast``, reduced sweeps of the same experiments:

    python -m experiments.<name> [--profile paper|fast] [--stage run|plot|all] [--jobs N] [--cpus SPEC]

Results go to results/<profile>/<experiment>/, figures to figures/<profile>/, tables to tables/<profile>/.
"""

from __future__ import annotations

import argparse
import os
import pickle
import shutil
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def results_dir(profile, exp):
    d = os.path.join(ROOT, "results", profile, exp)
    os.makedirs(d, exist_ok=True)
    return d


def figure_path(profile, name):
    d = os.path.join(ROOT, "figures", profile)
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, name)


def table_path(profile, name):
    d = os.path.join(ROOT, "tables", profile)
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, name + ".tex")


def dump(obj, path):
    """Atomic pickle (readers never see a partial file)."""
    with open(path + ".tmp", "wb") as fh:
        pickle.dump(obj, fh)
    os.replace(path + ".tmp", path)


def load(path):
    with open(path, "rb") as fh:
        return pickle.load(fh)


def cli(doc, configs, extra=None):
    """Parse --profile/--stage/--cpus (and script-specific arguments); returns (args, config)."""
    ap = argparse.ArgumentParser(description=doc, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--profile", choices=sorted(configs), default="paper")
    ap.add_argument("--stage", choices=("run", "plot", "all"), default="all")
    ap.add_argument("--cpus", default=None, help="CPUs for the workers, e.g. '0-7' or 'all' (default: one per physical core)")
    if extra:
        extra(ap)
    args = ap.parse_args()
    return args, dict(configs[args.profile])


def available_cpus():
    return sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else list(range(os.cpu_count() or 1))


def physical_cpus():
    """One logical CPU per physical core: the first of its hyperthread siblings (read from Linux sysfs;
    elsewhere every available CPU)."""
    seen, out = set(), []
    for c in available_cpus():
        try:
            with open(f"/sys/devices/system/cpu/cpu{c}/topology/thread_siblings_list") as fh:
                sib = fh.read().strip()
        except OSError:
            sib = str(c)
        if sib not in seen:
            seen.add(sib)
            out.append(c)
    return out


def parse_cpus(spec):
    """'0-3,6' -> [0, 1, 2, 3, 6]; 'all' -> every logical CPU; None -> one CPU per physical core."""
    if not spec:
        return physical_cpus()
    if spec == "all":
        return available_cpus()
    out = []
    for part in spec.split(","):
        a, _, b = part.partition("-")
        out += list(range(int(a), int(b or a) + 1))
    return out


def run_workers(module, profile, tasks, jobs, log_dir, cpus=None, extra_args=()):
    """Run ``python -m <module> --profile <profile> --worker <task> [extra_args]`` for every task, at most
    ``jobs`` at a time, each single-threaded (JAX would otherwise start ~50 threads per process) and
    pinned to one CPU of ``cpus`` if ``taskset`` is available.  Longest tasks should come first.
    Logs: <log_dir>/<task>.log"""
    import subprocess
    os.makedirs(log_dir, exist_ok=True)
    env = dict(os.environ, XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1",
               OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    pin = shutil.which("taskset") is not None
    pending, running, free = list(tasks), {}, parse_cpus(cpus)[:jobs]
    t0 = time.time()
    while pending or running:
        while pending and free:
            task, core = pending.pop(0), free.pop(0)
            log = open(os.path.join(log_dir, task.replace("/", "_").replace(":", "_") + ".log"), "w")
            cmd = (["taskset", "-c", str(core)] if pin else []) + [
                sys.executable, "-m", module, "--profile", profile, "--stage", "run", "--worker", task, *extra_args]
            running[subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)] = (task, core, log)
        time.sleep(1.0)
        for proc in [p for p in running if p.poll() is not None]:
            task, core, log = running.pop(proc)
            log.close()
            free.append(core)
            status = "ok" if proc.returncode == 0 else f"FAILED ({proc.returncode})"
            print(f"  [{time.time() - t0:6.0f}s] {task}: {status}", flush=True)


class Timer:
    def __init__(self, label):
        self.label = label

    def __enter__(self):
        self.t0 = time.time()
        return self

    def __exit__(self, *a):
        print(f"[{self.label}] {time.time() - self.t0:.1f}s", flush=True)
