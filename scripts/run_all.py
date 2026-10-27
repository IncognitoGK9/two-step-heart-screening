"""
run_all.py - one command for the whole study: training -> ablation ladder -> evaluation.

    python scripts/run_all.py            # full study (3 x 10 outer folds, 5 configurations)
    python scripts/run_all.py --quick    # 1 outer repeat
    python scripts/run_all.py --jobs 4   # limit CPU cores
Library versions and timings go to results/run_log.txt.
"""
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parent / "results"
RESULTS.mkdir(exist_ok=True)
args = sys.argv[1:]
log = open(RESULTS / "run_log.txt", "w")


def say(s):
    print(s, flush=True)
    log.write(s + "\n")
    log.flush()


import joblib, matplotlib, numpy, pandas, scipy, sklearn  # noqa: E401
say(f"run started {time.strftime('%Y-%m-%d %H:%M:%S')} | args: {args}")
say(f"python {platform.python_version()} | scikit-learn {sklearn.__version__} | pandas {pandas.__version__} | "
    f"numpy {numpy.__version__} | scipy {scipy.__version__} | matplotlib {matplotlib.__version__} | joblib {joblib.__version__}")
say(f"cpu cores: {os.cpu_count()} | platform: {platform.platform()}")
t_all = time.time()
export_only = "--export-only" in args
args = [a for a in args if a != "--export-only"]
steps = [("06_export_models.py", args)] if export_only else [
    ("02_train_models.py", args), ("03_ladder.py", args), ("04_evaluate.py", []), ("05_site_validation.py", args),
    ("06_export_models.py", args)]
for script, a in steps:
    t0 = time.time()
    say(f"\n=== {script} ===")
    p = subprocess.Popen([sys.executable, str(HERE / script), *a], cwd=HERE, stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, text=True, env={**os.environ, "PYTHONWARNINGS": "ignore"})
    for line in p.stdout:
        if script != "04_evaluate.py" or line.startswith(("Traceback", " ", "Error")):
            say(line.rstrip())
    if p.wait():
        say(f"FAILED: {script}")
        sys.exit(1)
    say(f"--- {script} finished in {time.time() - t0:.0f}s")
say(f"\nAll done in {time.time() - t_all:.0f}s. See results/summary.md and figures/.")
