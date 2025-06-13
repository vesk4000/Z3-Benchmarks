import shlex
import subprocess
import os
import time
import csv
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

TASK_DIR = Path("./datasets/")
GLOB = "**/vlsat3_a[5-9][0-9].smt2"
MAX_PARALLEL = 16
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
RESULTS_DIR = Path(f"./results/{timestamp}/")
MEMORY_LIMIT = "2500MB"
TIME_LIMIT = "00:20:00"
SCRIPT_FILE = Path("benchy_manual_no_exact.sh")
RESULTS_DIR.mkdir(exist_ok=True)


commands = [
	("z3-bit-blast", ["z3", "-smt2", "tactic.default_tactic='(then simplify propagate-values solve-eqs ctx-simplify simplify smt)'"]),
	("z3-int-blast", ["z3", "sat.smt=true", "smt.bv.solver=2", "-smt2", "tactic.default_tactic='(then simplify propagate-values solve-eqs ctx-simplify simplify smt)'"]),
	("z3-sls-and-bit-blasting-sequential", ["z3", "smt.sls.enable=true", "smt.sls.parallel=false", "-smt2", "tactic.default_tactic='(then simplify propagate-values solve-eqs ctx-simplify simplify smt)'"])
]

def main():
    task_files = list(TASK_DIR.glob(GLOB))

    task_command_pairs = []
    for task in task_files:
        for command_name, command in commands:
            task_command_pairs.append((task, (command_name, command)))

    # build one big line of srun calls separated by ampersands
    lines = []
    for task_path, (command_name, command) in task_command_pairs:
        task_path = str(task_path).replace("\\", "/")
        timing = ["/usr/bin/time", "--verbose"] + command + [str(task_path)]
        srun_cmd = [
            "srun", "--exclusive",
            "--ntasks=1", "--cpus-per-task=1",
            f"--mem-per-cpu={MEMORY_LIMIT}",
            f"--time={TIME_LIMIT}"
        ] + timing
        print(str(task_path))
        # shell‐escape and join
        lines.append(shlex.join(srun_cmd))

    # print them all on one line, separated by " & "
    # print(" & \n".join(lines))
    
	    # append to run_all.sh
    with SCRIPT_FILE.open("a") as f:
        for cmd in lines:
            f.write(cmd)
            f.write(" &\n")
        f.write("wait\n")
        f.write("\n")

if __name__ == "__main__":
    main()