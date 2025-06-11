#!/usr/bin/env python3
import subprocess
import os
import time
import csv
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

# Configuration
TASK_DIR = Path("./datasets/")
GLOB = "**/vlsat3_g[5-9][0-9].smt2"
MAX_PARALLEL = 16
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
RESULTS_DIR = Path(f"./results/{timestamp}/")
MEMORY_LIMIT = "2500MB"
TIME_LIMIT = "00:20:00"

RESULTS_DIR.mkdir(exist_ok=True)

# commands = [
# 	("z3-bit-blast", "z3 -smt2 \"tactic.default_tactic='(then simplify propagate-values solve-eqs ctx-simplify simplify smt)'\""),
# 	("z3-int-blast", "z3 sat.smt=true smt.bv.solver=2 -smt2 \"tactic.default_tactic='(then simplify propagate-values solve-eqs ctx-simplify simplify smt)'\""),
# 	("z3-sls-and-bit-blasting-sequential", "z3 smt.sls.enable=true smt.sls.parallel=false -smt2 \"tactic.default_tactic='(then simplify propagate-values solve-eqs ctx-simplify simplify smt)'\"")
# ]

commands = [
	("z3-bit-blast", ["z3", "-smt2", "tactic.default_tactic='(then simplify propagate-values solve-eqs ctx-simplify simplify smt)'"]),
	("z3-int-blast", ["z3", "sat.smt=true", "smt.bv.solver=2", "-smt2", "tactic.default_tactic='(then simplify propagate-values solve-eqs ctx-simplify simplify smt)'"]),
	("z3-sls-and-bit-blasting-sequential", ["z3", "smt.sls.enable=true", "smt.sls.parallel=false", "-smt2", "tactic.default_tactic='(then simplify propagate-values solve-eqs ctx-simplify simplify smt)'"])
]


def run_task(task_command_pair: tuple[Path, tuple[str, list[str]]]):
	(task_path, (command_name, command)) = task_command_pair
	task_name = task_path.name
	out_file = RESULTS_DIR / f"{task_name}_{command_name}.out"
	err_file = RESULTS_DIR / f"{task_name}_{command_name}.err"

	timing_command = ["/usr/bin/time", "--verbose"]
	timing_command.extend(command)
	timing_command.append(str(task_path))
	#print(str(task_path))
	# timing_command = f"/usr/bin/time --verbose {command} {task_path}"
	#print(timing_command)
	srun_cmd = ["srun", "--exact", "--ntasks=1", "--cpus-per-task=1", f"--mem-per-cpu={MEMORY_LIMIT}", f"--time={TIME_LIMIT}"]
	srun_cmd.extend(timing_command)
	# srun_cmd = f"srun --exclusive --ntasks=1 --cpus-per-task=1 --mem-per-cpu={MEMORY_LIMIT} {timing_command}"

	with open(out_file, 'w') as stdout, open(err_file, 'w') as stderr:
		subprocess.run(srun_cmd, shell=False, stdout=stdout, stderr=stderr)

	return (task_name + " " + command_name)

def main():
	task_files = list(TASK_DIR.glob(GLOB))

	task_command_pairs = []

	for task in task_files:
		for command in commands:
			task_command_pairs.append((task, command))

	with ThreadPoolExecutor(max_workers=MAX_PARALLEL) as executor:
		futures = {executor.submit(run_task, task): task for task in task_command_pairs}
		for future in as_completed(futures):
			print(datetime.now().strftime("%Y%m%d_%H%M%S") + " " + future.result())

if __name__ == "__main__":
	main()
