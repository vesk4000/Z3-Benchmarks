#!/usr/bin/env python3
import random
import subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import argparse

# Configuration
TASK_DIR = Path("./datasets/")
GLOB = "**/vlsat3_g[5-9][0-9].smt2"
MAX_PARALLEL = 16
MEMORY_LIMIT = "2500MB"
TIME_LIMIT = "00:20:00"
PARALLEL = 1
SRUN = True

four_horsemen = [
	{
		"name": "z3-bit-blast",
		"command": ["z3", "-smt2", "tactic.default_tactic='(then simplify propagate-values solve-eqs ctx-simplify simplify smt)'"]
	},
	{
		"name": "z3-int-blast",
		"command": ["z3", "sat.smt=true", "smt.bv.solver=2", "-smt2", "tactic.default_tactic='(then simplify propagate-values solve-eqs ctx-simplify simplify smt)'"],
	},
	{
		"name": "z3-lazy-bit-blast",
		"command": ["z3", "--poly", "sat.smt=true", "smt.bv.solver=1", "-smt2", "tactic.default_tactic='(then simplify propagate-values solve-eqs ctx-simplify simplify smt)'"],
	},
	{
		"name": "z3-sls-and-bit-blasting-sequential",
		"command": ["z3", "smt.sls.enable=true", "smt.sls.parallel=false", "-smt2", "tactic.default_tactic='(then simplify propagate-values solve-eqs ctx-simplify simplify smt)'"],
	}
]

benches = {
	"vlsat3_a": {
		"version": 1,
		"glob": "VLSAT3/**/vlsat3_a*.smt2",
		"time_limit": "00:40:00",
		"memory_limit": "3000MB",
		"threads": 31,
		"commands": four_horsemen,
	},
	"vlsat3_g": {
		"version": 1,
		"glob": "VLSAT3/**/vlsat3_g*.smt2",
		"time_limit": "00:40:00",
		"memory_limit": "3000MB",
		"threads": 31,
		"commands": four_horsemen,
	},
	"smt-comp_2024": {
		"version": 1,
		"glob": "COWABUNGA",
		"time_limit": "00:01:30",
		"memory_limit": "3000MB",
		"threads": 64,
		"commands": four_horsemen,
		"task_list": "SMT-COMP_2024_tasks.txt",
	},
	"smart_contracts": {
		"version": 1,
		"glob": "Smart_Contract_Verification/**/*.smt2",
		"time_limit": "01:00:00",
		"memory_limit": "12000MB",
		"threads": 16,
		"commands": four_horsemen,
	}
}



def run_task(task_command_pair: tuple[Path, tuple[str, list[str]]]):
	(task_path, cmd) = task_command_pair
	command_name = cmd["name"]
	command = cmd["command"]
	task_name = task_path.name

	# Calculate relative path from TASK_DIR to maintain directory structure
	relative_path = task_path.relative_to(TASK_DIR)
	result_subdir = RESULTS_DIR / relative_path.parent
	
	# Create the subdirectory structure if it doesn't exist
	result_subdir.mkdir(parents=True, exist_ok=True)
	
	out_file = result_subdir / f"{task_name}_{command_name}.out"
	err_file = result_subdir / f"{task_name}_{command_name}.err"

	timing_command = ["/usr/bin/time", "--verbose"]
	timing_command.extend(command)
	timing_command.append(str(task_path))
	
	final_cmd = timing_command
	if SRUN:
		final_cmd = ["srun", "--exact", "--nodes=1", "--ntasks=1", f"--cpus-per-task={PARALLEL}", f"--mem-per-cpu={MEMORY_LIMIT}", f"--time={TIME_LIMIT}"]
		final_cmd.extend(timing_command)
	
	print("Starting: " + datetime.now().strftime("%Y%m%d_%H%M%S") + " " + str(relative_path) + " with command: " + command_name)

	with open(out_file, 'w') as stdout, open(err_file, 'w') as stderr:
		subprocess.run(final_cmd, shell=False, stdout=stdout, stderr=stderr)

	return (str(relative_path) + " " + command_name)

def main():
	global TASK_DIR, GLOB, MAX_PARALLEL, MEMORY_LIMIT, TIME_LIMIT, RESULTS_DIR, SRUN, PARALLEL

	parser = argparse.ArgumentParser(description="Run SMT2 tasks under srun with configurable limits")
	parser.add_argument("--name", default="DEFAULT", help="name for this benchmark run")
	parser.add_argument("--time-limit", default="DEFAULT", help="time limit for each task, e.g. '00:20:00'")
	parser.add_argument("--memory-limit", default="DEFAULT", help="memory limit for each task, e.g. '3000MB'")
	parser.add_argument("--glob", default="DEFAULT", help="glob pattern for tasks")
	parser.add_argument("--threads", type=int, default=-1, help="max number of parallel tasks")
	parser.add_argument("--task-list", default="DEFAULT", help="file containing a list of files to run")
	parser.add_argument("--no-srun", action="store_false")
	parser.add_argument("--parallel", type=int, default=1, help="number of parallel threads to use for each task (default: 1)")
	args = parser.parse_args()

	TIME_LIMIT = benches[args.name]["time_limit"] if args.time_limit == "DEFAULT" else args.time_limit
	MEMORY_LIMIT = benches[args.name]["memory_limit"] if args.memory_limit == "DEFAULT" else args.memory_limit
	GLOB = benches[args.name]["glob"] if args.glob == "DEFAULT" else args.glob
	MAX_PARALLEL = benches[args.name]["threads"] if args.threads == -1 else args.threads
	PARALLEL = args.parallel
	task_list = benches[args.name].get("task_list", "DEFAULT") if args.task_list == "DEFAULT" else args.task_list

	SRUN = args.no_srun
	
	timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
	RESULTS_DIR = Path(f"./results/{timestamp}_{args.name}/")
	RESULTS_DIR.mkdir(exist_ok=True)
	
	task_files = []
	if task_list == "DEFAULT":
		task_files = list(TASK_DIR.glob(GLOB))
	else:
		with open(Path(task_list), 'r') as f:
			for line in f:
				task_path = TASK_DIR / line.strip()
				task_files.append(task_path)
	
	random.seed(42)
	random.shuffle(task_files)

	task_command_pairs = []
	for task in task_files:
		for command in benches[args.name]["commands"]:
			task_command_pairs.append((task, command))

	with ThreadPoolExecutor(max_workers=MAX_PARALLEL) as executor:
		futures = {executor.submit(run_task, task): task for task in task_command_pairs}
		for future in as_completed(futures):
			print("Finished: " + datetime.now().strftime("%Y%m%d_%H%M%S") + " " + future.result())

if __name__ == "__main__":
	main()
