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

concurrent_horsemen = [
	{
		"name": "CDCL_16",
		"command": ["z3", "sat.threads=16", "-smt2"]
	},
	{
		"name": "LS_16",
		"command": ["z3", "sat.threads=0", "sat.local_search_threads=16", "-smt2"]
	},
	{
		"name": "DDFW_16",
		"command": ["z3", "sat.threads=0", "sat.ddfw.threads=16", "-smt2"]
	},
	{
		"name": "PCC_def",
		"command": ["z3", "parallel.enable=true", "parallel.threads.max=16", "-smt2"]
	},
	{
		"name": "PCC_lowBatch",
		"command": ["z3", "parallel.enable=true", "parallel.threads.max=16", "parallel.conquer.batch_size=100", "parallel.conquer.delay=10", "-smt2"]
	},
	{
		"name": "PCC_hiBatch",
		"command": ["z3", "parallel.enable=true", "parallel.threads.max=16", "parallel.conquer.batch_size=1000", "parallel.conquer.delay=10", "-smt2"]
	},
	{
	"name": "PCC_noDelay",
	"command": ["z3", "parallel.enable=true", "parallel.threads.max=16", "parallel.conquer.batch_size=100", "parallel.conquer.delay=0", "-smt2"]
	},
	{
	"name": "PCC_hiDelay",
	"command": ["z3", "parallel.enable=true", "parallel.threads.max=16", "parallel.conquer.batch_size=100", "parallel.conquer.delay=50", "-smt2"]
	},
	{
	"name": "PCC8_CDCL8",
	"command": ["z3", "parallel.enable=true", "parallel.threads.max=8", "sat.threads=8", "-smt2"]
	},
	{
	"name": "PCC8_LS8",
	"command": ["z3", "parallel.enable=true", "parallel.threads.max=8", "sat.threads=0", "sat.local_search_threads=8", "-smt2"]
	},
	{
	"name": "PCC8_DDFW8",
	"command": ["z3", "parallel.enable=true", "parallel.threads.max=8", "sat.threads=0", "sat.ddfw.threads=8", "-smt2"]
	},
	{
	"name": "PCC4_CDCL12",
	"command": ["z3", "parallel.enable=true", "parallel.threads.max=4", "sat.threads=12", "-smt2"]
	},
	{
	"name": "PCC12_CDCL4",
	"command": ["z3", "parallel.enable=true", "parallel.threads.max=12", "sat.threads=4", "-smt2"]
	},
	{
	"name": "LS8_DDFW8",
	"command": ["z3", "sat.threads=0", "sat.local_search_threads=8", "sat.ddfw.threads=8", "-smt2"]
	},
	{
	"name": "CDCL8_LS8",
	"command": ["z3", "sat.threads=8", "sat.local_search_threads=8", "-smt2"]
	},
	{
	"name": "CDCL8_DDFW8",
	"command": ["z3", "sat.threads=8", "sat.ddfw.threads=8", "-smt2"]
	},
	{
	"name": "CDCL6_LS5_DDFW5",
	"command": ["z3", "sat.threads=6", "sat.local_search_threads=5", "sat.ddfw.threads=5", "-smt2"]
	}
]


from typing import List, Dict

def get_z3_parallel_configs(threads: int = 8) -> List[Dict]:
    """
    Generate a list of Z3 parallel configurations (each uses exactly `threads` total threads).
    Returns a list of dicts: {'name': str, 'command': List[str]}
    """
    configs = []
    # 1) Pure modes
    configs.append({
        "name": f"CDCL_{threads}",
        "command": ["z3", f"sat.threads={threads}", "-smt2"]
    })
    configs.append({
        "name": f"LS_{threads}",
        "command": ["z3", "sat.threads=0", f"sat.local_search_threads={threads}", "-smt2"]
    })
    configs.append({
        "name": f"DDFW_{threads}",
        "command": ["z3", "sat.threads=0", f"sat.ddfw.threads={threads}", "-smt2"]
    })
    configs.append({
        "name": "PCC_def",
        "command": ["z3", "parallel.enable=true", f"parallel.threads.max={threads}", "-smt2"]
    })

    # 2) Pairwise hybrids: split threads evenly
    half = threads // 2
    configs.append({
        "name": f"CDCL{half}_LS{half}",
        "command": ["z3", f"sat.threads={half}", f"sat.local_search_threads={half}", "-smt2"]
    })
    configs.append({
        "name": f"CDCL{half}_DDFW{half}",
        "command": ["z3", f"sat.threads={half}", f"sat.ddfw.threads={half}", "-smt2"]
    })
    configs.append({
        "name": f"LS{half}_DDFW{half}",
        "command": ["z3", "sat.threads=0", f"sat.local_search_threads={half}", f"sat.ddfw.threads={half}", "-smt2"]
    })

    # 3) Tri-hybrid (majority CDCL)
    configs.append({
        "name": f"CDCL{threads-2}_LS1_DDFW1",
        "command": ["z3", f"sat.threads={threads-2}", "sat.local_search_threads=1", "sat.ddfw.threads=1", "-smt2"]
    })

    # 4) PCC grid (batch_size x delay x restart)
    batch_sizes = [50, 100, 200, 500]
    delays = [0, 5, 10, 20]
    restarts = [1, 2, 5]
    for bs in batch_sizes:
        for d in delays:
            for r in restarts:
                name = f"PCC_bs{bs}_d{d}_r{r}"
                cmd = [
                    "z3",
                    "parallel.enable=true",
                    f"parallel.threads.max={threads}",
                    f"parallel.conquer.batch_size={bs}",
                    f"parallel.conquer.delay={d}",
                    f"parallel.conquer.restart.max={r}",
                    "-smt2"
                ]
                configs.append({"name": name, "command": cmd})

    # 5) PCC + SAT/SLS/DDFW hybrids: split threads.max + sat threads
    for split in [2, 4, 6]:
        if split >= threads:
            continue
        rem = threads - split
        configs.append({
            "name": f"PCC{split}_CDCL{rem}",
            "command": ["z3", "parallel.enable=true", f"parallel.threads.max={split}", f"sat.threads={rem}", "-smt2"]
        })
        configs.append({
            "name": f"PCC{split}_LS{rem}",
            "command": ["z3", "parallel.enable=true", f"parallel.threads.max={split}", "sat.threads=0", f"sat.local_search_threads={rem}", "-smt2"]
        })
        configs.append({
            "name": f"PCC{split}_DDFW{rem}",
            "command": ["z3", "parallel.enable=true", f"parallel.threads.max={split}", "sat.threads=0", f"sat.ddfw.threads={rem}", "-smt2"]
        })

    return configs



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
	},
	"parallel-hyperparameter-search": {
		"version": 1,
		"glob": "COWABUNGA",
		"time_limit": "00:10:00",
		"memory_limit": "3000MB",
		"threads": 8,
		"parallel": 8,
		"commands": get_z3_parallel_configs(threads=8),
		"task_list": "SMT-COMP_2024_tasks_all.txt",
	},
	"parallel-hyperparameter-search-vlsat": {
		"version": 1,
		"glob": "VLSAT3/**/*.smt2",
		"time_limit": "00:10:00",
		"memory_limit": "3000MB",
		"threads": 8,
		"parallel": 8,
		"commands": [
			{
				"name": "default_parallel",
				"command": ["z3", "parallel.enable=true", "parallel.threads.max=8", "-smt2"]
			}
		],
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
		final_cmd = ["srun", "--exact", "--nodes=1", "--ntasks=1", f"--cpus-per-task={PARALLEL}", f"--mem-per-cpu={MEMORY_LIMIT}", f"--time={TIME_LIMIT}", "--cpu-bind=cores"]
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
	PARALLEL = benches[args.name]["parallel"] if "parallel" in benches[args.name] else args.parallel
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
