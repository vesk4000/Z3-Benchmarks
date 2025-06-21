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

def make_best_parallel(num_parallel: int):
	return [
		{
			"name": "z3-parallel",
			"command": [
				"z3",
				f"sat.max_memory=2900",
				"parallel.enable=true",
				f"parallel.threads.max={num_parallel}",
				f"parallel.conquer.batch_size=50",
				f"parallel.conquer.delay=0",
				f"parallel.conquer.restart.max=10",
				"parallel.conquer.backtrack_frequency=1",
			],
		}
	]

from typing import List, Dict

def get_z3_parallel_configs(threads: int = 8, num_random_configs: int = 30) -> List[Dict]:
	"""
	Generate a list of Z3 parallel configurations with random search.
	
	Parameters:
	- threads: int - Number of threads for each configuration
	- num_random_configs: int - Number of random configurations to generate
	
	Returns a list of dicts: {'name': str, 'command': List[str]}
	"""
	configs = []
	
	# Memory limits (using a bit less than 3000MB to be safe)
	mem_limit = 2900
	mem_params = [f"sat.max_memory={mem_limit}"]
	sls_mem_param = f"sls.max_memory={mem_limit}"  # Corrected parameter name
	
	# 1) Pure modes (always included)
	configs.append({
		"name": f"CDCL_{threads}",
		"command": ["z3", f"sat.threads={threads}", mem_params[0], "-smt2"]
	})
	configs.append({
		"name": f"LS_{threads}",
		"command": ["z3", "sat.threads=0", f"sat.local_search_threads={threads}", mem_params[0], "-smt2"]
	})
	configs.append({
		"name": f"DDFW_{threads}",
		"command": ["z3", "sat.threads=0", f"sat.ddfw.threads={threads}", mem_params[0], "-smt2"]
	})
	configs.append({
		"name": f"SLS_{threads}",
		"command": ["z3", "smt.sls.enable=true", mem_params[0], sls_mem_param, "-smt2"]
	})
	configs.append({
		"name": "PCC_def",
		"command": ["z3", "parallel.enable=true", f"parallel.threads.max={threads}", mem_params[0], "-smt2"]
	})
	
	# Parameter ranges for random search
	param_ranges = {
		"batch_sizes": [50, 100, 200, 500, 1000, 2000],
		"delays": [0, 5, 10, 20, 50, 100, 500],
		"restarts": [1, 2, 5, 10, 25, 50],
	}
	
	# Generate random configurations
	for _ in range(num_random_configs):
		# Randomly choose configuration type
		config_type = random.choice([
			"hybrid_pair",        # Two techniques combined
			"hybrid_triple",      # Three techniques combined
			"pcc_with_params",    # PCC with random params
			"pcc_hybrid"          # PCC combined with another technique
		])
		
		if config_type == "hybrid_pair":
			# Randomly select two techniques and distribute threads between them
			techniques = random.sample(["CDCL", "LS", "DDFW", "SLS"], 2)
			split = random.randint(1, threads-1)
			remaining = threads - split
			
			cmd = ["z3"]
			name_parts = []
			has_sls = False
			
			for i, technique in enumerate(techniques):
				t_threads = split if i == 0 else remaining
				
				if technique == "CDCL":
					cmd.append(f"sat.threads={t_threads}")
					name_parts.append(f"CDCL{t_threads}")
				elif technique == "LS":
					if "sat.threads=0" not in cmd and "sat.threads=" not in "".join(cmd):
						cmd.append("sat.threads=0")
					cmd.append(f"sat.local_search_threads={t_threads}")
					name_parts.append(f"LS{t_threads}")
				elif technique == "DDFW":
					if "sat.threads=0" not in cmd and "sat.threads=" not in "".join(cmd):
						cmd.append("sat.threads=0")
					cmd.append(f"sat.ddfw.threads={t_threads}")
					name_parts.append(f"DDFW{t_threads}")
				elif technique == "SLS":
					cmd.append("smt.sls.enable=true")
					name_parts.append(f"SLS{t_threads}")
					has_sls = True
			
			# Add memory limits
			cmd.append(mem_params[0])
			if has_sls:
				cmd.append(sls_mem_param)
			cmd.append("-smt2")
			configs.append({
				"name": "_".join(name_parts),
				"command": cmd
			})
			
		elif config_type == "hybrid_triple":
			# Randomly distribute threads among three techniques
			techniques = random.sample(["CDCL", "LS", "DDFW", "SLS"], 3)
			
			# Distribute threads among techniques
			t1_threads = random.randint(1, threads-2)
			t2_threads = random.randint(1, threads-t1_threads-1)
			t3_threads = threads - t1_threads - t2_threads
			
			thread_counts = [t1_threads, t2_threads, t3_threads]
			random.shuffle(thread_counts)  # Randomly assign thread counts to techniques
			
			cmd = ["z3"]
			name_parts = []
			has_sls = False
			
			for i, technique in enumerate(techniques):
				t_threads = thread_counts[i]
				
				if technique == "CDCL":
					cmd.append(f"sat.threads={t_threads}")
					name_parts.append(f"CDCL{t_threads}")
				elif technique == "LS":
					if "sat.threads=0" not in cmd and "sat.threads=" not in "".join(cmd):
						cmd.append("sat.threads=0")
					cmd.append(f"sat.local_search_threads={t_threads}")
					name_parts.append(f"LS{t_threads}")
				elif technique == "DDFW":
					if "sat.threads=0" not in cmd and "sat.threads=" not in "".join(cmd):
						cmd.append("sat.threads=0")
					cmd.append(f"sat.ddfw.threads={t_threads}")
					name_parts.append(f"DDFW{t_threads}")
				elif technique == "SLS":
					cmd.append("smt.sls.enable=true")
					name_parts.append(f"SLS{t_threads}")
					has_sls = True
			
			# Add memory limits
			cmd.append(mem_params[0])
			if has_sls:
				cmd.append(sls_mem_param)
			cmd.append("-smt2")
			configs.append({
				"name": "_".join(name_parts),
				"command": cmd
			})
			
		elif config_type == "pcc_with_params":
			# PCC with random parameter settings
			bs = random.choice(param_ranges["batch_sizes"])
			d = random.choice(param_ranges["delays"])
			r = random.choice(param_ranges["restarts"])
			
			cmd = [
				"z3",
				"parallel.enable=true",
				f"parallel.threads.max={threads}",
				f"parallel.conquer.batch_size={bs}",
				f"parallel.conquer.delay={d}",
				f"parallel.conquer.restart.max={r}",
				mem_params[0],
				"-smt2"
			]
			
			configs.append({
				"name": f"PCC_bs{bs}_d{d}_r{r}",
				"command": cmd
			})
			
		elif config_type == "pcc_hybrid":
			# PCC combined with another technique
			pcc_threads = random.randint(1, threads-1)
			other_threads = threads - pcc_threads
			
			technique = random.choice(["CDCL", "LS", "DDFW", "SLS"])
			
			bs = random.choice(param_ranges["batch_sizes"])
			d = random.choice(param_ranges["delays"])
			r = random.choice(param_ranges["restarts"])
			
			cmd = [
				"z3",
				"parallel.enable=true",
				f"parallel.threads.max={pcc_threads}",
				f"parallel.conquer.batch_size={bs}",
				f"parallel.conquer.delay={d}",
				f"parallel.conquer.restart.max={r}",
			]
			
			name = f"PCC{pcc_threads}_bs{bs}_d{d}_r{r}"
			has_sls = False
			
			if technique == "CDCL":
				cmd.append(f"sat.threads={other_threads}")
				name += f"_CDCL{other_threads}"
			elif technique == "LS":
				cmd.append("sat.threads=0")
				cmd.append(f"sat.local_search_threads={other_threads}")
				name += f"_LS{other_threads}"
			elif technique == "DDFW":
				cmd.append("sat.threads=0")
				cmd.append(f"sat.ddfw.threads={other_threads}")
				name += f"_DDFW{other_threads}"
			elif technique == "SLS":
				cmd.append("smt.sls.enable=true")
				name += f"_SLS{other_threads}"
				has_sls = True
			
			# Add memory limits
			cmd.append(mem_params[0])
			if has_sls:
				cmd.append(sls_mem_param)
			cmd.append("-smt2")
			configs.append({
				"name": name,
				"command": cmd
			})
	
	return configs


def get_z3_parallel_configs3(threads: int = 8, num_random_configs: int = 30) -> List[Dict]:
	"""
	Generate a list of Z3 parallel configurations with random search.
	
	Parameters:
	- threads: int - Number of threads for each configuration
	- num_random_configs: int - Number of random configurations to generate
	
	Returns a list of dicts: {'name': str, 'command': List[str]}
	"""
	random.seed(42)
	
	configs = []
	
	# Memory limits (using a bit less than 3000MB to be safe)
	mem_limit = 2900
	
	# Parameter ranges for random search
	param_ranges = {
		"batch_sizes": [50, 100, 200, 500, 1000, 2000],
		"delays": [0, 5, 10, 20, 50, 100, 500],
		"restarts": [1, 2, 5, 10, 25, 50],
		"backtrack_frequency": [1, 5, 10, 50, 100, 1000],
	}
	
	# Generate random configurations
	for _ in range(num_random_configs):
		# Randomly choose configuration type
		config_type = random.choice([
			"just_one",
			"hybrid_pair",        # Two techniques combined
			"hybrid_triple"
		])
		techniques = random.choices(["SAT", "CaC"], [1, 20])
		all_techniques = ["DDFW", "SLS"]

		if config_type == "hybrid_pair":
			techniques.append(random.choice(all_techniques))
		if config_type == "hybrid_triple":
			techniques.extend(all_techniques)

		max_threads = threads
		threads_per_technique = []
		while True:
			for i in range(0, len(techniques) - 1):
				threads_per_technique.append(random.randint(1, max_threads - 1))
			sum_of_threads_so_far = sum(threads_per_technique)
			if sum_of_threads_so_far < max_threads:
				flag_all_ok = True
				threads_per_technique.append(max_threads - sum_of_threads_so_far)
				for i in range(len(threads_per_technique)):
					if techniques[i] == "SAT" or techniques[i] == "CaC":
						if threads_per_technique[i] < 2:
							flag_all_ok = False
				if flag_all_ok:
					break
			threads_per_technique = []

		cmd = ["z3"]
		name_parts = []

		cmd.append(f"sat.max_memory={mem_limit}")

		if "SAT" in techniques:
			idx = techniques.index("SAT")
			cmd.append(f"sat.threads={threads_per_technique[idx]}")
			name_parts.append(f"SAT{threads_per_technique[idx]}")
		if "SLS" in techniques:
			idx = techniques.index("SLS")
			cmd.append("smt.sls.enable=true")
			cmd.append(f"sat.local_search_threads={threads_per_technique[idx]}")
			cmd.append(f"smt.sls.parallel=true")
			cmd.append(f"sls.max_memory={mem_limit}")
			name_parts.append(f"SLS{threads_per_technique[idx]}")
		if "DDFW" in techniques:
			idx = techniques.index("DDFW")
			cmd.append(f"sat.ddfw.threads={threads_per_technique[idx]}")
			name_parts.append(f"DDFW{threads_per_technique[idx]}")
		if "CaC" in techniques:
			idx = techniques.index("CaC")
			bs = random.choice(param_ranges["batch_sizes"])
			d = random.choice(param_ranges["delays"])
			r = random.choice(param_ranges["restarts"])
			bf = random.choice(param_ranges["backtrack_frequency"])
			cmd.append(f"parallel.enable=true")
			name_parts.append(f"CaC{threads_per_technique[idx]}")
			cmd.append(f"parallel.threads.max={threads_per_technique[idx]}")
			cmd.append(f"parallel.conquer.batch_size={bs}")
			name_parts.append(f"bs{bs}")
			cmd.append(f"parallel.conquer.delay={d}")
			name_parts.append(f"d{d}")
			cmd.append(f"parallel.conquer.restart.max={r}")
			name_parts.append(f"r{r}")
			cmd.append(f"parallel.conquer.backtrack_frequency={bf}")
			name_parts.append(f"bf{bf}")
			
		configs.append({
			"name": "_".join(name_parts),
			"command": cmd
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
	"vlsat3_a_rest": {
		"version": 1,
		"glob": "COWABUNGA",
		"time_limit": "00:40:00",
		"memory_limit": "3000MB",
		"threads": 63,
		"commands": four_horsemen,
		"task_list": "VLSAT3a_tasks_rest.txt",
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
		"time_limit": "00:20:00",
		"memory_limit": "3000MB",
		"threads": 64,
		"commands": four_horsemen,
		"task_list": "SMT-COMP_2024_tasks_all.txt",
	},
	"smt-comp_2024-electric-boogaloo": {
		"version": 1,
		"glob": "COWABUNGA",
		"time_limit": "00:20:00",
		"memory_limit": "3000MB",
		"threads": 64,
		"commands": four_horsemen,
		"task_list": "SMT-COMP_2024_rest.txt",
	},
	"smart_contracts": {
		"version": 1,
		"glob": "Smart_Contract_Verification/**/*.smt2",
		"time_limit": "01:00:00",
		"memory_limit": "12000MB",
		"threads": 16,
		"commands": four_horsemen,
	},
	# "parallel-scaling-1": {
	# 	"version": 1,
	# 	"glob": "COWABUNGA",
	# 	"time_limit": "00:20:00",
	# 	"memory_limit": "3000MB",
	# 	"threads": 64,
	# 	"parallel": 1,
	# 	"commands": make_best_parallel(1),
	# 	"task_list": "SMT-COMP_2024_tasks_all.txt", # z3 alpha doesn't timeout on these
	# },
	"parallel-scaling-2": {
		"version": 1,
		"glob": "COWABUNGA",
		"time_limit": "00:20:00",
		"memory_limit": "3000MB",
		"threads": 32,
		"parallel": 2,
		"commands": make_best_parallel(2),
		"task_list": "SMT-COMP_2024_tasks_all.txt", # z3 alpha doesn't timeout on these
	},
	"parallel-scaling-4": {
		"version": 1,
		"glob": "COWABUNGA",
		"time_limit": "00:20:00",
		"memory_limit": "3000MB",
		"threads": 16,
		"parallel": 4,
		"commands": make_best_parallel(4),
		"task_list": "SMT-COMP_2024_tasks_all.txt", # z3 alpha doesn't timeout on these
	},
	"parallel-scaling-8": {
		"version": 1,
		"glob": "COWABUNGA",
		"time_limit": "00:20:00",
		"memory_limit": "3000MB",
		"threads": 8,
		"parallel": 8,
		"commands": make_best_parallel(8),
		"task_list": "SMT-COMP_2024_tasks_all.txt", # z3 alpha doesn't timeout on these
	},
	"parallel-scaling-16": {
		"version": 1,
		"glob": "COWABUNGA",
		"time_limit": "00:20:00",
		"memory_limit": "3000MB",
		"threads": 4,
		"parallel": 16,
		"commands": make_best_parallel(16),
		"task_list": "SMT-COMP_2024_tasks_all.txt", # z3 alpha doesn't timeout on these
	},
	"parallel-scaling-32": {
		"version": 1,
		"glob": "COWABUNGA",
		"time_limit": "00:20:00",
		"memory_limit": "3000MB",
		"threads": 2,
		"parallel": 32,
		"commands": make_best_parallel(32),
		"task_list": "SMT-COMP_2024_tasks_all.txt", # z3 alpha doesn't timeout on these
	},
	"parallel-scaling-64": {
		"version": 1,
		"glob": "COWABUNGA",
		"time_limit": "00:20:00",
		"memory_limit": "3000MB",
		"threads": 1,
		"parallel": 64,
		"commands": make_best_parallel(64),
		"task_list": "SMT-COMP_2024_tasks_all.txt", # z3 alpha doesn't timeout on these
	},
	"parallel-hyperparameter-search": {
		"version": 1,
		"glob": "COWABUNGA",
		"time_limit": "00:04:00",
		"memory_limit": "3000MB",
		"threads": 8,
		"parallel": 8,
		"commands": get_z3_parallel_configs3(threads=8, num_random_configs=500),
		"task_list": "SMT-COMP_2024_tasks_all.txt", # z3 alpha doesn't timeout on these
	},
	"parallel-hyperparameter-search-2": {
		"version": 1,
		"glob": "COWABUNGA",
		"time_limit": "00:00:30",
		"memory_limit": "3000MB",
		"threads": 1,
		"parallel": 8,
		"commands": get_z3_parallel_configs(threads=8, num_random_configs=500),
		"task_list": "SMT-COMP_2024_tasks_one.txt",
	},
	"parallel-hyperparameter-search-3": {
		"version": 1,
		"glob": "COWABUNGA",
		"time_limit": "00:03:00",
		"memory_limit": "3000MB",
		"threads": 1,
		"parallel": 8,
		"commands": get_z3_parallel_configs3(threads=8, num_random_configs=500),
		"task_list": "SMT-COMP_2024_tasks_one.txt",
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

	print(benches[args.name]["commands"])
	print(len(benches[args.name]["commands"]))

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
