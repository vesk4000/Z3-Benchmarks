import json
import ijson
from collections import defaultdict
import matplotlib.pyplot as plt

def main():
	# Hard-coded input file for one-time use
	INPUT_FILE = "datasets/SMT-COMP_2024/results-sq-2024.json"

	seen = set()
	test_groups = defaultdict(lambda: {"times": [], "results": []})
	with open(INPUT_FILE, 'rb') as f:
		# Stream parse JSON objects under the 'results' array
		for entry in ijson.items(f, 'results.item'):
			meta = entry.get("file", {})
			if meta.get("logic") != "QF_BV":
				continue
			# only include the SingleQuery track
			if entry.get("track") != "SingleQuery":
				continue
			fam_list = meta.get("family", [])
			if not fam_list:
				continue
			name = meta.get("name")
			if not name:
				continue
			key = f"{'/'.join(fam_list)}/{name}"
			seen.add(key)
			# convert Decimal times to float

			# 
			if entry.get("solver") == "Z3-alpha" and entry.get("result") != "Timeout" and entry.get("wallclock_time", 5000.0) < 6000.0:
				test_groups[key]["results"].append(entry.get("result", ""))
				test_groups[key]["times"].append(float(entry.get("wallclock_time", 0.0)))
				test_groups[key]["memory"] = float(entry.get("memory_usage", 0.0))

	# unique test count
	print(f"Unique QF_BV tests: {len(seen)}")
	# Check for duplicate test names (final part after families)
	name_counts = defaultdict(list)
	for key in seen:
		# Extract the final part after the last '/'
		test_name = key.split('/')[-1]
		family_path = '/'.join(key.split('/')[:-1])
		name_counts[test_name].append(family_path)

	duplicates = {name: families for name, families in name_counts.items() if len(families) > 1}
	if duplicates:
		print(f"Found {len(duplicates)} duplicate test names:")
		for name, families in sorted(duplicates.items()):
			print(f"  '{name}': {len(families)} occurrences")
			for family in families:
				print(f"    - {family}")
	else:
		print("No duplicate test names found")
	# per-test statistics
	sum_min = 0.0
	sum_avg = 0.0
	unsolved_count = 0  # count of tests where all results are Timeout
	actual_tasks_to_run = []
	print("Test statistics (min_time, avg_time):")
	for key in sorted(test_groups):
		times = test_groups[key]["times"]
		results = test_groups[key]["results"]
		if not times:
			continue
		min_time = min(times)
		avg_time = sum(times) / len(times)
		
		sum_min += min_time
		sum_avg += avg_time

		actual_tasks_to_run.append(key)

		if all(r == "Timeout" for r in results):
			unsolved_count += 1

	# Calculate average memory usage and create histogram
	memory_values = []
	for key in test_groups:
		if "memory" in test_groups[key] and test_groups[key]["memory"] > 0:
			# Convert from MB to GB
			memory_gb = test_groups[key]["memory"] / (1024.0 ** 3)
			# if memory_gb > 1.0:
			memory_values.append(memory_gb)
	
	print(sum(memory_values) / len(memory_values))
	print(max(memory_values))


	# # Create histogram of memory usage
	# if memory_values:
	# 	plt.figure(figsize=(10, 6))
	# 	plt.hist(memory_values, bins=30, alpha=0.7, edgecolor='black')
	# 	plt.xlabel('Memory Usage (GB)')
	# 	plt.ylabel('Frequency')
	# 	plt.title('Distribution of Memory Usage for QF_BV Tests')
	# 	plt.grid(True, alpha=0.3)
	# 	plt.show()

	#print(f"Average memory usage: {avg_memory:.6f} MB")
	print(f"Sum of minimum times: {sum_min:.6f}")
	print(f"Sum of average times: {sum_avg:.6f}")
	print(f"Estimated time: {sum_avg / 3600 / 64 * 4 / 2:.2f} hours")
	print(f"Number of tests unsolved by any solver (all timed out): {unsolved_count}")
	total_tests = len(test_groups)
	print(f"Total QF_BV tests: {total_tests}")
	print(f"Number of tests solved by at least one solver: {total_tests - unsolved_count}")
	
	# Write all task paths to a file
	with open("SMT-COMP_2024_tasks_all.txt", "w") as f:
		for key in sorted(actual_tasks_to_run):
			f.write(f"SMT-COMP_2024/{key}\n")
	print(f"Written {len(actual_tasks_to_run)} task paths to SMT-COMP_2024_tasks_all.txt")

if __name__ == "__main__":
	main()
