#!/bin/sh
set -eu

# Provide compatibility symlink for VS Code’s default mount path
mkdir -p /workspaces
if [ ! -e /workspaces/Benchmarking ]; then
  ln -s /workspace /workspaces/Benchmarking
fi

# Detect cgroups v1: no cgroup.controllers file
if [ ! -f /sys/fs/cgroup/cgroup.controllers ]; then
  echo "Detected cgroups v1: mounting and granting permissions"
  # Ensure cgroup v1 is mounted
  mount -t cgroup cgroup /sys/fs/cgroup || true
  # Grant all users ability to create sub-cgroups
  chmod o+wt,g+w /sys/fs/cgroup
  # Launch as benchuser if necessary
  if [ "$(id -u)" = "0" ]; then
    exec su benchuser -c "$(printf '%q ' "$@")"
  else
    exec "$@"
  fi
fi

# If cgroup v2 controllers not available, mount a fresh cgroup2 tree
if [ ! -f /sys/fs/cgroup/cgroup.controllers ]; then
  echo "Mounting cgroup2 at /sys/fs/cgroup"
  mount -t cgroup2 none /sys/fs/cgroup
fi

# Create new sub-cgroups
# Note: While "init" can be renamed, the name "benchexec" is important
mkdir -p /sys/fs/cgroup/init /sys/fs/cgroup/benchexec
# Move the init process to that cgroup
echo $$ > /sys/fs/cgroup/init/cgroup.procs

# Enable controllers in subtrees for benchexec to use
for controller in $(cat /sys/fs/cgroup/cgroup.controllers); do
  echo "+$controller" > /sys/fs/cgroup/cgroup.subtree_control
  echo "+$controller" > /sys/fs/cgroup/benchexec/cgroup.subtree_control
done

# Give benchuser ownership of the init and benchexec cgroups
chown -R benchuser:benchuser /sys/fs/cgroup/init /sys/fs/cgroup/benchexec

# Chown and set permissions for controller-specific host cgroup directories
# so that benchuser can access and create subgroups
while IFS=: read -r subsystems cgrouppath _; do
  for controller in $(echo "$subsystems" | tr ',' ' '); do
    dir="/sys/fs/cgroup/${controller}${cgrouppath}"
    if [ -d "$dir" ]; then
      chown benchuser:benchuser "$dir"
      chmod o+wt "$dir"
    fi
  done
done < /proc/self/cgroup

# Grant write permission on controller-specific cgroups for this container
CGROUP_PATH=$(awk -F: '$3 ~ /^\/docker\// {print $3; exit}' /proc/self/cgroup)
for ctrl in cpuacct cpuset freezer memory; do
  dir="/sys/fs/cgroup/$ctrl$CGROUP_PATH"
  if [ -d "$dir" ]; then
    chmod o+wt "$dir"
  fi
done

# Execute the given command as benchuser
if [ "$(id -u)" = "0" ]; then
  exec su benchuser -c "$(printf '%q ' "$@")"
else
  exec "$@"
fi