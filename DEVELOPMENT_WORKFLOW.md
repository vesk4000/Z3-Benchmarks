# Development Workflow for BenchExec and Z3 in Docker

This document describes how to develop, tweak, and share a fully-configured Docker image for benchmarking Z3 with BenchExec, using the **docker commit** snapshot workflow.  No full Docker rebuilds required for each tweak.

---

## 1. Build & Publish the Base Image

1. In your workspace directory (contains Dockerfile):
   ```bash
   docker build -t you/z3-bench:base .
   ```
2. (Optional) Push to Docker Hub or your registry:
   ```bash
   docker push you/z3-bench:base
   ```

This image includes Ubuntu 24.04, BenchExec, multiple Z3 versions, Python deps, and all required tools.

## 2. Day-to-Day Development with `docker commit`

Whenever you need to install a new OS package, tweak configs, or add Python libs:

1. Start an interactive container:
   ```bash
   docker run -it --privileged \
     -v $(pwd):/workspace \
     --name z3-dev you/z3-bench:base bash
   ```
2. Inside the container, make your changes:
   ```bash
   sudo apt update && sudo apt install -y <package>
   pip3 install <python-package>
   # edit configs, scripts, etc.
   ```
3. Exit the shell.
4. Snapshot your changes into a new image:
   ```bash
   docker commit z3-dev you/z3-bench:dev
   ```
5. (Optional) Push the `:dev` tag:
   ```bash
   docker push you/z3-bench:dev
   ```

Now `you/z3-bench:dev` contains all your interactive tweaks without rebuilding from scratch.

## 3. Integrate with VS Code

1. In `.devcontainer/devcontainer.json`, reference your dev image:
   ```json
   {
     "image": "you/z3-bench:dev",
     "workspaceFolder": "/workspace",
     "runArgs": ["--privileged"],
     // ... other settings ...
   }
   ```
2. Reopen in Container: VS Code will pull and start your `:dev` image instantly.
3. Work, test, benchmark in `/workspace` with all your tools preinstalled.

## 4. Cleaning Up & Versioning

- Remove old containers and images you no longer need:
  ```bash
  docker rm z3-dev
  docker image prune -f
  ```
- Keep your container setup and benchmark scripts under Git in `/workspace` to version control your experiments.

## 5. Verifying Your Dev Container

1. In VS Code, open the Command Palette (`Ctrl+Shift+P`) and select **Dev Containers: Rebuild and Reopen in Container**.
2. Once the container starts, open an integrated terminal (View → Terminal).
3. Run the following to confirm tools are available and working:
   ```bash
   z3 --version
   benchexec --help
   benchexec sample-benchmark.xml
   ```
4. After `benchexec sample-benchmark.xml` completes, check that a results directory was created in `/workspace/results` with the run summary.
5. If all commands succeed, your container environment is correctly configured and ready for custom benchmarks.

---

With this pattern, you get an instantly startable, shareable image on Docker Hub, plus unlimited interactive tweaking without painful rebuilds.
