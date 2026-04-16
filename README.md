# CMPT 450 Project: Implementing and Evaluating a Markov Load Value Predictor in gem5

## Prerequisites
Before running the project on your machine, please ensure you have the following prerequisites installed:
* Python 3
* `scons` (build tool)
* A C++ compiler (e.g., GCC/G++) 
* Python plotting packages (e.g., `matplotlib`, `pandas`, depending on what `plots.py` requires) Ensure requirements inside `gem5-fresh/requirements.txt` are satisfied.

## Detailed Instructions to Run the Project

Please execute the following commands in the terminal sequentially. The experiment involves compiling the `gem5-fresh` codebase, running parallel simulation jobs, and evaluating the results using plots.

### Step 1: Build the gem5 binary
First, navigate into the custom gem5 folder. You must build the simulator binary for the X86 architecture.

```bash
cd ./gem5-fresh
scons build/X86/gem5.opt -j8
```
*(Tip: Replace `-j8` with the number of available CPU cores on your machine for faster compilation).*

### Step 2: Run the simulation jobs
Go back to the root directory of the project and run the simulation jobs script. This script will launch the simulations and generate all the necessary statistics. 

```bash
cd ..
python3 jobs.py 8
```
*(The integer `8` represents the number of parallel simulation processes).*

### Step 3: Generate the evaluation plots
Once the simulations complete and populate the `results/` folder, run the plotting script. This step will parse the generated statistics and plot the final charts directly into the `plots/` directory.

```bash
python3 plots.py \
    --depth-results results/X86/lvp_depth_sweep \
    --conf-results  results/X86/lvp_conf_sweep \
    --out-dir plots/lvp
```

### Output Expectations
1. **Simulation Stats**: Raw statistics outputs will be correctly written to `results/X86/lvp_depth_sweep` and `results/X86/lvp_conf_sweep`.
2. **Analysis Plots**: Visual representations analyzing the design will be directly exported to `plots/lvp`.