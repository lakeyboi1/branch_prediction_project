Instructions to run the experiment and generate plots:

Step 1: Enter the gem5-fresh directory
Build the gem5 binary, in case it is not built yet

cd ./gem5-fresh
scons build/X86/gem5.opt -j8

Step 2: Go back to the project root directory
Run the jobs.py script which will run the simulation,
and generate all the necessary statistics in the results/ directory.

cd ..
python3 jobs.py 8

Step 3: Run the plotting script, this will generate all the graphs
in the plots/ directory.

python3 plots.py \
    --depth-results results/X86/lvp_depth_sweep \
    --conf-results  results/X86/lvp_conf_sweep \
    --out-dir plots/lvp