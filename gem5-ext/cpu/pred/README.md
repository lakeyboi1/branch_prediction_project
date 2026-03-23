# gem5-ext

Custom gem5 extensions for the branch predictor project.
These files extend gem5 without modifying the base installation at /data/gem5-baseline.

## Directory Structure

    gem5-ext/
    └── cpu/pred/
        ├── multi_branch.hh       — C++ header for MultiBranchPredictor
        ├── multi_branch.cc       — C++ implementation
        ├── BranchPredictor.py    — gem5 Python params registration
        └── SConscript            — build system registration

## Building

From /data/gem5-baseline, run:

    scons build/X86/gem5.opt --ext-dir=$LAB_PATH/gem5-ext -j8

$LAB_PATH is already exported in submit.sh as $PWD (your project root).

## Using in gem5-config/system.py

    from m5.objects import MultiBranchPredictor

    cpu.branchPred = MultiBranchPredictor(
        local_table_size=2048,
        global_table_size=8192,
        global_history_bits=13,
        choice_table_size=8192
    )
