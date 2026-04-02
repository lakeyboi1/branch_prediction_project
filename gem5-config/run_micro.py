from __future__ import print_function

import argparse
import time
import m5
from m5.objects import *
from system import BaseTestSystem

# ---------------------------------------------------------------------------
# Functional unit descriptions
# ---------------------------------------------------------------------------
class IntALU(FUDesc):
    opList = [OpDesc(opClass='IntAlu', opLat=1)]
    count = 16

class IntMultDiv(FUDesc):
    opList = [OpDesc(opClass='IntMult', opLat=1),
              OpDesc(opClass='IntDiv',  opLat=1, pipelined=False)]
    count = 16

class FP_ALU(FUDesc):
    opList = [OpDesc(opClass='FloatAdd', opLat=1),
              OpDesc(opClass='FloatCmp', opLat=1),
              OpDesc(opClass='FloatCvt', opLat=1)]
    count = 16

class FP_MultDiv(FUDesc):
    opList = [OpDesc(opClass='FloatMult',    opLat=1),
              OpDesc(opClass='FloatMultAcc', opLat=1),
              OpDesc(opClass='FloatMisc',    opLat=1),
              OpDesc(opClass='FloatDiv',     opLat=1, pipelined=False),
              OpDesc(opClass='FloatSqrt',    opLat=1, pipelined=False)]
    count = 16

class SIMD_Unit(FUDesc):
    opList = [OpDesc(opClass='SimdAdd',         opLat=1),
              OpDesc(opClass='SimdAddAcc',       opLat=1),
              OpDesc(opClass='SimdAlu',          opLat=1),
              OpDesc(opClass='SimdCmp',          opLat=1),
              OpDesc(opClass='SimdCvt',          opLat=1),
              OpDesc(opClass='SimdMisc',         opLat=1),
              OpDesc(opClass='SimdMult',         opLat=1),
              OpDesc(opClass='SimdMultAcc',      opLat=1),
              OpDesc(opClass='SimdShift',        opLat=1),
              OpDesc(opClass='SimdShiftAcc',     opLat=1),
              OpDesc(opClass='SimdSqrt',         opLat=1),
              OpDesc(opClass='SimdFloatAdd',     opLat=1),
              OpDesc(opClass='SimdFloatAlu',     opLat=1),
              OpDesc(opClass='SimdFloatCmp',     opLat=1),
              OpDesc(opClass='SimdFloatCvt',     opLat=1),
              OpDesc(opClass='SimdFloatDiv',     opLat=1),
              OpDesc(opClass='SimdFloatMisc',    opLat=1),
              OpDesc(opClass='SimdFloatMult',    opLat=1),
              OpDesc(opClass='SimdFloatMultAcc', opLat=1),
              OpDesc(opClass='SimdFloatSqrt',    opLat=1)]
    count = 16

class MemPort(FUDesc):
    opList = [OpDesc(opClass='MemRead'),      OpDesc(opClass='MemWrite'),
              OpDesc(opClass='FloatMemRead'), OpDesc(opClass='FloatMemWrite')]
    count = 16

class Ideal_FUPool(FUPool):
    FUList = [IntALU(), IntMultDiv(), FP_ALU(), FP_MultDiv(),
              SIMD_Unit(), MemPort()]

# ---------------------------------------------------------------------------
# Wide O3 CPU — 16-wide pipeline with ideal FU pool so branch prediction
# is the dominant performance factor
# ---------------------------------------------------------------------------
class WideO3CPU(DerivO3CPU):
    fetchWidth       = 16
    decodeWidth      = 16
    renameWidth      = 16
    dispatchWidth    = 16
    issueWidth       = 16
    wbWidth          = 16
    commitWidth      = 16
    squashWidth      = 16
    fetchQueueSize   = 128
    LQEntries        = 128
    SQEntries        = 128
    numPhysIntRegs   = 256
    numPhysFloatRegs = 256
    numROBEntries    = 256
    instQueues       = [IQUnit(numEntries=256, fuPool=Ideal_FUPool())]

# ---------------------------------------------------------------------------
# Arguments
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("binary",
                    help="Path to the benchmark binary (static ELF)")
parser.add_argument("--bp_type",
                    default="multi_branch",
                    choices=["none", "local", "bimode",
                             "tage_base", "multi_branch"],
                    help="Branch predictor type (default: multi_branch)")
parser.add_argument("--max_insts", type=int, default=1_000_000,
                    help="Max instructions to simulate (default: 1M)")
parser.add_argument("--lvp",
                    action="store_true",
                    default=False,
                    help="Enable Markov Load Value Predictor")
parser.add_argument("--lvp_context_depth", type=int, default=1,
                    help="LVP context depth (1=last-value, 2=2nd-order)")
parser.add_argument("--lvp_table_size", type=int, default=4096,
                    help="LVP Markov table size (power of 2)")
args = parser.parse_args()

# ---------------------------------------------------------------------------
# Build and run
# ---------------------------------------------------------------------------
class MySystem(BaseTestSystem):
    _CPUModel            = WideO3CPU
    _BranchPredictorType = args.bp_type

system = MySystem()
if args.lvp:
    system.lvp.context_depth = args.lvp_context_depth
    system.lvp.table_size    = args.lvp_table_size
system.setTestBinary(args.binary)
system.cpu.max_insts_any_thread = args.max_insts

root = Root(full_system=False, system=system)
m5.instantiate()

start_tick  = m5.curTick()
start_insts = system.totalInsts()
globalStart = time.time()

exit_event = m5.simulate()
print("Exit Event: " + exit_event.getCause())
m5.stats.dump()

end_tick  = m5.curTick()
end_insts = system.totalInsts()

print("Performance statistics:")
print("  Simulated time : %.6f s"  % ((end_tick  - start_tick)  / 1e12))
print("  Instructions   : %d"      %  (end_insts - start_insts))
print("  Wallclock time : %.2f s"  %  (time.time() - globalStart))

if args.lvp:
    try:
        predicted  = int(system.lvp.predicted.value())
        correct    = int(system.lvp.correct.value())
        incorrect  = int(system.lvp.incorrect.value())
        total      = int(system.lvp.totalLoads.value())
        accuracy   = (correct / predicted * 100) if predicted > 0 else 0.0
        coverage   = (predicted / total    * 100) if total    > 0 else 0.0
        print("\nLVP statistics:")
        print(f"  Total loads    : {total}")
        print(f"  Predicted      : {predicted}  ({coverage:.1f}% coverage)")
        print(f"  Correct        : {correct}   ({accuracy:.1f}% accuracy)")
        print(f"  Incorrect      : {incorrect}")
    except Exception as e:
        print(f"  (LVP stats unavailable: {e})")