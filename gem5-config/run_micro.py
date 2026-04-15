from __future__ import print_function

import argparse
import time
import m5
from m5.objects import *
from m5.objects import MarkovLVP
from system import BaseTestSystem

# ---------------------------------------------------------------------------
# Functional unit pool
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
# CPU without value predictor (uses BaseO3CPU default MarkovLVP)
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
# CPU with Markov value predictor explicitly enabled
# ---------------------------------------------------------------------------
# WideO3CPUWithVP built dynamically below after args parsed

# ---------------------------------------------------------------------------
# Arguments
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("binary",
                    help="Path to the benchmark binary (static ELF)")
parser.add_argument("--bp_type",
                    default="local",
                    choices=["none", "local", "tage_base"],
                    help="Branch predictor type (default: local)")
parser.add_argument("--max_insts", type=int, default=1_000_000,
                    help="Max instructions to simulate (default: 1M)")
parser.add_argument("--value-pred", action="store_true", default=False,
                    help="Enable Markov load value predictor")
parser.add_argument("--context_depth", type=int, default=1,
                    help="Markov context depth (default: 1)")
parser.add_argument("--conf_threshold", type=int, default=1,
                    help="Confidence threshold to predict (default: 1)")
parser.add_argument("--max_conf", type=int, default=15,
                    help="Saturating counter ceiling (default: 15 = 4-bit)")
args = parser.parse_args()

# ---------------------------------------------------------------------------
# Build and run
# ---------------------------------------------------------------------------
if args.value_pred:
    class WideO3CPUWithVP(WideO3CPU):
        value_pred = MarkovLVP(
            table_size      = 4096,
            context_depth   = args.context_depth,
            conf_threshold  = args.conf_threshold,
            max_conf        = args.max_conf,
        )
    _cpu_model = WideO3CPUWithVP
else:
    _cpu_model = WideO3CPU

class MySystem(BaseTestSystem):
    _CPUModel            = _cpu_model
    _BranchPredictorType = args.bp_type

system = MySystem()
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
