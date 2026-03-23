from __future__ import print_function

import argparse
import time
import m5
from m5.objects import TimingSimpleCPU, DerivO3CPU, MinorCPU
from m5.objects import LTAGE
from m5.objects import Root
from m5.objects import *
from system import BaseTestSystem
from system import InfMemory, SingleCycleMemory, SlowMemory

BranchPredictor = LTAGE

class IntALU(FUDesc):
    opList = [OpDesc(opClass='IntAlu', opLat=1)]
    count = 32

class IntMultDiv(FUDesc):
    opList = [OpDesc(opClass='IntMult', opLat=1),
              OpDesc(opClass='IntDiv', opLat=20, pipelined=False)]
    if buildEnv['TARGET_ISA'] in ('x86'):
        opList[1].opLat = 1
    count = 32

class FP_ALU(FUDesc):
    opList = [OpDesc(opClass='FloatAdd', opLat=1),
              OpDesc(opClass='FloatCmp', opLat=1),
              OpDesc(opClass='FloatCvt', opLat=1)]
    count = 32

class FP_MultDiv(FUDesc):
    opList = [OpDesc(opClass='FloatMult', opLat=1),
              OpDesc(opClass='FloatMultAcc', opLat=1),
              OpDesc(opClass='FloatMisc', opLat=1),
              OpDesc(opClass='FloatDiv', opLat=1, pipelined=False),
              OpDesc(opClass='FloatSqrt', opLat=1, pipelined=False)]
    count = 32

class SIMD_Unit(FUDesc):
    opList = [OpDesc(opClass='SimdAdd', opLat=1),
              OpDesc(opClass='SimdAddAcc', opLat=1),
              OpDesc(opClass='SimdAlu', opLat=1),
              OpDesc(opClass='SimdCmp', opLat=1),
              OpDesc(opClass='SimdCvt', opLat=1),
              OpDesc(opClass='SimdMisc', opLat=1),
              OpDesc(opClass='SimdMult', opLat=1),
              OpDesc(opClass='SimdMultAcc', opLat=1),
              OpDesc(opClass='SimdShift', opLat=1),
              OpDesc(opClass='SimdShiftAcc', opLat=1),
              OpDesc(opClass='SimdSqrt', opLat=1),
              OpDesc(opClass='SimdFloatAdd', opLat=1),
              OpDesc(opClass='SimdFloatAlu', opLat=1),
              OpDesc(opClass='SimdFloatCmp', opLat=1),
              OpDesc(opClass='SimdFloatCvt', opLat=1),
              OpDesc(opClass='SimdFloatDiv', opLat=1),
              OpDesc(opClass='SimdFloatMisc', opLat=1),
              OpDesc(opClass='SimdFloatMult', opLat=1),
              OpDesc(opClass='SimdFloatMultAcc', opLat=1),
              OpDesc(opClass='SimdFloatSqrt', opLat=1)]
    count = 32

class ReadPort(FUDesc):
    opList = [OpDesc(opClass='MemRead'),
              OpDesc(opClass='FloatMemRead')]
    count = 32

class WritePort(FUDesc):
    opList = [OpDesc(opClass='MemWrite'),
              OpDesc(opClass='FloatMemWrite')]
    count = 32

class RdWrPort(FUDesc):
    opList = [OpDesc(opClass='MemRead'), OpDesc(opClass='MemWrite'),
              OpDesc(opClass='FloatMemRead'), OpDesc(opClass='FloatMemWrite')]
    count = 32

class IprPort(FUDesc):
    opList = [OpDesc(opClass='IprAccess', opLat=1, pipelined=False)]
    count = 32

class Ideal_FUPool(FUPool):
    FUList = [IntALU(), IntMultDiv(), FP_ALU(), FP_MultDiv(), ReadPort(),
              SIMD_Unit(), WritePort(), RdWrPort(), IprPort()]

class MinorIntFU(MinorFU):
    opClasses = minorMakeOpClassSet(['IntAlu'])
    timings = [MinorFUTiming(description="Int", srcRegsRelativeLats=[2])]
    opLat = 3
    issueLat = 1

class MinorIntMulFU(MinorFU):
    opClasses = minorMakeOpClassSet(['IntMult'])
    timings = [MinorFUTiming(description='Mul', srcRegsRelativeLats=[0])]
    opLat = 3

class MinorIntDivFU(MinorFU):
    opClasses = minorMakeOpClassSet(['IntDiv'])
    issueLat = 9
    opLat = 9

class MinorFloatSimdFU(MinorFU):
    opClasses = minorMakeOpClassSet([
        'FloatAdd', 'FloatCmp', 'FloatCvt', 'FloatMisc', 'FloatMult',
        'FloatMultAcc', 'FloatDiv', 'FloatSqrt',
        'SimdAdd', 'SimdAddAcc', 'SimdAlu', 'SimdCmp', 'SimdCvt',
        'SimdMisc', 'SimdMult', 'SimdMultAcc', 'SimdShift', 'SimdShiftAcc',
        'SimdDiv', 'SimdSqrt', 'SimdFloatAdd', 'SimdFloatAlu', 'SimdFloatCmp',
        'SimdFloatCvt', 'SimdFloatDiv', 'SimdFloatMisc', 'SimdFloatMult',
        'SimdFloatMultAcc', 'SimdFloatSqrt', 'SimdReduceAdd', 'SimdReduceAlu',
        'SimdReduceCmp', 'SimdFloatReduceAdd', 'SimdFloatReduceCmp',
        'SimdAes', 'SimdAesMix',
        'SimdSha1Hash', 'SimdSha1Hash2', 'SimdSha256Hash',
        'SimdSha256Hash2', 'SimdShaSigma2', 'SimdShaSigma3'])
    timings = [MinorFUTiming(description='FloatSimd', srcRegsRelativeLats=[2])]
    opLat = 6

class MinorPredFU(MinorFU):
    opClasses = minorMakeOpClassSet(['SimdPredAlu'])
    timings = [MinorFUTiming(description="Pred", srcRegsRelativeLats=[2])]
    opLat = 3

class MinorMemFU(MinorFU):
    opClasses = minorMakeOpClassSet(['MemRead', 'MemWrite', 'FloatMemRead', 'FloatMemWrite'])
    timings = [MinorFUTiming(description='Mem', srcRegsRelativeLats=[1], extraAssumedLat=2)]
    opLat = 1

class MinorMiscFU(MinorFU):
    opClasses = minorMakeOpClassSet(['IprAccess', 'InstPrefetch'])
    opLat = 1

class Minor4_FUPool(MinorFUPool):
    funcUnits = [MinorIntFU(), MinorIntFU(),
                 MinorIntMulFU(), MinorIntDivFU(),
                 MinorFloatSimdFU(), MinorPredFU(),
                 MinorMemFU(), MinorMiscFU()]

class Minor4CPU(MinorCPU):
    branchPred = BranchPredictor()
    executeFuncUnits = Minor4_FUPool()
    decodeInputWidth = 4
    executeInputWidth = 4
    executeIssueLimit = 4
    executeCommitLimit = 4

class O3_W256CPU(DerivO3CPU):
    branchPred = BranchPredictor()
    fuPool = Ideal_FUPool()
    fetchWidth = 32
    decodeWidth = 32
    renameWidth = 32
    dispatchWidth = 32
    issueWidth = 32
    wbWidth = 32
    commitWidth = 32
    squashWidth = 32
    fetchQueueSize = 256
    LQEntries = 250
    SQEntries = 250
    numPhysIntRegs = 256
    numPhysFloatRegs = 256
    numIQEntries = 256
    numROBEntries = 256

class O3_W2KCPU(DerivO3CPU):
    branchPred = BranchPredictor()
    fuPool = Ideal_FUPool()
    fetchWidth = 32
    decodeWidth = 32
    renameWidth = 32
    dispatchWidth = 32
    issueWidth = 32
    wbWidth = 32
    commitWidth = 32
    squashWidth = 32
    fetchQueueSize = 256
    LQEntries = 250
    SQEntries = 250
    numPhysIntRegs = 1024
    numPhysFloatRegs = 1024
    numIQEntries = 2096
    numROBEntries = 2096

class SimpleCPU(TimingSimpleCPU):
    branchPred = BranchPredictor()

class DefaultO3CPU(DerivO3CPU):
    branchPred = BranchPredictor()

valid_cpus = [SimpleCPU, Minor4CPU, DefaultO3CPU, O3_W256CPU, O3_W2KCPU]
valid_cpus = {cls.__name__[:-3]: cls for cls in valid_cpus}

valid_memories = [InfMemory, SingleCycleMemory, SlowMemory]
valid_memories = {cls.__name__[:-6]: cls for cls in valid_memories}

dram_choices = ["DDR3_1600_8x8", "DDR3_2133_8x8", "LPDDR2_S4_1066_1x32", "HBM_1000_4H_1x64"]

parser = argparse.ArgumentParser()
parser.add_argument("cpu", choices=valid_cpus.keys())
parser.add_argument("memory_or_dram", type=str)
parser.add_argument("binary", type=str)

parser.add_argument("--clock", default=None)
parser.add_argument("--cpu-clock", default=None)
parser.add_argument("--sys-clock", default=None)

parser.add_argument("--caches", action="store_true", default=False)
parser.add_argument("--l2cache", action="store_true", default=False)

parser.add_argument("--cacheline_size", type=int, default=64)

parser.add_argument("--l1i_size", default="32kB")
parser.add_argument("--l1d_size", default="32kB")
parser.add_argument("--l2_size", default="1MB")

parser.add_argument("--l1i_assoc", type=int, default=8)
parser.add_argument("--l1d_assoc", type=int, default=8)
parser.add_argument("--l2_assoc", type=int, default=16)

parser.add_argument("--max_insts", type=int, default=1_000_000)

parser.add_argument("--dram_type", default=None, choices=dram_choices)

args = parser.parse_args()

memory_model = None
dram_type = None
if args.dram_type is not None:
    dram_type = args.dram_type
else:
    if args.memory_or_dram in valid_memories:
        memory_model = args.memory_or_dram
    elif args.memory_or_dram in dram_choices:
        dram_type = args.memory_or_dram
        memory_model = "Slow"
    else:
        raise SystemExit("memory_or_dram must be one of: {} or {}".format(
            ",".join(valid_memories.keys()), ",".join(dram_choices)
        ))

if memory_model is None:
    memory_model = "Slow"

clk = args.sys_clock or args.cpu_clock or args.clock or "1GHz"
cpu_clk = args.cpu_clock or clk

class MySystem(BaseTestSystem):
    _CPUModel = valid_cpus[args.cpu]
    _MemoryModel = valid_memories[memory_model]
    _Clk = clk
    _CpuClk = cpu_clk
    _UseCaches = bool(args.caches or args.l2cache)
    _CacheLineSize = int(args.cacheline_size)
    _L1ICacheSize = args.l1i_size
    _L1DCacheSize = args.l1d_size
    _L2CacheSize = args.l2_size
    _L1IAssoc = int(args.l1i_assoc)
    _L1DAssoc = int(args.l1d_assoc)
    _L2Assoc = int(args.l2_assoc)
    _DramType = dram_type

system = MySystem()
system.setTestBinary(args.binary)
system.cpu.max_insts_any_thread = args.max_insts

root = Root(full_system=False, system=system)
m5.instantiate()

start_tick = m5.curTick()
start_insts = system.totalInsts()
globalStart = time.time()

exit_event = m5.simulate()
cause = exit_event.getCause()
print("Exit Event: " + cause)

m5.stats.dump()

end_tick = m5.curTick()
end_insts = system.totalInsts()

print("Performance statistics:")
print("Simulated time: %.6fs" % ((end_tick - start_tick) / 1e12))
print("Instructions executed: %d" % (end_insts - start_insts))
print("Ran a total of %.6f simulated seconds" % (m5.curTick() / 1e12))
print("Total wallclock time: %.2fs, %.2f min" %
      (time.time() - globalStart, (time.time() - globalStart) / 60.0))

m5.exit()
