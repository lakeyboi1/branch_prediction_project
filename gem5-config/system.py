import m5
from m5.objects import *
import argparse

class InfMemory(SimpleMemory):
    latency = '0ns'
    bandwidth = '0B/s'

_parser = argparse.ArgumentParser(add_help=False)
_parser.add_argument(
    "--bp_type",
    default="multi_branch",
    choices=["none", "local", "bimode", "tage_base", "multi_branch"],
    help="Branch predictor to use. All other system parameters are fixed."
)
options, _unknown = _parser.parse_known_args()

# ---------------------------------------------------------------------------
# Branch predictor factory
#
#   none         — NullBP: always predicts not-taken. Baseline to measure
#                  the maximum cost of having no prediction at all.
#   local        — LocalBP (2-bit local): classic 2-bit saturating counter
#                  indexed by branch PC. Simple and well-understood baseline.
#   bimode       — BiModeBP: two counter arrays + direction bit, reduces
#                  aliasing on highly-biased branches vs. local.
#   tage_base    — TAGE: tagged geometric history predictor, state of the
#                  art for general workloads. Upper-bound reference point.
#   multi_branch — MultiBranchPredictor: our custom tournament predictor
#                  combining a local 2-bit table with a global gshare table
#                  and a choice predictor to select between them.
# ---------------------------------------------------------------------------

def make_branch_predictor(bp_type: str):
    if bp_type == "none":
        return NullBP()
    elif bp_type == "local":
        return LocalBP(
            localPredictorSize=2048,
            localCtrBits=2,
        )
    elif bp_type == "bimode":
        return BiModeBP(
            globalPredictorSize=8192,
            choicePredictorSize=8192,
            globalCtrBits=2,
        )
    elif bp_type == "tage_base":
        return TAGE_base()
    elif bp_type == "multi_branch":
        return MultiBranchPredictor(
            local_table_size=2048,
            global_table_size=8192,
            global_history_bits=13,
            choice_table_size=8192,
        )
    else:
        raise ValueError(f"Unknown branch predictor type: {bp_type}")


class BaseTestSystem(System):
    _CPUModel = BaseCPU

    _Clk = "1GHz"

    _UseCaches = False

    # Branch predictor — the only variable in this experiment
    _BranchPredictorType = options.bp_type

    def totalInsts(self):
        return sum([cpu.totalInsts() for cpu in self.cpu])

    def __init__(self):
        super().__init__()

        self.clk_domain = SrcClockDomain(
            clock=self._Clk,
            voltage_domain=VoltageDomain()
        )
        self.mem_mode = "timing"
        self.mem_ranges = [AddrRange("2GB")]

        self.cpu = self._CPUModel()
        self.cpu.clk_domain = self.clk_domain

        if self._BranchPredictorType is not None and hasattr(self.cpu, 'branchPred'):
            self.cpu.branchPred = make_branch_predictor(self._BranchPredictorType)

        self.membus = SystemXBar(width=64)
        self.system_port = self.membus.slave
        self.cpu.icache_port = self.membus.slave
        self.cpu.dcache_port = self.membus.slave

        self.mem = InfMemory()
        self.mem.range = self.mem_ranges[0]
        self.mem.port = self.membus.master

        self.cpu.createInterruptController()
        if m5.defines.buildEnv["TARGET_ISA"] == "x86":
            self.cpu.interrupts[0].pio = self.membus.master
            self.cpu.interrupts[0].int_master = self.membus.slave
            self.cpu.interrupts[0].int_slave = self.membus.master

    def setTestBinary(self, binary_path):
        self.cpu.workload = Process(cmd=[binary_path])
        self.cpu.createThreads()