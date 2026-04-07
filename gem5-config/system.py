# -*- coding: utf-8 -*-
# system.py — Branch predictor + Markov value predictor experiment system.
#
# Fixed settings to isolate predictors as the sole variable:
#   - InfMemory (0ns latency): eliminates memory latency noise
#   - Caches disabled: eliminates cache miss noise
#   - 1GHz fixed clock: cancels out in relative comparisons

import m5
from m5.objects import *
import argparse

# ---------------------------------------------------------------------------
# Memory
# ---------------------------------------------------------------------------
class InfMemory(SimpleMemory):
    latency   = '0ns'
    bandwidth = '0B/s'

# ---------------------------------------------------------------------------
# Branch predictor argument
# ---------------------------------------------------------------------------
_parser = argparse.ArgumentParser(add_help=False)
_parser.add_argument(
    "--bp_type",
    default="local",
    choices=["none", "local", "tage_base"],
    help="Branch predictor to use."
)
options, _unknown = _parser.parse_known_args()

# ---------------------------------------------------------------------------
# Branch predictor factory
#
#   none      — 1-entry LocalBP: almost always mispredicts, lower bound
#   local     — LocalBP: classic 2-bit saturating counter per PC
#   tage_base — TAGE: state-of-the-art multi-history predictor, upper bound
# ---------------------------------------------------------------------------
def make_branch_predictor(bp_type: str):
    bp = BranchPredictor()
    if bp_type == "none":
        bp.conditionalBranchPred = LocalBP(
            localPredictorSize=2,
            localCtrBits=1,
        )
    elif bp_type == "local":
        bp.conditionalBranchPred = LocalBP(
            localPredictorSize=2048,
            localCtrBits=2,
        )
    elif bp_type == "tage_base":
        bp.conditionalBranchPred = TAGE(tage=TAGEBase())
    else:
        raise ValueError(f"Unknown branch predictor type: {bp_type}")
    return bp


# ---------------------------------------------------------------------------
# System
# ---------------------------------------------------------------------------
class BaseTestSystem(System):
    _CPUModel            = BaseCPU
    _Clk                 = "1GHz"
    _BranchPredictorType = options.bp_type

    def totalInsts(self):
        return sum([cpu.totalInsts() for cpu in self.cpu])

    def __init__(self):
        super().__init__()

        self.clk_domain = SrcClockDomain(
            clock=self._Clk,
            voltage_domain=VoltageDomain()
        )
        self.mem_mode   = "timing"
        self.mem_ranges = [AddrRange("2GB")]

        self.cpu = self._CPUModel()
        self.cpu.clk_domain = self.clk_domain

        # Wire in branch predictor
        if self._BranchPredictorType is not None and \
                hasattr(self.cpu, 'branchPred'):
            self.cpu.branchPred = make_branch_predictor(
                self._BranchPredictorType
            )

        # Memory bus
        self.membus = SystemXBar(width=64)
        self.system_port      = self.membus.cpu_side_ports
        self.cpu.icache_port  = self.membus.cpu_side_ports
        self.cpu.dcache_port  = self.membus.cpu_side_ports

        self.mem       = InfMemory()
        self.mem.range = self.mem_ranges[0]
        self.mem.port  = self.membus.mem_side_ports

        # X86 interrupt controller
        self.cpu.createInterruptController()
        self.cpu.interrupts[0].pio           = self.membus.mem_side_ports
        self.cpu.interrupts[0].int_requestor = self.membus.cpu_side_ports
        self.cpu.interrupts[0].int_responder = self.membus.mem_side_ports

    def setTestBinary(self, binary_path):
        self.workload     = SEWorkload.init_compatible(binary_path)
        process           = Process(cmd=[binary_path])
        self.cpu.workload = process
        self.cpu.createThreads()
