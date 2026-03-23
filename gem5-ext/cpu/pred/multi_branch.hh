#ifndef __CPU_PRED_MULTI_BRANCH_HH__
#define __CPU_PRED_MULTI_BRANCH_HH__

#include <vector>
#include "cpu/pred/bpred_unit.hh"
#include "params/MultiBranchPredictor.hh"

namespace gem5
{

namespace branch_prediction
{

class MultiBranchPredictor : public BPredUnit
{
  public:
    MultiBranchPredictor(const MultiBranchPredictorParams &params);

    bool lookup(ThreadID tid, Addr branch_addr, void * &bp_history) override;
    void update(ThreadID tid, Addr branch_addr, bool taken,
                void *bp_history, bool squashed,
                const StaticInstPtr &inst, Addr corrTarget) override;
    void squash(ThreadID tid, void *bp_history) override;
    void uncondBranch(ThreadID tid, Addr pc, void * &bp_history) override;

  private:
    struct BPHistory {
        uint8_t localCounter;
        uint8_t globalCounter;
        uint32_t globalHistoryReg;
    };

    // Local predictor — indexed by branch PC
    std::vector<uint8_t> localCounterTable;
    unsigned localTableSize;
    unsigned localIndexMask;

    // Global predictor — indexed by global history XOR PC (gshare)
    std::vector<uint8_t> globalCounterTable;
    unsigned globalTableSize;
    unsigned globalIndexMask;

    // Global history shift register
    uint32_t globalHistoryReg;
    unsigned globalHistoryBits;
    uint32_t globalHistoryMask;

    // Choice predictor — selects local vs global
    std::vector<uint8_t> choiceTable;
    unsigned choiceTableSize;
    unsigned choiceIndexMask;

    unsigned getLocalIndex(Addr branch_addr);
    unsigned getGlobalIndex(Addr branch_addr);
    unsigned getChoiceIndex(Addr branch_addr);
    void updateCounter(bool taken, uint8_t &counter);
};

} // namespace branch_prediction
} // namespace gem5

#endif // __CPU_PRED_MULTI_BRANCH_HH__
