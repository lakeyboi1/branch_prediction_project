#include "cpu/pred/multi_branch.hh"
#include "base/intmath.hh"
#include "debug/Fetch.hh"

namespace gem5
{

namespace branch_prediction
{

MultiBranchPredictor::MultiBranchPredictor(
        const MultiBranchPredictorParams &params)
    : BPredUnit(params),
      localTableSize(params.local_table_size),
      localCounterTable(params.local_table_size, 2),
      globalTableSize(params.global_table_size),
      globalCounterTable(params.global_table_size, 2),
      globalHistoryBits(params.global_history_bits),
      globalHistoryReg(0),
      choiceTableSize(params.choice_table_size),
      choiceTable(params.choice_table_size, 2)
{
    assert(isPowerOf2(localTableSize));
    assert(isPowerOf2(globalTableSize));
    assert(isPowerOf2(choiceTableSize));

    localIndexMask   = localTableSize - 1;
    globalIndexMask  = globalTableSize - 1;
    choiceIndexMask  = choiceTableSize - 1;
    globalHistoryMask = (1 << globalHistoryBits) - 1;
}

unsigned
MultiBranchPredictor::getLocalIndex(Addr branch_addr)
{
    return (branch_addr >> 2) & localIndexMask;
}

unsigned
MultiBranchPredictor::getGlobalIndex(Addr branch_addr)
{
    // gshare: XOR branch PC with global history
    return ((branch_addr >> 2) ^ globalHistoryReg) & globalIndexMask;
}

unsigned
MultiBranchPredictor::getChoiceIndex(Addr branch_addr)
{
    return (branch_addr >> 2) & choiceIndexMask;
}

void
MultiBranchPredictor::updateCounter(bool taken, uint8_t &counter)
{
    if (taken && counter < 3)
        counter++;
    else if (!taken && counter > 0)
        counter--;
}

bool
MultiBranchPredictor::lookup(ThreadID tid, Addr branch_addr,
                              void * &bp_history)
{
    BPHistory *history = new BPHistory();

    unsigned localIdx  = getLocalIndex(branch_addr);
    unsigned globalIdx = getGlobalIndex(branch_addr);
    unsigned choiceIdx = getChoiceIndex(branch_addr);

    bool localPred  = localCounterTable[localIdx] >= 2;
    bool globalPred = globalCounterTable[globalIdx] >= 2;
    bool useGlobal  = choiceTable[choiceIdx] >= 2;

    // Save snapshot for update/squash
    history->localCounter     = localCounterTable[localIdx];
    history->globalCounter    = globalCounterTable[globalIdx];
    history->globalHistoryReg = globalHistoryReg;

    bp_history = history;

    // Update global history speculatively
    globalHistoryReg = ((globalHistoryReg << 1) | (useGlobal ? globalPred : localPred))
                       & globalHistoryMask;

    return useGlobal ? globalPred : localPred;
}

void
MultiBranchPredictor::update(ThreadID tid, Addr branch_addr, bool taken,
                              void *bp_history, bool squashed,
                              const StaticInstPtr &inst, Addr corrTarget)
{
    if (squashed) return;

    assert(bp_history);
    BPHistory *history = static_cast<BPHistory*>(bp_history);

    unsigned localIdx  = getLocalIndex(branch_addr);
    unsigned globalIdx = getGlobalIndex(branch_addr);
    unsigned choiceIdx = getChoiceIndex(branch_addr);

    bool localPred  = history->localCounter >= 2;
    bool globalPred = history->globalCounter >= 2;

    // Update choice counter: push toward whichever was correct
    if (localPred != globalPred) {
        if (globalPred == taken)
            updateCounter(true,  choiceTable[choiceIdx]);  // favour global
        else
            updateCounter(false, choiceTable[choiceIdx]);  // favour local
    }

    // Update both local and global counters
    updateCounter(taken, localCounterTable[localIdx]);
    updateCounter(taken, globalCounterTable[globalIdx]);

    delete history;
}

void
MultiBranchPredictor::squash(ThreadID tid, void *bp_history)
{
    assert(bp_history);
    BPHistory *history = static_cast<BPHistory*>(bp_history);

    // Restore global history to pre-speculation state
    globalHistoryReg = history->globalHistoryReg;

    delete history;
}

void
MultiBranchPredictor::uncondBranch(ThreadID tid, Addr pc,
                                    void * &bp_history)
{
    BPHistory *history = new BPHistory();
    history->localCounter     = 0;
    history->globalCounter    = 0;
    history->globalHistoryReg = globalHistoryReg;
    bp_history = history;

    // Unconditional branches always taken — shift a 1 into history
    globalHistoryReg = ((globalHistoryReg << 1) | 1) & globalHistoryMask;
}

} // namespace branch_prediction
} // namespace gem5
