#include "cpu/lvp/markov_lvp.hh"
#include "base/intmath.hh"

namespace gem5 {

MarkovLVP::LVPStats::LVPStats(statistics::Group *parent)
    : statistics::Group(parent),
      ADD_STAT(totalLoads, "Total loads seen by LVP"),
      ADD_STAT(predicted,  "Loads for which a value was predicted"),
      ADD_STAT(correct,    "Correctly predicted load values"),
      ADD_STAT(incorrect,  "Incorrectly predicted load values")
{}

MarkovLVP::MarkovLVP(const MarkovLVPParams &p)
    : SimObject(p),
      tableSize(p.table_size),
      contextDepth(p.context_depth),
      confThreshold(p.conf_threshold),
      maxConf(p.max_conf),
      indexMask(p.table_size - 1),
      markovTable(p.table_size),
      stats(this)
{
    assert(isPowerOf2(tableSize));
    assert(contextDepth >= 1);
}

unsigned
MarkovLVP::hashContext(Addr pc, const VHTEntry &entry) const
{
    uint64_t h = pc ^ (pc >> 32);
    for (unsigned i = 0; i < contextDepth; ++i) {
        // Walk backwards through the ring buffer
        unsigned idx = (entry.writePtr + contextDepth - 1 - i) % contextDepth;
        h ^= entry.history[idx] * 2654435761ULL;
        h  = (h << 5) | (h >> 59);
    }
    return static_cast<unsigned>(h) & indexMask;
}

bool
MarkovLVP::predict(Addr pc, uint64_t &predVal)
{
    auto it = vht.find(pc);
    if (it == vht.end() || !it->second.valid)
        return false;

    unsigned mIdx = hashContext(pc, it->second);
    MarkovEntry &mEntry = markovTable[mIdx];

    if (!mEntry.valid || mEntry.confidence < confThreshold)
        return false;

    predVal = mEntry.predictedVal;
    ++stats.predicted;
    return true;
}

void
MarkovLVP::update(Addr pc, uint64_t actualVal,
                  bool wasPredicted, uint64_t predVal)
{
    ++stats.totalLoads;

    // 1. Get or create VHT entry
    VHTEntry &vEntry = vht[pc];
    if (!vEntry.valid) {
        vEntry.history.assign(contextDepth, 0);
        vEntry.writePtr = 0;
        vEntry.valid    = true;
    }

    // 2. Update Markov table using CURRENT context (before appending actualVal)
    unsigned mIdx = hashContext(pc, vEntry);
    MarkovEntry &mEntry = markovTable[mIdx];

    if (!mEntry.valid) {
        mEntry.predictedVal = actualVal;
        mEntry.confidence   = 1;
        mEntry.valid        = true;
    } else if (actualVal == mEntry.predictedVal) {
        if (mEntry.confidence < maxConf)
            ++mEntry.confidence;
    } else {
        if (mEntry.confidence > 0)
            --mEntry.confidence;
        if (mEntry.confidence == 0) {
            mEntry.predictedVal = actualVal;
            mEntry.confidence   = 1;
        }
    }

    // 3. Record prediction accuracy
    if (wasPredicted) {
        if (actualVal == predVal)
            ++stats.correct;
        else
            ++stats.incorrect;
    }

    // 4. Append actualVal to VHT ring buffer
    vEntry.history[vEntry.writePtr] = actualVal;
    vEntry.writePtr = (vEntry.writePtr + 1) % contextDepth;
}

} // namespace gem5