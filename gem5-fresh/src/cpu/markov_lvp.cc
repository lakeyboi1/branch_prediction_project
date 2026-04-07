#include "cpu/markov_lvp.hh"

#include <cstring>

#include "base/intmath.hh"
#include "base/logging.hh"

namespace gem5 {

MarkovLVP::LVPStats::LVPStats(statistics::Group *parent)
    : statistics::Group(parent),
      ADD_STAT(totalLoads,     "Total loads seen by LVP"),
      ADD_STAT(predicted,      "Loads for which a value was predicted"),
      ADD_STAT(correct,        "Correctly predicted load values"),
      ADD_STAT(incorrect,      "Incorrectly predicted load values"),
      ADD_STAT(noHistory,      "Predict calls with no VHT entry yet"),
      ADD_STAT(noStride,       "Predict calls with no stride yet"),
      ADD_STAT(lowConfidence,  "Predict calls below confidence threshold"),
      ADD_STAT(pendingChecked, "Times a pending prediction was evaluated")
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
        unsigned idx = (entry.writePtr + contextDepth - 1 - i) % contextDepth;
        h ^= entry.strides[idx] * 2654435761ULL;
        h  = (h << 5) | (h >> 59);
    }
    return static_cast<unsigned>(h) & indexMask;
}

bool
MarkovLVP::predict(Addr pc, uint64_t &predVal)
{
    auto it = vht.find(pc);
    if (it == vht.end() || !it->second.valid) {
        ++stats.noHistory;
        return false;
    }

    VHTEntry &vEntry = it->second;
    if (!vEntry.strideValid) {
        ++stats.noStride;
        return false;
    }

    unsigned mIdx = hashContext(pc, vEntry);
    const MarkovEntry &mEntry = markovTable[mIdx];

    if (!mEntry.valid || mEntry.confidence < confThreshold) {
        ++stats.lowConfidence;
        return false;
    }

    predVal = vEntry.lastVal + mEntry.predictedStride;
    vEntry.hasPending  = true;
    vEntry.pendingPred = predVal;

    ++stats.predicted;
    return true;
}

void
MarkovLVP::update(Addr pc, uint64_t actualVal,
                  bool wasPredicted, uint64_t predVal)
{
    ++stats.totalLoads;

    VHTEntry &vEntry = vht[pc];

    if (!vEntry.valid) {
        vEntry.strides.assign(contextDepth, 0);
        vEntry.writePtr    = 0;
        vEntry.valid       = true;
        vEntry.strideValid = false;
        vEntry.hasPending  = false;
        vEntry.lastVal     = actualVal;
        return;
    }

    // Check pending prediction accuracy
    if (vEntry.hasPending) {
        ++stats.pendingChecked;
        if (actualVal == vEntry.pendingPred)
            ++stats.correct;
        else
            ++stats.incorrect;
        vEntry.hasPending = false;
    }

    uint64_t stride = actualVal - vEntry.lastVal;

    if (vEntry.strideValid) {
        unsigned mIdx = hashContext(pc, vEntry);
        MarkovEntry &mEntry = markovTable[mIdx];

        if (!mEntry.valid) {
            mEntry.predictedStride = stride;
            mEntry.confidence      = 1;
            mEntry.valid           = true;
        } else if (stride == mEntry.predictedStride) {
            if (mEntry.confidence < maxConf)
                ++mEntry.confidence;
        } else {
            if (mEntry.confidence > 0)
                --mEntry.confidence;
            if (mEntry.confidence == 0) {
                mEntry.predictedStride = stride;
                mEntry.confidence      = 1;
            }
        }
    }

    vEntry.strides[vEntry.writePtr] = stride;
    vEntry.writePtr    = (vEntry.writePtr + 1) % contextDepth;
    vEntry.strideValid = true;
    vEntry.lastVal     = actualVal;
}

} // namespace gem5
