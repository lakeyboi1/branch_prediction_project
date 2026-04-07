#ifndef __CPU_LVP_MARKOV_LVP_HH__
#define __CPU_LVP_MARKOV_LVP_HH__

#include <unordered_map>
#include <vector>
#include <cstdint>

#include "base/types.hh"
#include "base/statistics.hh"
#include "params/MarkovLVP.hh"
#include "sim/sim_object.hh"

namespace gem5 {

class MarkovLVP : public SimObject
{
  public:
    MarkovLVP(const MarkovLVPParams &p);
    bool predict(Addr pc, uint64_t &predVal);
    void update(Addr pc, uint64_t actualVal,
                bool wasPredicted, uint64_t predVal);

  private:
    struct VHTEntry {
        std::vector<uint64_t> strides;
        uint64_t lastVal      = 0;
        unsigned writePtr     = 0;
        bool     valid        = false;
        bool     strideValid  = false;
        bool     hasPending   = false;
        uint64_t pendingPred  = 0;
    };
    std::unordered_map<Addr, VHTEntry> vht;

    struct MarkovEntry {
        uint64_t predictedStride = 0;
        uint8_t  confidence      = 0;
        bool     valid           = false;
    };
    std::vector<MarkovEntry> markovTable;

    unsigned tableSize;
    unsigned contextDepth;
    unsigned confThreshold;
    unsigned maxConf;
    unsigned indexMask;

    unsigned hashContext(Addr pc, const VHTEntry &entry) const;

    struct LVPStats : public statistics::Group {
        LVPStats(statistics::Group *parent);
        statistics::Scalar totalLoads;
        statistics::Scalar predicted;
        statistics::Scalar correct;
        statistics::Scalar incorrect;
        statistics::Scalar noHistory;
        statistics::Scalar noStride;
        statistics::Scalar lowConfidence;
        statistics::Scalar pendingChecked;
    } stats;
};

} // namespace gem5
#endif // __CPU_LVP_MARKOV_LVP_HH__
