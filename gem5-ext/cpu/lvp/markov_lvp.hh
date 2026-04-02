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

    // Called at load-dispatch. Returns true and sets predVal if confident.
    bool predict(Addr pc, uint64_t &predVal);

    // Called at load-commit with the real value from memory.
    void update(Addr pc, uint64_t actualVal,
                bool wasPredicted, uint64_t predVal);

  private:
    // ---- Value History Table ----
    struct VHTEntry {
        std::vector<uint64_t> history;
        unsigned writePtr = 0;
        bool     valid    = false;
    };
    std::unordered_map<Addr, VHTEntry> vht;

    // ---- Markov Prediction Table ----
    struct MarkovEntry {
        uint64_t predictedVal = 0;
        uint8_t  confidence   = 0;
        bool     valid        = false;
    };
    std::vector<MarkovEntry> markovTable;

    // Config params
    unsigned tableSize;
    unsigned contextDepth;
    unsigned confThreshold;
    unsigned maxConf;
    unsigned indexMask;

    unsigned hashContext(Addr pc, const VHTEntry &entry) const;

    // Stats
    struct LVPStats : public statistics::Group {
        LVPStats(statistics::Group *parent);
        statistics::Scalar totalLoads;
        statistics::Scalar predicted;
        statistics::Scalar correct;
        statistics::Scalar incorrect;
    } stats;
};

} // namespace gem5

#endif // __CPU_LVP_MARKOV_LVP_HH__