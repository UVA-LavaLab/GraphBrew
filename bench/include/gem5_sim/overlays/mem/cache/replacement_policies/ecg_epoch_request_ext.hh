// GraphBrew ECG epoch REQUEST SIDEBAND (the OoO / multicore-general delivery).
//
// The single-slot ecg.extract mailbox (setDecodedEcgExtractHint) and the per-vertex
// table (storeEcgMetadataByVertex) are both compromises: the mailbox races under an
// out-of-order CPU (a later ecg.load's epoch can overwrite an earlier one before its
// fill stamps the line), and the table is O(num_vertices) (the very cost ECG avoids).
//
// The race-free AND HW-realizable delivery is a per-REQUEST sideband: the ecg.load AGU
// tags the demand Request with {dest, epoch} (a few tag bits riding the in-flight load),
// and the LLC reads it on the fill. Because the epoch travels WITH the specific request,
// there is no shared structure to race and no per-vertex storage. gem5's Request is
// Extensible<Request>, so this is a first-class Request::Extension.
//
// In-order TimingSimpleCPU case study: loads are serialized, so the single-slot mailbox
// holds exactly the demanded vertex's epoch when its fill reaches the LLC -> the mailbox
// is mathematically equivalent to this sideband (no race possible). The replacement
// policy is therefore validated in-order via the mailbox; this extension is the same
// information delivered race-free for the O3CPU / multicore form. The read hook below is
// PREFERRED when present (so an O3 ecg.load that attaches it is correct), and falls back
// to the mailbox otherwise.
#ifndef GRAPHBREW_ECG_EPOCH_REQUEST_EXT_HH
#define GRAPHBREW_ECG_EPOCH_REQUEST_EXT_HH

#include <cstdint>
#include <memory>

#include "base/extensible.hh"
#include "mem/request.hh"

namespace gem5 {
namespace replacement_policy {
namespace graph {

// Per-request ECG metadata sideband. Carries the next-reference epoch (the eviction
// signal) plus the optional GRASP/POPT tiers, attached to the ecg.load demand Request.
class EcgEpochExtension
    : public gem5::Extension<gem5::Request, EcgEpochExtension> {
  public:
    EcgEpochExtension(uint32_t dest, uint16_t epoch,
                      uint8_t dbg = 0, uint8_t popt = 0)
        : dest_(dest), epoch_(epoch), dbg_(dbg), popt_(popt) {}

    std::unique_ptr<gem5::ExtensionBase> clone() const override {
        return std::make_unique<EcgEpochExtension>(*this);
    }

    uint32_t dest()  const { return dest_; }
    uint16_t epoch() const { return epoch_; }
    uint8_t  dbg()   const { return dbg_; }
    uint8_t  popt()  const { return popt_; }

  private:
    uint32_t dest_;
    uint16_t epoch_;
    uint8_t  dbg_;
    uint8_t  popt_;
};

// O3/OoO ATTACH (the ecg.load AGU side): tag the demand request with its epoch sideband.
// For the in-order case study this is unused (the mailbox is the equivalent model); a
// custom ecg.load format's initiateAcc calls this on the request it issues.
inline void attachEcgEpoch(const gem5::RequestPtr& req, uint32_t dest, uint16_t epoch,
                           uint8_t dbg = 0, uint8_t popt = 0) {
    if (req) {
        req->setExtension(
            std::make_shared<EcgEpochExtension>(dest, epoch, dbg, popt));
    }
}

// LLC fill READ (the replacement-policy side): if the request carries the sideband,
// return its metadata. Race-free under OoO (no shared mailbox).
inline bool readEcgEpoch(const gem5::RequestPtr& req, uint16_t& epoch_out,
                         uint8_t& dbg_out, uint8_t& popt_out) {
    if (!req) return false;
    auto ext = req->getExtension<EcgEpochExtension>();
    if (!ext) return false;
    epoch_out = ext->epoch();
    dbg_out = ext->dbg();
    popt_out = ext->popt();
    return true;
}

}  // namespace graph
}  // namespace replacement_policy
}  // namespace gem5

#endif  // GRAPHBREW_ECG_EPOCH_REQUEST_EXT_HH
