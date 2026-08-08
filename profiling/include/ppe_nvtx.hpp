#pragma once

#ifdef PPE_ENABLE_NVTX
#include <nvtx3/nvToolsExt.h>
#endif

namespace ppe::profiling {

class ScopedRange {
public:
    explicit ScopedRange(const char* name) noexcept {
#ifdef PPE_ENABLE_NVTX
        nvtxRangePushA(name);
#else
        (void)name;
#endif
    }

    ~ScopedRange() {
#ifdef PPE_ENABLE_NVTX
        nvtxRangePop();
#endif
    }

    ScopedRange(const ScopedRange&) = delete;
    ScopedRange& operator=(const ScopedRange&) = delete;
};

}  // namespace ppe::profiling

#define PPE_NVTX_JOIN_IMPL(left, right) left##right
#define PPE_NVTX_JOIN(left, right) PPE_NVTX_JOIN_IMPL(left, right)
#define PPE_NVTX_RANGE(name) \
    ::ppe::profiling::ScopedRange PPE_NVTX_JOIN(ppe_nvtx_range_, __LINE__)(name)
