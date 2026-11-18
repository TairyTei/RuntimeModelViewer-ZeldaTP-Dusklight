#pragma once
// Shared includes + logging helper. IMPORT_SERVICE lives in mod.cpp only; every other file just
// includes the service headers and uses svc_* directly.
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cmath>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "mods/service.hpp"
#include "mods/svc/log.h"

namespace mi {

// level: 0 info, 1 warn, 2 error
inline void logf(int level, const char* fmt, ...) {
    char buf[512];
    va_list ap;
    va_start(ap, fmt);
    std::vsnprintf(buf, sizeof(buf), fmt, ap);
    va_end(ap);
    if (level >= 2) svc_log->error(mod_ctx, buf);
    else if (level == 1) svc_log->warn(mod_ctx, buf);
    else svc_log->info(mod_ctx, buf);
}

}  // namespace mi
