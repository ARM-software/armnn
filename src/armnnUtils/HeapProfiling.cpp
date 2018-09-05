//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#ifdef ARMNN_HEAP_PROFILING_ENABLED

#include "HeapProfiling.hpp"
#include "gperftools/heap-profiler.h"
#include <sstream>
#include <cstdlib>

namespace armnnUtils
{

ScopedHeapProfiler::ScopedHeapProfiler(const std::string & tag)
: m_Location("/tmp")
, m_Tag(tag)
{
    char * locationFromEnv = ::getenv(ARMNN_HEAP_PROFILE_DUMP_DIR);
    if (locationFromEnv)
    {
        m_Location = locationFromEnv;
    }
    std::stringstream ss;
    ss << m_Location << "/" << m_Tag << ".hprof";
    HeapProfilerStart(ss.str().c_str());
    HeapProfilerDump(m_Tag.c_str());
}

ScopedHeapProfiler::~ScopedHeapProfiler()
{
    HeapProfilerDump(m_Tag.c_str());
}

} // namespace armnnUtils

#endif // ARMNN_HEAP_PROFILING_ENABLED
