//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#ifdef ARMNN_LEAK_CHECKING_ENABLED

#include "LeakChecking.hpp"
#include "gperftools/heap-checker.h"

namespace armnnUtils
{

struct ScopedLeakChecker::Impl
{
    HeapLeakChecker m_LeakChecker;

    Impl(const std::string & name)
    : m_LeakChecker(name.c_str())
    {
    }
};

ScopedLeakChecker::ScopedLeakChecker(const std::string & name)
: m_Impl(new Impl(name))
{
}

ScopedLeakChecker::~ScopedLeakChecker() {}

bool ScopedLeakChecker::IsActive()
{
    return HeapLeakChecker::IsActive();
}

bool ScopedLeakChecker::NoLeaks()
{
    return (IsActive() ? m_Impl->m_LeakChecker.NoLeaks() : true);
}

ssize_t ScopedLeakChecker::BytesLeaked() const
{
    return (IsActive() ? m_Impl->m_LeakChecker.BytesLeaked(): 0);
}

ssize_t ScopedLeakChecker::ObjectsLeaked() const
{
    return (IsActive() ? m_Impl->m_LeakChecker.ObjectsLeaked(): 0 );
}

struct ScopedDisableLeakChecking::Impl
{
    HeapLeakChecker::Disabler m_Disabler;
};

ScopedDisableLeakChecking::ScopedDisableLeakChecking()
: m_Impl(new Impl)
{
}

ScopedDisableLeakChecking::~ScopedDisableLeakChecking()
{
}

void LocalLeakCheckingOnly()
{
    auto * globalChecker = HeapLeakChecker::GlobalChecker();
    if (globalChecker)
    {
        // Don't care about global leaks and make sure we won't report any.
        // This is because leak checking supposed to run in well defined
        // contexts through the ScopedLeakChecker, otherwise we risk false
        // positives because of external factors.
        globalChecker->NoGlobalLeaks();
        globalChecker->CancelGlobalCheck();
    }
}

} // namespace armnnUtils

#endif // ARMNN_LEAK_CHECKING_ENABLED
