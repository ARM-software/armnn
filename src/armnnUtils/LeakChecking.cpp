//
// Copyright © 2017 Arm Ltd. All rights reserved.
// See LICENSE file in the project root for full license information.
//

#ifdef ARMNN_LEAK_CHECKING_ENABLED

#include "LeakChecking.hpp"
#include "gperftools/heap-checker.h"

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

#endif // ARMNN_LEAK_CHECKING_ENABLED
