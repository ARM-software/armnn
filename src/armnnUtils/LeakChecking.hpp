//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//
#pragma once

#ifdef ARMNN_LEAK_CHECKING_ENABLED

#include <string>
#include <cstddef>
#include <memory>

namespace armnnUtils
{

class ScopedLeakChecker final
{
public:
    ScopedLeakChecker(const std::string & name);
    ~ScopedLeakChecker();

    // Forwarding these to Google Performance Tools.
    static bool IsActive();
    bool NoLeaks();
    // Note that the following two functions only work after
    // NoLeaks() has been called. See explanations in
    // heap-checker.h
    ssize_t BytesLeaked() const;
    ssize_t ObjectsLeaked() const;

private:
    // Hides imlementation so we don't litter other's namespaces
    // with heap checker related stuff.
    struct Impl;
    std::unique_ptr<Impl> m_Impl;

    // No default construction and copying.
    ScopedLeakChecker() = delete;
    ScopedLeakChecker(const ScopedLeakChecker &) = delete;
    ScopedLeakChecker & operator=(const ScopedLeakChecker &) = delete;
};

class ScopedDisableLeakChecking final
{
public:
    ScopedDisableLeakChecking();
    ~ScopedDisableLeakChecking();

private:
    // Hides imlementation so we don't litter other's namespaces
    // with heap checker related stuff.
    struct Impl;
    std::unique_ptr<Impl> m_Impl;

    // No copying.
    ScopedDisableLeakChecking(const ScopedDisableLeakChecking &) = delete;
    ScopedDisableLeakChecking & operator=(const ScopedDisableLeakChecking &) = delete;
};

// disable global leak checks starting from 'main'
void LocalLeakCheckingOnly();

} // namespace armnnUtils

#define ARMNN_SCOPED_LEAK_CHECKER(TAG) \
    armnnUtils::ScopedLeakChecker __scoped_armnn_leak_checker__(TAG)

#define ARMNN_LEAK_CHECKER_IS_ACTIVE() \
    armnnUtils::ScopedLeakChecker::IsActive()

#define ARMNN_NO_LEAKS_IN_SCOPE() \
    __scoped_armnn_leak_checker__.NoLeaks()

#define ARMNN_BYTES_LEAKED_IN_SCOPE() \
    __scoped_armnn_leak_checker__.BytesLeaked()

#define ARMNN_OBJECTS_LEAKED_IN_SCOPE() \
    __scoped_armnn_leak_checker__.ObjectsLeaked()

#define ARMNN_DISABLE_LEAK_CHECKING_IN_SCOPE() \
    armnnUtils::ScopedDisableLeakChecking __disable_leak_checking_in_scope__

#define ARMNN_LOCAL_LEAK_CHECKING_ONLY() \
    armnnUtils::LocalLeakCheckingOnly()

#else // ARMNN_LEAK_CHECKING_ENABLED

#define ARMNN_SCOPED_LEAK_CHECKER(TAG)
#define ARMNN_LEAK_CHECKER_IS_ACTIVE() false
#define ARMNN_NO_LEAKS_IN_SCOPE() true
#define ARMNN_BYTES_LEAKED_IN_SCOPE() 0
#define ARMNN_OBJECTS_LEAKED_IN_SCOPE() 0
#define ARMNN_DISABLE_LEAK_CHECKING_IN_SCOPE()
#define ARMNN_LOCAL_LEAK_CHECKING_ONLY()

#endif // ARMNN_LEAK_CHECKING_ENABLED
