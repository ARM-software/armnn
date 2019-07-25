//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include "IBackendInternal.hpp"

#include <armnn/Exceptions.hpp>

#include <string>
#include <dlfcn.h>

#include <boost/format.hpp>

namespace armnn
{

class DynamicBackendUtils
{
public:
    static void* OpenHandle(const std::string& sharedObjectPath);
    static void CloseHandle(const void* sharedObjectHandle);

    template<typename EntryPointType>
    static EntryPointType GetEntryPoint(const void* sharedObjectHandle, const char* symbolName);

    static bool IsBackendCompatible(const BackendVersion& backendVersion);

protected:
    /// Protected for testing purposes
    static bool IsBackendCompatibleImpl(const BackendVersion& backendApiVersion, const BackendVersion& backendVersion);

private:
    static std::string GetDlError();

    /// This class is to hold utility functions only
    DynamicBackendUtils() = delete;
};

template<typename EntryPointType>
EntryPointType DynamicBackendUtils::GetEntryPoint(const void* sharedObjectHandle, const char* symbolName)
{
    if (sharedObjectHandle == nullptr)
    {
        throw RuntimeException("GetEntryPoint error: invalid handle");
    }

    if (symbolName == nullptr)
    {
        throw RuntimeException("GetEntryPoint error: invalid symbol");
    }

    auto entryPoint = reinterpret_cast<EntryPointType>(dlsym(const_cast<void*>(sharedObjectHandle), symbolName));
    if (!entryPoint)
    {
        throw RuntimeException(boost::str(boost::format("GetEntryPoint error: %1%") % GetDlError()));
    }

    return entryPoint;
}

} // namespace armnn
