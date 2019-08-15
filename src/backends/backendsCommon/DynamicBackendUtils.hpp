//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include "IBackendInternal.hpp"
#include "DynamicBackend.hpp"

#include <armnn/Exceptions.hpp>

#include <string>
#include <dlfcn.h>
#include <vector>

#include <boost/format.hpp>

#if !defined(DYNAMIC_BACKEND_PATHS)
#define DYNAMIC_BACKEND_PATHS ""
#endif

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

    static std::vector<std::string> GetBackendPaths(const std::string& overrideBackendPath = "");
    static bool IsPathValid(const std::string& path);
    static std::vector<std::string> GetSharedObjects(const std::vector<std::string>& backendPaths);

    static std::vector<DynamicBackendPtr> CreateDynamicBackends(const std::vector<std::string>& sharedObjects);
    static BackendIdSet RegisterDynamicBackends(const std::vector<DynamicBackendPtr>& dynamicBackends);

protected:
    /// Protected methods for testing purposes
    static bool IsBackendCompatibleImpl(const BackendVersion& backendApiVersion, const BackendVersion& backendVersion);
    static std::vector<std::string> GetBackendPathsImpl(const std::string& backendPaths);
    static BackendIdSet RegisterDynamicBackendsImpl(BackendRegistry& backendRegistry,
                                                    const std::vector<DynamicBackendPtr>& dynamicBackends);

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
