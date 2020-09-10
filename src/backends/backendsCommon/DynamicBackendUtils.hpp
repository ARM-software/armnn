//
// Copyright © 2017 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <armnn/backends/DynamicBackend.hpp>
#include <armnn/backends/IBackendInternal.hpp>

#include <armnn/Exceptions.hpp>

#include <fmt/format.h>

#include <string>
#include <vector>
#if defined(__unix__) || defined(__APPLE__)
#include <dlfcn.h>
#endif

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
    static void DeregisterDynamicBackends(const BackendIdSet& dynamicBackends);

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
#if defined(__unix__) || defined(__APPLE__)
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
        throw RuntimeException(fmt::format("GetEntryPoint error: {}", GetDlError()));
    }

    return entryPoint;
#else
    throw RuntimeException("Dynamic backends not supported on this platform");
#endif
}

} // namespace armnn
