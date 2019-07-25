//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "DynamicBackendUtils.hpp"

namespace armnn
{

void* DynamicBackendUtils::OpenHandle(const std::string& sharedObjectPath)
{
    if (sharedObjectPath.empty())
    {
        throw RuntimeException("OpenHandle error: shared object path must not be empty");
    }

    void* sharedObjectHandle = dlopen(sharedObjectPath.c_str(), RTLD_LAZY | RTLD_GLOBAL);
    if (!sharedObjectHandle)
    {
        throw RuntimeException(boost::str(boost::format("OpenHandle error: %1%") % GetDlError()));
    }

    return sharedObjectHandle;
}

void DynamicBackendUtils::CloseHandle(const void* sharedObjectHandle)
{
    if (!sharedObjectHandle)
    {
        return;
    }

    dlclose(const_cast<void*>(sharedObjectHandle));
}

bool DynamicBackendUtils::IsBackendCompatible(const BackendVersion &backendVersion)
{
    BackendVersion backendApiVersion = IBackendInternal::GetApiVersion();

    return IsBackendCompatibleImpl(backendApiVersion, backendVersion);
}

bool DynamicBackendUtils::IsBackendCompatibleImpl(const BackendVersion &backendApiVersion,
                                                  const BackendVersion &backendVersion)
{
    return backendVersion.m_Major == backendApiVersion.m_Major &&
           backendVersion.m_Minor <= backendApiVersion.m_Minor;
}

std::string DynamicBackendUtils::GetDlError()
{
    const char* errorMessage = dlerror();
    if (!errorMessage)
    {
        return "";
    }

    return std::string(errorMessage);
}

} // namespace armnn
