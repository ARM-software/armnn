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
        throw RuntimeException(boost::str(boost::format("OpenHandle error: %1") % GetDlError()));
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
