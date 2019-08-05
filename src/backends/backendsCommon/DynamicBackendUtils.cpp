//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "DynamicBackendUtils.hpp"

#include <boost/filesystem/operations.hpp>
#include <boost/algorithm/string.hpp>
#include <boost/log/trivial.hpp>

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

std::vector<std::string> DynamicBackendUtils::GetBackendPaths(const std::string& overrideBackendPath)
{
    // Check if a path where to dynamically load the backends from is given
    if (!overrideBackendPath.empty())
    {
        if (!IsPathValid(overrideBackendPath))
        {
            BOOST_LOG_TRIVIAL(warning) << "WARNING: The given override path for dynamic backends \""
                                       << overrideBackendPath << "\" is not valid";

            return {};
        }

        return std::vector<std::string>{ overrideBackendPath };
    }

    // Expects a colon-separated list: DYNAMIC_BACKEND_PATHS="PATH_1:PATH_2:...:PATH_N"
    const std::string backendPaths = DYNAMIC_BACKEND_PATHS;

    return GetBackendPathsImpl(backendPaths);
}

std::vector<std::string> DynamicBackendUtils::GetBackendPathsImpl(const std::string& backendPaths)
{
    std::unordered_set<std::string> uniqueBackendPaths;
    std::vector<std::string> tempBackendPaths;
    std::vector<std::string> validBackendPaths;

    // Split the given list of paths
    boost::split(tempBackendPaths, backendPaths, boost::is_any_of(":"));

    for (const std::string& path : tempBackendPaths)
    {
        // Check whether the path is valid
        if (!IsPathValid(path))
        {
            continue;
        }

        // Check whether the path is a duplicate
        auto it = uniqueBackendPaths.find(path);
        if (it != uniqueBackendPaths.end())
        {
            // The path is a duplicate
            continue;
        }

        // Add the path to the set of unique paths
        uniqueBackendPaths.insert(path);

        // Add the path to the list of valid paths
        validBackendPaths.push_back(path);
    }

    return validBackendPaths;
}

bool DynamicBackendUtils::IsPathValid(const std::string& path)
{
    if (path.empty())
    {
        BOOST_LOG_TRIVIAL(warning) << "WARNING: The given backend path is empty";
        return false;
    }

    boost::filesystem::path boostPath(path);

    if (!boost::filesystem::exists(boostPath))
    {
        BOOST_LOG_TRIVIAL(warning) << "WARNING: The given backend path \"" << path << "\" does not exist";
        return false;
    }

    if (!boost::filesystem::is_directory(boostPath))
    {
        BOOST_LOG_TRIVIAL(warning) << "WARNING: The given backend path \"" << path << "\" is not a directory";
        return false;
    }

    if (!boostPath.is_absolute())
    {
        BOOST_LOG_TRIVIAL(warning) << "WARNING: The given backend path \"" << path << "\" is not absolute";
        return false;
    }

    return true;
}

} // namespace armnn
