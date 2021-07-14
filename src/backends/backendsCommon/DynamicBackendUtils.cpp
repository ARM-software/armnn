//
// Copyright © 2017 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include <armnn/Logging.hpp>
#include <backendsCommon/DynamicBackendUtils.hpp>
#include "armnn/utility/StringUtils.hpp"
#include <armnnUtils/Filesystem.hpp>

#include <regex>

namespace armnn
{

void* DynamicBackendUtils::OpenHandle(const std::string& sharedObjectPath)
{
#if defined(__unix__) || defined(__APPLE__)
    if (sharedObjectPath.empty())
    {
        throw RuntimeException("OpenHandle error: shared object path must not be empty");
    }

    void* sharedObjectHandle = dlopen(sharedObjectPath.c_str(), RTLD_LAZY);
    if (!sharedObjectHandle)
    {
        throw RuntimeException(fmt::format("OpenHandle error: {}", GetDlError()));
    }

    return sharedObjectHandle;
#else
    throw RuntimeException("Dynamic backends not supported on this platform");
#endif
}

void DynamicBackendUtils::CloseHandle(const void* sharedObjectHandle)
{
#if defined(__unix__) || defined(__APPLE__)
    if (!sharedObjectHandle)
    {
        return;
    }

    dlclose(const_cast<void*>(sharedObjectHandle));
#else
    throw RuntimeException("Dynamic backends not supported on this platform");
#endif
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
#if defined(__unix__) || defined(__APPLE__)
    const char* errorMessage = dlerror();
    if (!errorMessage)
    {
        return "";
    }

    return std::string(errorMessage);
#else
    throw RuntimeException("Dynamic backends not supported on this platform");
#endif
}

std::vector<std::string> DynamicBackendUtils::GetBackendPaths(const std::string& overrideBackendPath)
{
    // Check if a path where to dynamically load the backends from is given
    if (!overrideBackendPath.empty())
    {
        if (!IsPathValid(overrideBackendPath))
        {
            ARMNN_LOG(warning) << "WARNING: The given override path for dynamic backends \""
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
    // Check if there's any path to process at all
    if (backendPaths.empty())
    {
        // Silently return without issuing a warning as no paths have been passed, so
        // the whole dynamic backend loading feature can be considered as disabled
        return {};
    }

    std::unordered_set<std::string> uniqueBackendPaths;
    std::vector<std::string> validBackendPaths;

    // Split the given list of paths
    std::vector<std::string> tempBackendPaths = armnn::stringUtils::StringTokenizer(backendPaths, ":");

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
        ARMNN_LOG(warning) << "WARNING: The given backend path is empty";
        return false;
    }

    fs::path fsPath(path);

    if (!fs::exists(fsPath))
    {
        ARMNN_LOG(warning) << "WARNING: The given backend path \"" << path << "\" does not exist";
        return false;
    }

    if (!fs::is_directory(fsPath))
    {
        ARMNN_LOG(warning) << "WARNING: The given backend path \"" << path << "\" is not a directory";
        return false;
    }

    if (!fsPath.is_absolute())
    {
        ARMNN_LOG(warning) << "WARNING: The given backend path \"" << path << "\" is not absolute";
        return false;
    }

    return true;
}

std::vector<std::string> DynamicBackendUtils::GetSharedObjects(const std::vector<std::string>& backendPaths)
{
    std::unordered_set<std::string> uniqueSharedObjects;
    std::vector<std::string> sharedObjects;

    for (const std::string& backendPath : backendPaths)
    {
        using namespace fs;

        // Check if the path is valid. In case of error, IsValidPath will log an error message
        if (!IsPathValid(backendPath))
        {
            continue;
        }

        // Get all the files in the current path in alphabetical order
        std::vector<path> backendPathFiles;
        std::copy(directory_iterator(backendPath), directory_iterator(), std::back_inserter(backendPathFiles));
        std::sort(backendPathFiles.begin(), backendPathFiles.end());

        // Go through all the files in the current backend path
        for (const path& backendPathFile : backendPathFiles)
        {
            // Get only the name of the file (without the full path)
            std::string filename = backendPathFile.filename().string();

            if (filename.empty())
            {
                // Empty filename
                continue;
            }

            path canonicalPath;
            try
            {
                // Get the canonical path for the current file, it will throw if for example the file is a
                // symlink that cannot be resolved
                canonicalPath = canonical(backendPathFile);
            }
            catch (const filesystem_error& e)
            {
                ARMNN_LOG(warning) << "GetSharedObjects warning: " << e.what();
            }
            if (canonicalPath.empty())
            {
                // No such file or perhaps a symlink that couldn't be resolved
                continue;
            }

            // Check if the current filename matches the expected naming convention
            // The expected format is: <vendor>_<name>_backend.so[<version>]
            // e.g. "Arm_GpuAcc_backend.so" or "Arm_GpuAcc_backend.so.1.2"
            const std::regex dynamicBackendRegex("^[a-zA-Z0-9]+_[a-zA-Z0-9]+_backend.so(\\.[0-9]+)*$");

            bool filenameMatch = false;
            try
            {
                // Match the filename to the expected naming scheme
                filenameMatch = std::regex_match(filename, dynamicBackendRegex);
            }
            catch (const std::exception& e)
            {
                ARMNN_LOG(warning) << "GetSharedObjects warning: " << e.what();
            }
            if (!filenameMatch)
            {
                // Filename does not match the expected naming scheme (or an error has occurred)
                continue;
            }

            // Append the valid canonical path to the output list only if it's not a duplicate
            std::string validCanonicalPath = canonicalPath.string();
            auto it = uniqueSharedObjects.find(validCanonicalPath);
            if (it == uniqueSharedObjects.end())
            {
                // Not a duplicate, append the canonical path to the output list
                sharedObjects.push_back(validCanonicalPath);

                // Add the canonical path to the collection of unique shared objects
                uniqueSharedObjects.insert(validCanonicalPath);
            }
        }
    }

    return sharedObjects;
}

std::vector<DynamicBackendPtr> DynamicBackendUtils::CreateDynamicBackends(const std::vector<std::string>& sharedObjects)
{
    // Create a list of dynamic backends
    std::vector<DynamicBackendPtr> dynamicBackends;
    for (const std::string& sharedObject : sharedObjects)
    {
        // Create a handle to the shared object
        void* sharedObjectHandle = nullptr;
        try
        {
            sharedObjectHandle = DynamicBackendUtils::OpenHandle(sharedObject);
        }
        catch (const RuntimeException& e)
        {
            ARMNN_LOG(warning) << "Cannot create a handle to the shared object file \""
                               << sharedObject << "\": " << e.what();
            continue;
        }
        if (!sharedObjectHandle)
        {
            ARMNN_LOG(warning) << "Invalid handle to the shared object file \"" << sharedObject << "\"";

            continue;
        }

        // Create a dynamic backend object
        DynamicBackendPtr dynamicBackend;
        try
        {
            dynamicBackend.reset(new DynamicBackend(sharedObjectHandle));
        }
        catch (const Exception& e)
        {
            ARMNN_LOG(warning) << "Cannot create a valid dynamic backend from the shared object file \""
                               << sharedObject << "\": " << e.what();
            continue;
        }
        if (!dynamicBackend)
        {
            ARMNN_LOG(warning) << "Invalid dynamic backend object for the shared object file \""
                               << sharedObject << "\"";
            continue;
        }

        // Append the newly created dynamic backend to the list
        dynamicBackends.push_back(std::move(dynamicBackend));
    }

    return dynamicBackends;
}

void DynamicBackendUtils::DeregisterDynamicBackends(const BackendIdSet& dynamicBackends)
{
    // Get a reference of the backend registry
    BackendRegistry& backendRegistry = BackendRegistryInstance();

    for (const auto& id : dynamicBackends)
    {
        backendRegistry.Deregister(id);
    }

}

BackendIdSet DynamicBackendUtils::RegisterDynamicBackends(const std::vector<DynamicBackendPtr>& dynamicBackends)
{
    // Get a reference of the backend registry
    BackendRegistry& backendRegistry = BackendRegistryInstance();

    // Register the dynamic backends in the backend registry, and return a list of registered backend ids
    return RegisterDynamicBackendsImpl(backendRegistry, dynamicBackends);
}

BackendIdSet DynamicBackendUtils::RegisterDynamicBackendsImpl(BackendRegistry& backendRegistry,
                                                              const std::vector<DynamicBackendPtr>& dynamicBackends)
{
    // Initialize the list of registered backend ids
    BackendIdSet registeredBackendIds;

    // Register the dynamic backends in the backend registry
    for (const DynamicBackendPtr& dynamicBackend : dynamicBackends)
    {
        // Get the id of the dynamic backend
        BackendId dynamicBackendId;
        try
        {
            dynamicBackendId = dynamicBackend->GetBackendId();
        }
        catch (const RuntimeException& e)
        {
            ARMNN_LOG(warning) << "Cannot register dynamic backend, "
                               << "an error has occurred when getting the backend id: " << e.what();
            continue;
        }
        if (dynamicBackendId.IsEmpty() ||
            dynamicBackendId.IsUndefined())
        {
            ARMNN_LOG(warning) << "Cannot register dynamic backend, invalid backend id: " << dynamicBackendId;
            continue;
        }

        // Check whether the dynamic backend is already registered
        bool backendAlreadyRegistered = backendRegistry.IsBackendRegistered(dynamicBackendId);
        if (backendAlreadyRegistered)
        {
            ARMNN_LOG(warning) << "Cannot register dynamic backend \"" << dynamicBackendId
                               << "\": backend already registered";
            continue;
        }

        // Get the dynamic backend factory function
        BackendRegistry::FactoryFunction dynamicBackendFactoryFunction = nullptr;
        try
        {
            dynamicBackendFactoryFunction = dynamicBackend->GetFactoryFunction();
        }
        catch (const RuntimeException& e)
        {
            ARMNN_LOG(warning) << "Cannot register dynamic backend \"" << dynamicBackendId
                               << "\": an error has occurred when getting the backend factory function: "
                               << e.what();
            continue;
        }
        if (dynamicBackendFactoryFunction == nullptr)
        {
            ARMNN_LOG(warning) << "Cannot register dynamic backend \"" << dynamicBackendId
                               << "\": invalid backend factory function";
            continue;
        }

        // Register the dynamic backend
        try
        {
            backendRegistry.Register(dynamicBackendId, dynamicBackendFactoryFunction);
        }
        catch (const InvalidArgumentException& e)
        {
            ARMNN_LOG(warning) << "An error has occurred when registering the dynamic backend \""
                               << dynamicBackendId << "\": " << e.what();
            continue;
        }

        // Add the id of the dynamic backend just registered to the list of registered backend ids
        registeredBackendIds.insert(dynamicBackendId);
    }

    return registeredBackendIds;
}

} // namespace armnn
