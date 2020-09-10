//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include <armnn/backends/DynamicBackend.hpp>
#include <backendsCommon/DynamicBackendUtils.hpp>

namespace armnn
{

DynamicBackend::DynamicBackend(const void* sharedObjectHandle)
    : m_BackendIdFunction(nullptr)
    , m_BackendVersionFunction(nullptr)
    , m_BackendFactoryFunction(nullptr)
    , m_Handle(const_cast<void*>(sharedObjectHandle), &DynamicBackendUtils::CloseHandle)
{
    if (m_Handle == nullptr)
    {
        throw InvalidArgumentException("Cannot create a DynamicBackend object from an invalid shared object handle");
    }

    // These calls will throw in case of error
    m_BackendIdFunction      = SetFunctionPointer<IdFunctionType>("GetBackendId");
    m_BackendVersionFunction = SetFunctionPointer<VersionFunctionType>("GetVersion");
    m_BackendFactoryFunction = SetFunctionPointer<FactoryFunctionType>("BackendFactory");

    // Check that the backend is compatible with the current Backend API
    BackendId backendId = GetBackendId();
    BackendVersion backendVersion = GetBackendVersion();
    if (!DynamicBackendUtils::IsBackendCompatible(backendVersion))
    {
        // This exception message could not be formatted simply using fmt::format
        std::stringstream message;
        message << "The dynamic backend " << backendId << " (version " << backendVersion <<
        ") is not compatible with the current Backend API (version " << IBackendInternal::GetApiVersion() << ")";

        throw RuntimeException(message.str());
    }
}

BackendId DynamicBackend::GetBackendId()
{
    if (m_BackendIdFunction == nullptr)
    {
        throw RuntimeException("GetBackendId error: invalid function pointer");
    }

    const char* backendId = m_BackendIdFunction();
    if (backendId == nullptr)
    {
        throw RuntimeException("GetBackendId error: invalid backend id");
    }

    return BackendId(backendId);
}

BackendVersion DynamicBackend::GetBackendVersion()
{
    if (m_BackendVersionFunction == nullptr)
    {
        throw RuntimeException("GetBackendVersion error: invalid function pointer");
    }

    uint32_t major = 0;
    uint32_t minor = 0;
    m_BackendVersionFunction(&major, &minor);

    return BackendVersion{ major, minor };
}

IBackendInternalUniquePtr DynamicBackend::GetBackend()
{
    // This call throws in case of error
    return CreateBackend();
}

BackendRegistry::FactoryFunction DynamicBackend::GetFactoryFunction()
{
    if (m_BackendFactoryFunction == nullptr)
    {
        throw RuntimeException("GetFactoryFunction error: invalid function pointer");
    }

    return [this]() -> IBackendInternalUniquePtr
    {
        // This call throws in case of error
        return CreateBackend();
    };
}

template<typename BackendFunctionType>
BackendFunctionType DynamicBackend::SetFunctionPointer(const std::string& backendFunctionName)
{
    if (m_Handle == nullptr)
    {
        throw RuntimeException("SetFunctionPointer error: invalid shared object handle");
    }

    if (backendFunctionName.empty())
    {
        throw RuntimeException("SetFunctionPointer error: backend function name must not be empty");
    }

    // This call will throw in case of error
    auto functionPointer = DynamicBackendUtils::GetEntryPoint<BackendFunctionType>(m_Handle.get(),
                                                                                   backendFunctionName.c_str());
    if (!functionPointer)
    {
        throw RuntimeException("SetFunctionPointer error: invalid backend function pointer returned");
    }

    return functionPointer;
}

IBackendInternalUniquePtr DynamicBackend::CreateBackend()
{
    if (m_BackendFactoryFunction == nullptr)
    {
        throw RuntimeException("CreateBackend error: invalid function pointer");
    }

    auto backendPointer = reinterpret_cast<IBackendInternal*>(m_BackendFactoryFunction());
    if (backendPointer == nullptr)
    {
        throw RuntimeException("CreateBackend error: backend instance must not be null");
    }

    return std::unique_ptr<IBackendInternal>(backendPointer);
}

} // namespace armnn
