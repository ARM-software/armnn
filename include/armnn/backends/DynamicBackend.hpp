//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include "IBackendInternal.hpp"

#include <armnn/BackendRegistry.hpp>

#include <functional>
#include <memory>

namespace armnn
{

class DynamicBackend final
{
public:
    using HandleCloser = std::function<void(const void*)>;
    using HandlePtr = std::unique_ptr<void, HandleCloser>;

    explicit DynamicBackend(const void* sharedObjectHandle);

    /// Public dynamic backend functions
    BackendId GetBackendId();
    BackendVersion GetBackendVersion();
    IBackendInternalUniquePtr GetBackend();
    BackendRegistry::FactoryFunction GetFactoryFunction();

private:
    /// Private utility functions
    template<typename BackendFunctionType>
    BackendFunctionType SetFunctionPointer(const std::string& backendFunctionName);
    IBackendInternalUniquePtr CreateBackend();

    /// Backend function pointer types
    using IdFunctionType      = const char*(*)();
    using VersionFunctionType = void(*)(uint32_t*, uint32_t*);
    using FactoryFunctionType = void*(*)();

    /// Backend function pointers
    IdFunctionType      m_BackendIdFunction;
    VersionFunctionType m_BackendVersionFunction;
    FactoryFunctionType m_BackendFactoryFunction;

    /// Shared object handle
    HandlePtr m_Handle;
};

using DynamicBackendPtr = std::unique_ptr<DynamicBackend>;

} // namespace armnn
