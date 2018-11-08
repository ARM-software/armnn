//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//
#pragma once

#include "RegistryCommon.hpp"
#include <armnn/Types.hpp>

namespace armnn
{
class IBackendInternal;
using IBackendInternalUniquePtr = std::unique_ptr<IBackendInternal>;
using BackendRegistry = RegistryCommon<IBackendInternal, IBackendInternalUniquePtr>;

BackendRegistry& BackendRegistryInstance();

template <>
struct RegisteredTypeName<IBackend>
{
    static const char * Name() { return "IBackend"; }
};

} // namespace armnn