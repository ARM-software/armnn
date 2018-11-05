//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//
#pragma once

#include "IBackendInternal.hpp"
#include "RegistryCommon.hpp"

#include <armnn/Types.hpp>

namespace armnn
{

using BackendRegistry = RegistryCommon<IBackendInternal, IBackendInternalUniquePtr>;

BackendRegistry& BackendRegistryInstance();

template <>
struct RegisteredTypeName<IBackend>
{
    static const char * Name() { return "IBackend"; }
};

} // namespace armnn
