//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//
#pragma once

#include <armnn/Types.hpp>
#include "RegistryCommon.hpp"
#include "IBackendInternal.hpp"

namespace armnn
{

using BackendRegistry = RegistryCommon<IBackendInternal,
                                       IBackendInternalUniquePtr,
                                       EmptyInitializer>;

BackendRegistry& BackendRegistryInstance();

template <>
struct RegisteredTypeName<IBackend>
{
    static const char * Name() { return "IBackend"; }
};

} // namespace armnn
