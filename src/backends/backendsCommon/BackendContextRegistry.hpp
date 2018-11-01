//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//
#pragma once

#include "IBackendContext.hpp"
#include "RegistryCommon.hpp"

#include <armnn/IRuntime.hpp>
#include <armnn/Types.hpp>

namespace armnn
{

using BackendContextRegistry = RegistryCommon<IBackendContext,
                                              IBackendContextUniquePtr,
                                              IRuntime::CreationOptions>;

BackendContextRegistry& BackendContextRegistryInstance();

template <>
struct RegisteredTypeName<IBackendContext>
{
    static const char * Name() { return "IBackendContext"; }
};

} // namespace armnn
