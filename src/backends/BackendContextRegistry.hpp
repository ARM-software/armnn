//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//
#pragma once

#include <armnn/Types.hpp>
#include <armnn/IRuntime.hpp>
#include "RegistryCommon.hpp"
#include "IBackendContext.hpp"

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
