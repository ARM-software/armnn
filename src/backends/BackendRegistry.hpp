//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//
#pragma once

#include <armnn/Types.hpp>
#include "RegistryCommon.hpp"

namespace armnn
{

using BackendRegistry = RegistryCommon<IBackend, IBackendUniquePtr>;

BackendRegistry& BackendRegistryInstance();

template <>
struct RegisteredTypeName<IBackend>
{
    static const char * Name() { return "IBackend"; }
};

} // namespace armnn
