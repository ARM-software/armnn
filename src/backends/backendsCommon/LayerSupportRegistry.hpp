//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//
#pragma once

#include "RegistryCommon.hpp"
#include <armnn/ILayerSupport.hpp>
#include <armnn/Types.hpp>

namespace armnn
{
using LayerSupportRegistry = RegistryCommon<ILayerSupport,
                                            ILayerSupportSharedPtr,
                                            EmptyInitializer>;

LayerSupportRegistry& LayerSupportRegistryInstance();

template <>
struct RegisteredTypeName<ILayerSupport>
{
    static const char * Name() { return "ILayerSupport"; }
};

} // namespace armnn
