//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//
#pragma once

#include "RegistryCommon.hpp"
#include <armnn/ILayerSupport.hpp>

namespace armnn
{

using LayerSupportRegistry = RegistryCommon<ILayerSupport, ILayerSupportSharedPtr>;

LayerSupportRegistry& LayerSupportRegistryInstance();

template <>
struct RegisteredTypeName<ILayerSupport>
{
    static const char * Name() { return "ILayerSupport"; }
};

} // namespace armnn
