//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "LayerSupportRegistry.hpp"

namespace armnn
{

LayerSupportRegistry& LayerSupportRegistryInstance()
{
    static LayerSupportRegistry instance;
    return instance;
}

} // namespace armnn
