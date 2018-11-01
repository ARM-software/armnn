//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "BackendRegistry.hpp"

namespace armnn
{

BackendRegistry& BackendRegistryInstance()
{
    static BackendRegistry instance;
    return instance;
}

} // namespace armnn
