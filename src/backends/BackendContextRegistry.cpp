//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "BackendContextRegistry.hpp"

namespace armnn
{

BackendContextRegistry& BackendContextRegistryInstance()
{
    static BackendContextRegistry instance;
    return instance;
}

} // namespace armnn
