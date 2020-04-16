//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "InternalTypes.hpp"

#include <armnn/utility/Assert.hpp>

namespace armnn
{

char const* GetLayerTypeAsCString(LayerType type)
{
    switch (type)
    {
#define X(name) case LayerType::name: return #name;
      LIST_OF_LAYER_TYPE
#undef X
        default:
            ARMNN_ASSERT_MSG(false, "Unknown layer type");
            return "Unknown";
    }
}

}
