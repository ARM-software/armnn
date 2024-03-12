//
// Copyright © 2017,2024 Arm Ltd. All rights reserved.
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
            throw armnn::InvalidArgumentException("Unknown layer type");
            return "Unknown";
    }
}

}
