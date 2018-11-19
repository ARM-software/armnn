//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "DataLayoutIndexed.hpp"

namespace armnn {

// Definition in include/armnn/Types.hpp
bool operator==(const DataLayout& dataLayout, const DataLayoutIndexed& indexed)
{
    return dataLayout == indexed.GetDataLayout();
}

// Definition in include/armnn/Types.hpp
bool operator==(const DataLayoutIndexed& indexed, const DataLayout& dataLayout)
{
    return indexed.GetDataLayout() == dataLayout;
}

}
