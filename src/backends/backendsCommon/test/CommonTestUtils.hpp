//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//
#pragma once

#include <Graph.hpp>

using namespace armnn;

namespace
{

// Connects two layers.
void Connect(IConnectableLayer* from, IConnectableLayer* to, const TensorInfo& tensorInfo,
             unsigned int fromIndex = 0, unsigned int toIndex = 0)
{
    from->GetOutputSlot(fromIndex).Connect(to->GetInputSlot(toIndex));
    from->GetOutputSlot(fromIndex).SetTensorInfo(tensorInfo);
}

}
