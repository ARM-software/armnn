//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "TestUtils.hpp"

#include <boost/assert.hpp>

using namespace armnn;

void Connect(armnn::IConnectableLayer* from, armnn::IConnectableLayer* to, const armnn::TensorInfo& tensorInfo,
             unsigned int fromIndex, unsigned int toIndex)
{
    BOOST_ASSERT(from);
    BOOST_ASSERT(to);

    from->GetOutputSlot(fromIndex).Connect(to->GetInputSlot(toIndex));
    from->GetOutputSlot(fromIndex).SetTensorInfo(tensorInfo);
}
