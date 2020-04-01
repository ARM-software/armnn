//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "TestUtils.hpp"

#include <armnn/utility/Assert.hpp>

using namespace armnn;

void Connect(armnn::IConnectableLayer* from, armnn::IConnectableLayer* to, const armnn::TensorInfo& tensorInfo,
             unsigned int fromIndex, unsigned int toIndex)
{
    ARMNN_ASSERT(from);
    ARMNN_ASSERT(to);

    from->GetOutputSlot(fromIndex).Connect(to->GetInputSlot(toIndex));
    from->GetOutputSlot(fromIndex).SetTensorInfo(tensorInfo);
}

namespace armnn
{

profiling::ProfilingService& GetProfilingService(armnn::Runtime* runtime)
{
    return runtime->m_ProfilingService;
}

}