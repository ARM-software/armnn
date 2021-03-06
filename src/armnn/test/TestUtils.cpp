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

Graph& GetGraphForTesting(IOptimizedNetwork* optNet)
{
    return optNet->pOptimizedNetworkImpl->GetGraph();
}

ModelOptions& GetModelOptionsForTesting(IOptimizedNetwork* optNet)
{
    return optNet->pOptimizedNetworkImpl->GetModelOptions();
}

profiling::ProfilingService& GetProfilingService(armnn::RuntimeImpl* runtime)
{
    return runtime->m_ProfilingService;
}

}