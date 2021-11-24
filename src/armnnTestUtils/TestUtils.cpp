//
// Copyright © 2017 Arm Ltd and Contributors. All rights reserved.
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

    try
    {
        from->GetOutputSlot(fromIndex).Connect(to->GetInputSlot(toIndex));
    }
    catch (const std::out_of_range& exc)
    {
        std::ostringstream message;

        if (to->GetType() == armnn::LayerType::FullyConnected && toIndex == 2)
        {
            message << "Tried to connect bias to FullyConnected layer when bias is not enabled: ";
        }

        message << "Failed to connect to input slot "
                << toIndex
                << " on "
                << GetLayerTypeAsCString(to->GetType())
                << " layer "
                << std::quoted(to->GetName())
                << " as the slot does not exist or is unavailable";
        throw LayerValidationException(message.str());
    }

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