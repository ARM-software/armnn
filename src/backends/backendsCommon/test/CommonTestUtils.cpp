//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "CommonTestUtils.hpp"

#include <backendsCommon/IBackendInternal.hpp>

using namespace armnn;

void Connect(armnn::IConnectableLayer* from, armnn::IConnectableLayer* to, const armnn::TensorInfo& tensorInfo,
             unsigned int fromIndex, unsigned int toIndex)
{
    from->GetOutputSlot(fromIndex).Connect(to->GetInputSlot(toIndex));
    from->GetOutputSlot(fromIndex).SetTensorInfo(tensorInfo);
}

SubgraphView::InputSlots CreateInputsFrom(const std::vector<Layer*>& layers)
{
    SubgraphView::InputSlots result;
    for (auto&& layer : layers)
    {
        for (auto&& it = layer->BeginInputSlots(); it != layer->EndInputSlots(); ++it)
        {
            result.push_back(&(*it));
        }
    }
    return result;
}

SubgraphView::OutputSlots CreateOutputsFrom(const std::vector<Layer*>& layers)
{
    SubgraphView::OutputSlots result;
    for (auto && layer : layers)
    {
        for (auto&& it = layer->BeginOutputSlots(); it != layer->EndOutputSlots(); ++it)
        {
            result.push_back(&(*it));
        }
    }
    return result;
}

SubgraphView::SubgraphViewPtr CreateSubgraphViewFrom(SubgraphView::InputSlots&& inputs,
                                                     SubgraphView::OutputSlots&& outputs,
                                                     SubgraphView::Layers&& layers)
{
    return std::make_unique<SubgraphView>(std::move(inputs), std::move(outputs), std::move(layers));
}

armnn::IBackendInternalUniquePtr CreateBackendObject(const armnn::BackendId& backendId)
{
    auto& backendRegistry = BackendRegistryInstance();
    auto  backendFactory  = backendRegistry.GetFactory(backendId);
    auto  backendObjPtr   = backendFactory();

    return backendObjPtr;
}
