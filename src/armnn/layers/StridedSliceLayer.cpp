//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//
#include "StridedSliceLayer.hpp"

#include "LayerCloneBase.hpp"

#include <backendsCommon/WorkloadData.hpp>
#include <backendsCommon/WorkloadFactory.hpp>

namespace armnn
{

StridedSliceLayer::StridedSliceLayer(const armnn::StridedSliceDescriptor& param, const char* name)
    : LayerWithParameters(1, 1, LayerType::StridedSlice, param, name)
{
}

std::unique_ptr<IWorkload> StridedSliceLayer::CreateWorkload(const Graph& graph,
                                                             const IWorkloadFactory& factory) const
{
    StridedSliceQueueDescriptor descriptor;

    descriptor.m_Parameters.m_Begin  = m_Param.m_Begin;
    descriptor.m_Parameters.m_End    = m_Param.m_End;
    descriptor.m_Parameters.m_Stride = m_Param.m_Stride;

    // Optional parameters
    descriptor.m_Parameters.m_BeginMask      = m_Param.m_BeginMask;
    descriptor.m_Parameters.m_EndMask        = m_Param.m_EndMask;
    descriptor.m_Parameters.m_EllipsisMask   = m_Param.m_EllipsisMask;
    descriptor.m_Parameters.m_NewAxisMask    = m_Param.m_NewAxisMask;
    descriptor.m_Parameters.m_ShrinkAxisMask = m_Param.m_ShrinkAxisMask;

    return factory.CreateStridedSlice(descriptor, PrepInfoAndDesc(descriptor, graph));
}

StridedSliceLayer* StridedSliceLayer::Clone(Graph& graph) const
{
    return CloneBase<StridedSliceLayer>(graph, m_Param, GetName());
}

void StridedSliceLayer::ValidateTensorShapesFromInputs()
{
    VerifyLayerConnections(1, CHECK_LOCATION());

    auto inferredShapes = InferOutputShapes({GetInputSlot(0).GetConnection()->GetTensorInfo().GetShape()});

    BOOST_ASSERT(inferredShapes.size() == 1);

    ConditionalThrowIfNotEqual<LayerValidationException>(
                    "StridedSlice: TensorShape set on OutputSlot[0] does not match the inferred shape.",
                    GetOutputSlot(0).GetTensorInfo().GetShape(),
                    inferredShapes[0]);
}

} // namespace armnn