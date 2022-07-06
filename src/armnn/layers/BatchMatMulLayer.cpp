//
// Copyright © 2022 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//
#include "BatchMatMulLayer.hpp"

#include <armnn/backends/WorkloadFactory.hpp>
#include "layers/LayerCloneBase.hpp"

namespace armnn
{

BatchMatMulLayer::BatchMatMulLayer(const BatchMatMulDescriptor& param, const char* name)
    : LayerWithParameters(2, 1, LayerType::BatchMatMul, param, name)
{}

std::unique_ptr<IWorkload> BatchMatMulLayer::CreateWorkload(const IWorkloadFactory& factory) const
{
    BatchMatMulQueueDescriptor descriptor;
    SetAdditionalInfo(descriptor);

    return factory.CreateWorkload(LayerType::BatchMatMul, descriptor, PrepInfoAndDesc(descriptor));
}

BatchMatMulLayer* BatchMatMulLayer::Clone(Graph& graph) const
{
    auto layer = CloneBase<BatchMatMulLayer>(graph, m_Param, GetName());

    return std::move(layer);
}

std::vector<TensorShape> BatchMatMulLayer::InferOutputShapes(const std::vector<TensorShape>& inputShapes) const
{
    ARMNN_ASSERT(inputShapes.size() == 2);

    TensorShape inputXShape = inputShapes[0];
    TensorShape inputYShape = inputShapes[1];

    // Note: Take into account what pre-adjoint or pre-transposing will do to the inferred output shape

    TensorShape& longerInput = inputXShape.GetNumDimensions() >= inputYShape.GetNumDimensions()?
                               inputXShape:inputYShape;
    TensorShape& shorterInput = inputXShape.GetNumDimensions() >= inputYShape.GetNumDimensions()?
                                inputYShape:inputXShape;

    unsigned int inputNumDimsOffset = longerInput.GetNumDimensions() - shorterInput.GetNumDimensions();

    unsigned int outputNumDimensions = longerInput.GetNumDimensions();

    std::vector<unsigned int> tensorDimensions(outputNumDimensions, 0);

    auto axesToMul = BatchMatMulDescriptor::GetAxesToMul(m_Param, inputXShape, inputYShape);
    const auto& longerAxesToMul = (axesToMul.first.first >= axesToMul.second.first &&
                             axesToMul.first.second >= axesToMul.second.second) ?
                                 axesToMul.first : axesToMul.second;

    for (unsigned int i = 0; i < outputNumDimensions; ++i)
    {
        if (i == longerAxesToMul.first)
        {
            tensorDimensions[i] = &shorterInput == &inputXShape ? inputXShape[i - inputNumDimsOffset] : inputXShape[i];
        }
        else if(i == longerAxesToMul.second)
        {
            tensorDimensions[i] = &shorterInput == &inputYShape ? inputYShape[i - inputNumDimsOffset] : inputYShape[i];
        }
        else // The other dimensions not to be multiplied (but may be broadcasted)
        {
            // Does NOT validate whether it's a valid broadcast - that's done in the validate func in WorkloadData.cpp
            tensorDimensions[i] = static_cast<int>(i) - static_cast<int>(inputNumDimsOffset) < 0 ?
                longerInput[i] :
                std::max(longerInput[i], shorterInput[i - inputNumDimsOffset]);
        }
    }

    auto outputShape = TensorShape(outputNumDimensions, tensorDimensions.data());
    return std::vector<TensorShape>({ outputShape });
}

void BatchMatMulLayer::ValidateTensorShapesFromInputs()
{
    VerifyLayerConnections(2, CHECK_LOCATION());

    const TensorShape& outputShape = GetOutputSlot(0).GetTensorInfo().GetShape();

    VerifyShapeInferenceType(outputShape, m_ShapeInferenceMethod);

    auto inferredShapes = InferOutputShapes({
        GetInputSlot(0).GetConnection()->GetTensorInfo().GetShape(),
        GetInputSlot(1).GetConnection()->GetTensorInfo().GetShape() });

    ARMNN_ASSERT(inferredShapes.size() == 1);

    ValidateAndCopyShape(outputShape, inferredShapes[0], m_ShapeInferenceMethod, "BatchMatMulLayer");
}

} // namespace armnn