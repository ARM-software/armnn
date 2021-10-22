//
// Copyright © 2020 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include "Optimization.hpp"
#include <armnnUtils/DataLayoutIndexed.hpp>
#include <ResolveType.hpp>

namespace armnn
{
namespace optimizations
{

template <typename ConvLayer, armnn::DataType ArmnnType,
          typename T = armnn::ResolveType<ArmnnType>>
class FuseBatchNorm
{
public:
    /// Run for every exclusive connection between any base Convolution layer and a child BatchNorm layer for not
    /// quantized layers.
    /// The child will be removed, the base will be removed if it's left unconnected. A new Convolution layer will
    /// be added, its weights and bias will be calculated using the weights and bias of the base Convolution layer
    /// combined with the parameters of the child BatchNorm layer.
    void Run(Graph& graph, InputSlot& connection) const
    {
        Layer& base  = connection.GetConnectedOutputSlot()->GetOwningLayer();
        Layer& child = connection.GetOwningLayer();

        bool depthwise = (base.GetType() == LayerType::DepthwiseConvolution2d);

        ARMNN_ASSERT(base.GetType() == LayerType::Convolution2d || depthwise);
        ARMNN_ASSERT(child.GetType() == LayerType::BatchNormalization);

        if (base.GetDataType() == ArmnnType && child.GetDataType() == ArmnnType)
        {
            OutputSlot* parentOut = base.GetInputSlot(0).GetConnectedOutputSlot();
            auto convLayer      = PolymorphicDowncast<ConvLayer*>(&base);
            auto batchNormLayer = PolymorphicDowncast<BatchNormalizationLayer*>(&child);

            // Read convolution and batch norm parameters
            BatchNormalizationDescriptor batchNormDescriptor = batchNormLayer->GetParameters();
            auto epsilon = batchNormDescriptor.m_Eps;
            IgnoreUnused(epsilon);

            ConstTensor betaTensor(batchNormLayer->m_Beta->GetTensorInfo(), batchNormLayer->m_Beta->Map(true));
            ConstTensor gammaTensor(batchNormLayer->m_Gamma->GetTensorInfo(), batchNormLayer->m_Gamma->Map(true));
            ConstTensor meanTensor(batchNormLayer->m_Mean->GetTensorInfo(), batchNormLayer->m_Mean->Map(true));
            ConstTensor varTensor(batchNormLayer->m_Variance->GetTensorInfo(), batchNormLayer->m_Variance->Map(true));

            auto convDescriptor = convLayer->GetParameters();
            auto weightsInfo(convLayer->m_Weight->GetTensorInfo());
            ConstTensor weightsTensor(weightsInfo, convLayer->m_Weight->Map(true));

            armnnUtils::DataLayoutIndexed dataLayout(convDescriptor.m_DataLayout);
            auto weightsShape = weightsInfo.GetShape();
            const unsigned int inputChannels   = parentOut->GetTensorInfo().GetShape()[dataLayout.GetChannelsIndex()];
            const unsigned int depthMultiplier = depthwise ? weightsShape[3] / inputChannels : 1;
            const unsigned int outputChannels  = depthwise ? weightsShape[3] : weightsShape[0];
            const unsigned int weightsHeight   = depthwise ? weightsShape[1] :
                                                             weightsShape[dataLayout.GetHeightIndex()];
            const unsigned int weightsWidth    = depthwise ? weightsShape[2] :
                                                             weightsShape[dataLayout.GetWidthIndex()];

            const auto* weightsBuffer = static_cast<const T*>(weightsTensor.GetMemoryArea());
            const auto* betaBuffer    = static_cast<const T*>(betaTensor.GetMemoryArea());
            const auto* gammaBuffer   = static_cast<const T*>(gammaTensor.GetMemoryArea());
            const auto* meanBuffer    = static_cast<const T*>(meanTensor.GetMemoryArea());
            const auto* varBuffer     = static_cast<const T*>(varTensor.GetMemoryArea());

            std::vector<T> weightsVector (weightsBuffer, weightsBuffer + weightsTensor.GetNumElements());
            std::vector<T> betaVector    (betaBuffer, betaBuffer + betaTensor.GetNumElements());
            std::vector<T> gammaVector   (gammaBuffer, gammaBuffer + gammaTensor.GetNumElements());
            std::vector<T> meanVector    (meanBuffer, meanBuffer + meanTensor.GetNumElements());
            std::vector<T> varianceVector(varBuffer, varBuffer + varTensor.GetNumElements());

            // fusedWeights = ( gamma * weights ) / ( std - epsilon);
            std::vector<T> fusedWeightsVector(weightsVector.size());

            for (unsigned int cInput = 0; cInput < inputChannels; ++cInput)
            {
                for (unsigned int cOut = 0; cOut < outputChannels; ++cOut)
                {
                    T mult = gammaVector[cOut] / static_cast<T>(sqrtf (varianceVector[cOut] + epsilon));

                    for (unsigned int h = 0; h < weightsHeight; ++h)
                    {
                        for (unsigned int w = 0; w < weightsWidth; ++w)
                        {
                            unsigned int weightsIdx = 0;

                            if (depthwise)
                            {
                                cInput = cOut / depthMultiplier;
                                weightsIdx = w * outputChannels + cOut +
                                             h * weightsWidth * outputChannels;
                            }
                            else if (convDescriptor.m_DataLayout == DataLayout::NHWC)
                            {
                                weightsIdx = cOut * weightsHeight * weightsWidth * inputChannels +
                                             h * weightsWidth * inputChannels +
                                             w * inputChannels +
                                             cInput;
                            }
                            else
                            {
                                weightsIdx = cOut * weightsWidth * weightsHeight * inputChannels +
                                             cInput * weightsWidth * weightsHeight +
                                             h * weightsWidth +
                                             w;
                            }
                            fusedWeightsVector[weightsIdx] = mult * weightsVector[weightsIdx];
                        }
                    }
                }
            }
            ConstTensor fusedWeightsTensor(weightsInfo, fusedWeightsVector);

            //  fusedBias = (gamma * (bias - mean)) / (variance - epsilon) + beta;
            std::vector<T> fusedBiasVector(outputChannels);
            if (convDescriptor.m_BiasEnabled)
            {
                ARMNN_ASSERT_MSG(convLayer->m_Bias != nullptr,
                                 "FuseBatchNorm: Bias data should not be null if bias is enabled.");

                ConstTensor biasTensor(convLayer->m_Bias->GetTensorInfo(), convLayer->m_Bias->Map(true));
                const auto* biasBuffer = static_cast<const T*>(biasTensor.GetMemoryArea());
                std::vector<T> biasVector(biasBuffer, biasBuffer + biasTensor.GetNumElements());

                for (unsigned int cOut = 0; cOut < outputChannels; ++cOut)
                {
                    fusedBiasVector[cOut] = ((gammaVector[cOut] * (biasVector[cOut] - meanVector[cOut])) /
                                             sqrtf(varianceVector[cOut] + epsilon)) + betaVector[cOut];
                }
            }
            else
            {
                convDescriptor.m_BiasEnabled = true;
                std::vector<T> biasVector(outputChannels, T(0));

                for (unsigned int cOut = 0; cOut < outputChannels; ++cOut)
                {
                    fusedBiasVector[cOut] = ((gammaVector[cOut] * (biasVector[cOut] - meanVector[cOut])) /
                                             sqrtf(varianceVector[cOut] + epsilon)) + betaVector[cOut];
                }
            }
            ConstTensor fusedBiasTensor(TensorInfo({outputChannels}, ArmnnType, 0.0f, 0, true), fusedBiasVector);

            // Insert the new convolution layer that has batch norm parameters fused into
            const std::string name = std::string("fused-") + child.GetName() + std::string("-into-") + base.GetName();
            auto& newConv2dLayer = *graph.InsertNewLayer<ConvLayer>(base.GetInputSlot(0),
                                                                    convDescriptor,
                                                                    name.c_str());
            newConv2dLayer.m_Weight = std::make_unique<ScopedTensorHandle>(fusedWeightsTensor);
            newConv2dLayer.m_Bias = std::make_unique<ScopedTensorHandle>(ConstTensor(fusedBiasTensor));

            // Reconnects with original parent.
            newConv2dLayer.GetOutputSlot().MoveAllConnections(*parentOut);
            // Parent is now the new convolution2d layer.
            parentOut = &newConv2dLayer.GetOutputSlot();

            // Moves connections in child output to parent layer.
            // Child layer will be removed as it's left unconnected.
            // Base layer will be removed if left unconnected.
            child.GetOutputSlot().MoveAllConnections(*parentOut);
        }
    }
protected:
    FuseBatchNorm()  = default;
    ~FuseBatchNorm() = default;
};

using FuseBatchNormIntoConvolution2DFloat32 =
        OptimizeForExclusiveConnection<Convolution2dLayer,
                                       BatchNormalizationLayer,
                                       FuseBatchNorm<Convolution2dLayer, armnn::DataType::Float32>>;

using FuseBatchNormIntoConvolution2DFloat16 =
        OptimizeForExclusiveConnection<Convolution2dLayer,
                                       BatchNormalizationLayer,
                                       FuseBatchNorm<Convolution2dLayer, armnn::DataType::Float16>>;

using FuseBatchNormIntoDepthwiseConvolution2DFloat32 =
        OptimizeForExclusiveConnection<DepthwiseConvolution2dLayer,
                                       BatchNormalizationLayer,
                                       FuseBatchNorm<DepthwiseConvolution2dLayer, armnn::DataType::Float32>>;

using FuseBatchNormIntoDepthwiseConvolution2DFloat16 =
        OptimizeForExclusiveConnection<DepthwiseConvolution2dLayer,
                                       BatchNormalizationLayer,
                                       FuseBatchNorm<DepthwiseConvolution2dLayer, armnn::DataType::Float16>>;

} // namespace optimizations
} // namespace armnn