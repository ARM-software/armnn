//
// Copyright © 2020 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include "Optimization.hpp"
#include <armnnUtils/DataLayoutIndexed.hpp>

namespace armnn
{
namespace optimizations
{

template <typename ConvLayer>
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

        ARMNN_ASSERT(base.GetType() == LayerType::Convolution2d);
        ARMNN_ASSERT(child.GetType() == LayerType::BatchNormalization);

        if (base.GetDataType() == DataType::Float32 && child.GetDataType() == DataType::Float32)
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
            ConstTensor weightsTensor(convLayer->m_Weight->GetTensorInfo(), convLayer->m_Weight->Map(true));

            armnnUtils::DataLayoutIndexed dataLayout(convDescriptor.m_DataLayout);
            auto weightsShape = convLayer->m_Weight->GetTensorInfo().GetShape();
            const unsigned int outputChannels  = weightsShape[0];
            const unsigned int inputChannels   = weightsShape[dataLayout.GetChannelsIndex()];
            const unsigned int weightsHeight   = weightsShape[dataLayout.GetHeightIndex()];
            const unsigned int weightsWidth    = weightsShape[dataLayout.GetWidthIndex()];

            const auto* weightsBuffer = static_cast<const float*>(weightsTensor.GetMemoryArea());
            const auto* betaBuffer    = static_cast<const float*>(betaTensor.GetMemoryArea());
            const auto* gammaBuffer   = static_cast<const float*>(gammaTensor.GetMemoryArea());
            const auto* meanBuffer    = static_cast<const float*>(meanTensor.GetMemoryArea());
            const auto* varBuffer     = static_cast<const float*>(varTensor.GetMemoryArea());

            std::vector<float> weightsVector (weightsBuffer, weightsBuffer + weightsTensor.GetNumElements());
            std::vector<float> betaVector    (betaBuffer, betaBuffer + betaTensor.GetNumElements());
            std::vector<float> gammaVector   (gammaBuffer, gammaBuffer + gammaTensor.GetNumElements());
            std::vector<float> meanVector    (meanBuffer, meanBuffer + meanTensor.GetNumElements());
            std::vector<float> varianceVector(varBuffer, varBuffer + varTensor.GetNumElements());

            // fusedWeights = ( gamma * weights ) / ( std - epsilon);
            std::vector<float> fusedWeightsVector(weightsVector.size());

            unsigned int i = 0;
            for (unsigned int cOut = 0; cOut < outputChannels; ++cOut)
            {
                auto mult = gammaVector[cOut] / sqrtf (varianceVector[cOut] + epsilon);
                for (unsigned int cInput = 0; cInput < inputChannels; ++cInput)
                {
                    for (unsigned int h = 0; h < weightsHeight; ++h)
                    {
                        for (unsigned int w = 0; w < weightsWidth; ++w)
                        {
                            fusedWeightsVector[i] = mult * weightsVector[i];
                            i++;
                        }
                    }
                }
            }
            ConstTensor fusedWeightsTensor(convLayer->m_Weight->GetTensorInfo(), fusedWeightsVector);

            //  fusedBias = (gamma * (bias - mean)) / (variance - epsilon) + beta;
            std::vector<float> fusedBiasVector(outputChannels);
            if (convDescriptor.m_BiasEnabled)
            {
                ARMNN_ASSERT_MSG(convLayer->m_Bias != nullptr,
                                 "FuseBatchNorm: Bias data should not be null if bias is enabled.");

                ConstTensor biasTensor(convLayer->m_Bias->GetTensorInfo(), convLayer->m_Bias->Map(true));
                const auto* biasBuffer = static_cast<const float*>(biasTensor.GetMemoryArea());
                std::vector<float> biasVector(biasBuffer, biasBuffer + biasTensor.GetNumElements());

                for (unsigned int cOut = 0; cOut < outputChannels; ++cOut)
                {
                    fusedBiasVector[cOut] = ((gammaVector[cOut] * (biasVector[cOut] - meanVector[cOut])) /
                                             sqrtf(varianceVector[cOut] + epsilon)) + betaVector[cOut];
                }
            }
            else
            {
                convDescriptor.m_BiasEnabled = true;
                std::vector<float> biasVector(outputChannels, 0);

                for (unsigned int cOut = 0; cOut < outputChannels; ++cOut)
                {
                    fusedBiasVector[cOut] = ((gammaVector[cOut] * (biasVector[cOut] - meanVector[cOut])) /
                                             sqrtf(varianceVector[cOut] + epsilon)) + betaVector[cOut];
                }
            }
            ConstTensor fusedBiasTensor(TensorInfo({outputChannels}, DataType::Float32), fusedBiasVector);

            // Insert the new convolution layer that has batch norm parameters fused into
            const std::string name = std::string("fused-") + child.GetName() + std::string("-into-") + base.GetName();
            auto& newConv2dLayer = *graph.InsertNewLayer<ConvLayer>(base.GetInputSlot(0),
                                                                    convDescriptor,
                                                                    name.c_str());
            newConv2dLayer.m_Weight = std::make_unique<ScopedCpuTensorHandle>(fusedWeightsTensor);
            newConv2dLayer.m_Bias = std::make_unique<ScopedCpuTensorHandle>(ConstTensor(fusedBiasTensor));

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

using FuseBatchNormIntoConvolution2D          =
        OptimizeForExclusiveConnection<Convolution2dLayer,
                                       BatchNormalizationLayer,
                                       FuseBatchNorm<Convolution2dLayer>>;

} // namespace optimizations
} // namespace armnn