//
// Copyright © 2020 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <armnn/backends/OptimizationViews.hpp>
#include <armnn/utility/Assert.hpp>

#include <aclCommon/ArmComputeUtils.hpp>

namespace armnn
{

namespace
{

//
// this helper only works if all layers where the inputs connect to are not selected
//

SubgraphView::IInputSlots CreateIInputsFrom(const std::vector<armnn::IConnectableLayer*>& layers)
{
    SubgraphView::IInputSlots result;
    for (auto&& layer : layers)
    {
        for (unsigned int i = 0 ; i < layer->GetNumInputSlots(); ++i)
        {
            result.push_back(&(layer->GetInputSlot(i)));
        }
    }
    return result;
}

//
// this helper only works if all layers where the outputs connect to are not selected
//

SubgraphView::IOutputSlots CreateIOutputsFrom(const std::vector<armnn::IConnectableLayer*>& layers)
{
    SubgraphView::IOutputSlots result;
    for (auto &&layer: layers)
    {
        for (unsigned int i = 0; i < layer->GetNumOutputSlots(); ++i)
        {
            result.push_back(&(layer->GetOutputSlot(i)));
        }
    }
    return result;
}

bool checkDataTypeInputandOutput(const Layer& layer)
{
    auto inputInfo = layer.GetInputSlot(0).GetConnection()->GetTensorInfo();
    auto outputInfo = layer.GetOutputSlot(0).GetTensorInfo();
    bool sameDataType = (inputInfo.GetDataType() == outputInfo.GetDataType());

    // Check is same quantization info (same scale and offset)
    if (sameDataType)
    {
        if (IsQuantizedType(inputInfo.GetDataType()))
        {
            bool sameScale = (inputInfo.GetQuantizationScale() == outputInfo.GetQuantizationScale());
            bool sameOffset = (inputInfo.GetQuantizationOffset() == outputInfo.GetQuantizationOffset());

            return (sameScale && sameOffset);
        }
        else
        {
            return true;
        }
    }
    else
    {
        return false;
    }
}

} // namespace

inline void ReportUntouchedLayers(OptimizationViews& optimizationViews, std::map<LayerGuid, Layer*> untouched)
{
    std::vector<Layer*> untouchedVector;
    for (const auto& pair : untouched)
    {
        Layer* layer = pair.second;
        SubgraphView subgraphView({layer},
                                  CreateIInputsFrom({layer}),
                                  CreateIOutputsFrom({layer}));
        optimizationViews.AddUntouchedSubgraph(std::move(subgraphView));
    }
}

template<typename LayerType>
LayerType* FuseLayer(OptimizationViews& optimizationViews,
                     LayerType* baseLayer,
                     LayerType* replacementLayer,
                     ActivationLayer* activationLayer,
                     ActivationDescriptor& activationDesc)
{
    replacementLayer->SetAdditionalInfoForObject(
        std::make_shared<ActivationDescriptor>(activationDesc));

    SubgraphView substitutionSubgraph({baseLayer, activationLayer},
                                      CreateIInputsFrom({baseLayer}),
                                      CreateIOutputsFrom({activationLayer}));
    SubgraphView replacementSubgraph(replacementLayer);

    optimizationViews.AddSubstitution({substitutionSubgraph, replacementSubgraph});

    return replacementLayer;
}

template<typename LayerType>
LayerType* FuseAdditionLayer(OptimizationViews& optimizationViews,
                             LayerType* baseLayer,
                             ActivationLayer* activationLayer,
                             ActivationDescriptor& activationDesc,
                             std::string name)
{
    IConnectableLayer* replacement = optimizationViews.GetINetwork()->AddAdditionLayer(name.c_str());
    LayerType* replacementLayer = PolymorphicDowncast<LayerType*>(replacement);

    FuseLayer(optimizationViews,
              baseLayer,
              replacementLayer,
              activationLayer,
              activationDesc);

    return replacementLayer;
}

template<typename LayerType>
LayerType* FuseSubtractionLayer(OptimizationViews& optimizationViews,
                                LayerType* baseLayer,
                                ActivationLayer* activationLayer,
                                ActivationDescriptor& activationDesc,
                                std::string name)
{
    IConnectableLayer* replacement = optimizationViews.GetINetwork()->AddSubtractionLayer(name.c_str());
    LayerType* replacementLayer = PolymorphicDowncast<LayerType*>(replacement);

    FuseLayer(optimizationViews,
              baseLayer,
              replacementLayer,
              activationLayer,
              activationDesc);

    return replacementLayer;
}

template<typename LayerType>
LayerType* FuseDivisionLayer(OptimizationViews& optimizationViews,
                             LayerType* baseLayer,
                             ActivationLayer* activationLayer,
                             ActivationDescriptor& activationDesc,
                             std::string name)
{
    IConnectableLayer* replacement = optimizationViews.GetINetwork()->AddDivisionLayer(name.c_str());
    LayerType* replacementLayer = PolymorphicDowncast<LayerType*>(replacement);

    FuseLayer(optimizationViews,
              baseLayer,
              replacementLayer,
              activationLayer,
              activationDesc);

    return replacementLayer;
}

template<typename LayerType>
LayerType* FuseMultiplicationLayer(OptimizationViews& optimizationViews,
                                   LayerType* baseLayer,
                                   ActivationLayer* activationLayer,
                                   ActivationDescriptor& activationDesc,
                                   std::string name)
{
    IConnectableLayer* replacement = optimizationViews.GetINetwork()->AddMultiplicationLayer(name.c_str());
    LayerType* replacementLayer = PolymorphicDowncast<LayerType*>(replacement);

    FuseLayer(optimizationViews,
              baseLayer,
              replacementLayer,
              activationLayer,
              activationDesc);

    return replacementLayer;
}

template<typename LayerType>
LayerType* FuseBatchNormalizationLayer(OptimizationViews& optimizationViews,
                                       LayerType* baseLayer,
                                       ActivationLayer* activationLayer,
                                       ActivationDescriptor& activationDesc,
                                       std::string name)
{
    IConnectableLayer* replacement =
        optimizationViews.GetINetwork()->AddBatchNormalizationLayer(baseLayer->GetParameters(),
                                                                    ConstTensor(),
                                                                    ConstTensor(),
                                                                    ConstTensor(),
                                                                    ConstTensor(),
                                                                    name.c_str());
    LayerType* replacementLayer = PolymorphicDowncast<LayerType*>(replacement);

    FuseLayer(optimizationViews,
              baseLayer,
              replacementLayer,
              activationLayer,
              activationDesc);

    SubgraphView substitutionSubgraph({baseLayer, activationLayer},
                                      CreateIInputsFrom({baseLayer}),
                                      CreateIOutputsFrom({activationLayer}));
    SubgraphView replacementSubgraph(replacementLayer);

    return replacementLayer;
}

template<typename LayerType>
LayerType* FuseConvolution2dLayer(OptimizationViews& optimizationViews,
                                  LayerType* baseLayer,
                                  ActivationLayer* activationLayer,
                                  ActivationDescriptor& activationDesc,
                                  std::string name)
{
    std::shared_ptr<ConstTensorHandle> weightHandle = baseLayer->m_Weight;
    TensorInfo weightInfo = weightHandle->GetTensorInfo();

    std::shared_ptr<ConstTensorHandle> biasHandle = baseLayer->m_Bias;
    ConstTensor biasTensor;
    if (!biasHandle)
    {
        biasTensor = ConstTensor();
    }
    else
    {
        biasTensor = ConstTensor(biasHandle->GetTensorInfo(), biasHandle->Map(true));
    }

    IConnectableLayer* replacement =
        optimizationViews.GetINetwork()->
            AddConvolution2dLayer(baseLayer->GetParameters(),
                                  ConstTensor(weightInfo, weightHandle->Map(true)),
                                  Optional<ConstTensor>(biasTensor),
                                  name.c_str());
    LayerType* replacementLayer = PolymorphicDowncast<LayerType*>(replacement);

    FuseLayer(optimizationViews,
              baseLayer,
              replacementLayer,
              activationLayer,
              activationDesc);

    return replacementLayer;
}

template<typename LayerType>
LayerType* FuseDepthwiseConvolution2dLayer(OptimizationViews& optimizationViews,
                                           LayerType* baseLayer,
                                           ActivationLayer* activationLayer,
                                           ActivationDescriptor& activationDesc,
                                           std::string name)
{
    std::shared_ptr<ConstTensorHandle> weightHandle = baseLayer->m_Weight;
    TensorInfo weightInfo = weightHandle->GetTensorInfo();

    std::shared_ptr<ConstTensorHandle> biasHandle = baseLayer->m_Bias;
    ConstTensor biasTensor;
    if (!biasHandle)
    {
        biasTensor = ConstTensor();
    }
    else
    {
        biasTensor = ConstTensor(biasHandle->GetTensorInfo(), biasHandle->Map(true));
    }

    IConnectableLayer* replacement =
        optimizationViews.GetINetwork()->
            AddDepthwiseConvolution2dLayer(baseLayer->GetParameters(),
                                           ConstTensor(weightInfo, weightHandle->Map(true)),
                                           Optional<ConstTensor>(biasTensor),
                                           name.c_str());
    LayerType* replacementLayer = PolymorphicDowncast<LayerType*>(replacement);

    FuseLayer(optimizationViews,
              baseLayer,
              replacementLayer,
              activationLayer,
              activationDesc);

    return replacementLayer;
}

template<typename LayerType>
LayerType* FuseFullyConnectedLayer(OptimizationViews& optimizationViews,
                                   LayerType* baseLayer,
                                   ActivationLayer* activationLayer,
                                   ActivationDescriptor& activationDesc,
                                   std::string name)
{
    IConnectableLayer* replacement =
        optimizationViews.GetINetwork()->AddFullyConnectedLayer(baseLayer->GetParameters(),
                                                                name.c_str());
    LayerType* replacementLayer = PolymorphicDowncast<LayerType*>(replacement);

    FuseLayer(optimizationViews,
              baseLayer,
              replacementLayer,
              activationLayer,
              activationDesc);

    replacementLayer->m_Weight = std::move(baseLayer->m_Weight);
    replacementLayer->m_Bias   = std::move(baseLayer->m_Bias);

    return replacementLayer;
}

//
// If reduce layer has multiple axes, add new layer for each axis to simulate the same behaviour
// as currently only one axis is supported.
//
template<typename LayerType>
std::vector<IConnectableLayer*> ChainReduceLayers(OptimizationViews& optimizationViews,
                                      LayerType* baseLayer,
                                      ReduceDescriptor& desc)
{
    // Vector of new chained layers, used for substitution.
    std::vector<IConnectableLayer*> layers;

    // Vector of axes so each layer is reshaped correctly.
    std::vector<uint32_t> axes;
    unsigned int recalulatedAxis = 0;

    for (unsigned int i = 0; i != desc.m_vAxis.size(); ++i)
    {
        // Get TensorInfo from base layer and reduce shape using axis.
        TensorInfo layerInfo = baseLayer->GetInputSlot(0).GetConnectedOutputSlot()->GetTensorInfo();

        axes.emplace_back(desc.m_vAxis[i]);

        const TensorInfo& reducedTensorInfo = ComputeReductionTensorShape(layerInfo,
                                                                          axes,
                                                                          desc.m_KeepDims);

        // Create a vector for the single axis to be assigned to the descriptor.
        // Update axis if keepDims is set reduce layers correctly.
        std::vector<uint32_t> singleAxis(1, desc.m_vAxis[i] - recalulatedAxis);

        // Create a descriptor and assign single axis.
        ReduceDescriptor newReduceDescriptor = baseLayer->GetParameters();
        newReduceDescriptor.m_vAxis.assign(singleAxis.begin(), singleAxis.end());

        // Add new layer to graph.
        std::string layerName = "reduce_layer_" + std::to_string(i);

        Layer* replacementLayer = PolymorphicDowncast<Layer*>(
            optimizationViews.GetINetwork()->AddReduceLayer(newReduceDescriptor,
                                                            layerName.c_str()));

        // Connect previous layer with new layer.
        // The first and last layer will be connected when the subgraph is replaced.
        if (!layers.empty())
        {
            layers[i - 1]->GetOutputSlot(0).Connect(replacementLayer->GetInputSlot(0));
        }

        // Set updated tensorInfo for new layer.
        replacementLayer->GetOutputSlot(0).SetTensorInfo(reducedTensorInfo);

        if (!desc.m_KeepDims)
        {
            recalulatedAxis++;
        }

        layers.emplace_back(replacementLayer);
    }

    // Check if the TensorInfo from the last layer equals the inferred output from the original layer.
    ARMNN_ASSERT(baseLayer->GetOutputSlot(0).GetTensorInfo() ==
                 PolymorphicDowncast<Layer*>(layers.back())->GetOutputSlot().GetTensorInfo());

    return layers;
}

//
// Substitute baseLayer with new subgraph
//
template<typename LayerType>
void ReplaceLayers(OptimizationViews& optimizationViews,
                   LayerType* baseLayer,
                   std::vector<IConnectableLayer*>& layers)
{
    std::list<IConnectableLayer*> replacementLayers(layers.begin(), layers.end());

    SubgraphView substitutionSubgraph(baseLayer);
    SubgraphView replacementSubgraph(std::move(replacementLayers),
                                     CreateIInputsFrom({replacementLayers.front()}),
                                     CreateIOutputsFrom({replacementLayers.back()}));

    optimizationViews.AddSubstitution({substitutionSubgraph, replacementSubgraph});
}

} // namespace armnn
