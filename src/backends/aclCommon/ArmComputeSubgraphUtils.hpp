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

//
// this helper only works if all layers where the outputs connect to are not selected
//
SubgraphView::OutputSlots CreateOutputsFrom(const std::vector<Layer*>& layers)
{
    SubgraphView::OutputSlots result;
    for (auto&& layer : layers)
    {
        for (auto&& it = layer->BeginOutputSlots(); it != layer->EndOutputSlots(); ++it)
        {
            result.push_back(&(*it));
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
        SubgraphView subgraphView(CreateInputsFrom({layer}),
                                  CreateOutputsFrom({layer}),
                                  {layer});
        optimizationViews.AddUntouchedSubgraph(std::move(subgraphView));
    }
}

template<typename LayerType>
LayerType* FuseLayerWithoutParameters(OptimizationViews& optimizationViews,
                                      LayerType* baseLayer,
                                      ActivationLayer* activationLayer,
                                      ActivationDescriptor& activationDesc,
                                      std::string name)
{
    LayerType* replacementLayer = optimizationViews.GetGraph().AddLayer<LayerType>(name.c_str());

    replacementLayer->SetAdditionalInfoForObject(std::make_shared<ActivationDescriptor>(activationDesc));

    SubgraphView substitutionSubgraph(CreateInputsFrom({baseLayer}),
                                      CreateOutputsFrom({activationLayer}),
                                      {baseLayer, activationLayer});
    SubgraphView replacementSubgraph(replacementLayer);

    optimizationViews.AddSubstitution({substitutionSubgraph, replacementSubgraph});
    return replacementLayer;
}

template<typename LayerType>
LayerType* FuseLayerWithParameters(OptimizationViews& optimizationViews,
                                   LayerType* baseLayer,
                                   ActivationLayer* activationLayer,
                                   ActivationDescriptor& activationDesc,
                                   std::string name)
{
    LayerType* replacementLayer = optimizationViews.GetGraph().AddLayer<LayerType>(baseLayer->GetParameters(),
                                                                                   name.c_str());

    replacementLayer->SetAdditionalInfoForObject(std::make_shared<ActivationDescriptor>(activationDesc));

    SubgraphView substitutionSubgraph(CreateInputsFrom({baseLayer}),
                                      CreateOutputsFrom({activationLayer}),
                                      {baseLayer, activationLayer});
    SubgraphView replacementSubgraph(replacementLayer);

    optimizationViews.AddSubstitution({substitutionSubgraph, replacementSubgraph});
    return replacementLayer;
}

template<typename LayerType>
LayerType* FuseLayerWithWeightsAndBiases(OptimizationViews& optimizationViews,
                                         LayerType* baseLayer,
                                         ActivationLayer* activationLayer,
                                         ActivationDescriptor& activationDesc,
                                         std::string name)
{
    LayerType* replacementLayer = FuseLayerWithParameters(optimizationViews,
                                                          baseLayer,
                                                          activationLayer,
                                                          activationDesc,
                                                          name);

    replacementLayer->m_Weight = std::move(baseLayer->m_Weight);
    replacementLayer->m_Bias   = std::move(baseLayer->m_Bias);

    return replacementLayer;
}

//
// If reduce layer has multiple axes, add new layer for each axis to simulate the same behaviour
// as currently only one axis is supported.
//
template<typename LayerType>
std::vector<Layer*> ChainReduceLayers(OptimizationViews& optimizationViews,
                                      LayerType* baseLayer,
                                      ReduceDescriptor& desc)
{
    // Vector of new chained layers, used for substitution.
    std::vector<Layer*> layers;

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
        Layer* replacementLayer = optimizationViews.GetGraph().AddLayer<LayerType>(newReduceDescriptor,
                                                                                   layerName.c_str());
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
    ARMNN_ASSERT(baseLayer->GetOutputSlot(0).GetTensorInfo() == layers.back()->GetOutputSlot().GetTensorInfo());

    return layers;
}

//
// Substitute baseLayer with new subgraph
//
template<typename LayerType>
void ReplaceLayers(OptimizationViews& optimizationViews,
                   LayerType* baseLayer,
                   std::vector<Layer*>& layers)
{
    std::list<Layer*> replacementLayers(layers.begin(), layers.end());

    SubgraphView substitutionSubgraph(baseLayer);
    SubgraphView replacementSubgraph(CreateInputsFrom({replacementLayers.front()}),
                                     CreateOutputsFrom({replacementLayers.back()}),
                                     std::move(replacementLayers));

    optimizationViews.AddSubstitution({substitutionSubgraph, replacementSubgraph});
}

} // namespace armnn
