//
// Copyright © 2020 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <armnn/backends/OptimizationViews.hpp>

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

} // namespace

inline const TensorInfo GetOverriddenDataType(const TensorInfo& info, Optional<DataType> type)
{
    if (!type)
    {
        return info;
    }

    return TensorInfo(info.GetShape(), type.value(), info.GetQuantizationScale(), info.GetQuantizationOffset());
}

inline armnn::Optional<armnn::DataType> GetOptionalBiasTypeFromWeightsType(armnn::Optional<armnn::DataType> weightsType)
{
    if (!weightsType)
    {
        return weightsType;
    }

    switch(weightsType.value())
    {
        case armnn::DataType::BFloat16:
        case armnn::DataType::Float16:
        case armnn::DataType::Float32:
            return weightsType;
        case armnn::DataType::QAsymmS8:
        case armnn::DataType::QAsymmU8:
        case armnn::DataType::QSymmS8:
        case armnn::DataType::QSymmS16:
            return armnn::DataType::Signed32;
        default:
            ARMNN_ASSERT_MSG(false, "GetBiasTypeFromWeightsType(): Unsupported data type.");
    }
    return armnn::EmptyOptional();
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

} // namespace armnn
