//
// Copyright © 2022 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <optimizations/FoldPadIntoLayer2d.hpp>

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

}

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
LayerType* FoldPadLayer(OptimizationViews& optimizationViews,
                        LayerType* baseLayer,
                        LayerType* replacementLayer,
                        PadLayer* padLayer)
{
    SubgraphView substitutionSubgraph({padLayer, baseLayer},
                                      CreateIInputsFrom({padLayer}),
                                      CreateIOutputsFrom({baseLayer}));
    SubgraphView replacementSubgraph(replacementLayer);

    optimizationViews.AddSubstitution({substitutionSubgraph, replacementSubgraph});

    return replacementLayer;
}

template<typename LayerType>
LayerType* FoldPadIntoAveragePool2d(OptimizationViews& optimizationViews,
                                    Pooling2dLayer* baseLayer,
                                    Pooling2dDescriptor& poolDescriptor,
                                    PadLayer* padLayer)
{
        IConnectableLayer* replacement =
            optimizationViews.GetINetwork()->AddPooling2dLayer(poolDescriptor, "folded-pad-into-pool2d");
        LayerType* replacementLayer = PolymorphicDowncast<LayerType*>(replacement);

        FoldPadLayer(optimizationViews,
                     baseLayer,
                     replacementLayer,
                     padLayer);

        return replacementLayer;
}

} // namespace armnn
