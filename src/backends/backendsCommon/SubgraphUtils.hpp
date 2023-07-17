//
// Copyright © 2022-2023 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <armnn/StrategyBase.hpp>
#include <armnn/Descriptors.hpp>
#include <optimizations/FoldPadIntoLayer2d.hpp>

namespace armnn
{

namespace
{

/// Checks if a Layer has a DataLayout that is either NCHW or NCDHW.
class CheckForNCHW : public StrategyBase<NoThrowStrategy>
{
public:
    CheckForNCHW()
    {}

    void ExecuteStrategy(const armnn::IConnectableLayer* layer,
                         const armnn::BaseDescriptor& descriptor,
                         const std::vector<armnn::ConstTensor>& constants,
                         const char* name,
                         const armnn::LayerBindingId id = 0) override
    {
        armnn::IgnoreUnused(layer, constants, id, name);
        switch (layer->GetType())
        {
            case armnn::LayerType::BatchMatMul:
            {
                auto desc = static_cast<const armnn::BatchMatMulDescriptor &>(descriptor);
                m_Result = desc.m_DataLayoutX == DataLayout::NCHW || desc.m_DataLayoutY == DataLayout::NCHW;
                break;
            }
            case armnn::LayerType::BatchNormalization:
            {
                CheckDescForNCHW(static_cast<const armnn::BatchNormalizationDescriptor&>(descriptor));
                break;
            }
            case armnn::LayerType::BatchToSpaceNd:
            {
                CheckDescForNCHW(static_cast<const armnn::BatchToSpaceNdDescriptor&>(descriptor));
                break;
            }
            case armnn::LayerType::Convolution2d:
            {
                CheckDescForNCHW(static_cast<const armnn::Convolution2dDescriptor&>(descriptor));
                break;
            }
            case armnn::LayerType::Convolution3d:
            {
                CheckDescForNCHW(static_cast<const armnn::Convolution3dDescriptor&>(descriptor));
                break;
            }
            case armnn::LayerType::DepthwiseConvolution2d:
            {
                CheckDescForNCHW(static_cast<const armnn::DepthwiseConvolution2dDescriptor&>(descriptor));
                break;
            }
            case armnn::LayerType::InstanceNormalization:
            {
                CheckDescForNCHW(static_cast<const armnn::InstanceNormalizationDescriptor&>(descriptor));
                break;
            }
            case armnn::LayerType::L2Normalization:
            {
                CheckDescForNCHW(static_cast<const armnn::L2NormalizationDescriptor&>(descriptor));
                break;
            }
            case armnn::LayerType::Normalization:
            {
                CheckDescForNCHW(static_cast<const armnn::NormalizationDescriptor&>(descriptor));
                break;
            }
            case armnn::LayerType::Pooling2d:
            {
                CheckDescForNCHW(static_cast<const armnn::Pooling2dDescriptor&>(descriptor));
                break;
            }
            case armnn::LayerType::Pooling3d:
            {
                CheckDescForNCHW(static_cast<const armnn::Pooling3dDescriptor&>(descriptor));
                break;
            }
            case armnn::LayerType::SpaceToBatchNd:
            {
                CheckDescForNCHW(static_cast<const armnn::SpaceToBatchNdDescriptor&>(descriptor));
                break;
            }
            case armnn::LayerType::SpaceToDepth:
            {
                CheckDescForNCHW(static_cast<const armnn::SpaceToDepthDescriptor&>(descriptor));
                break;
            }
            case armnn::LayerType::StridedSlice:
            {
                CheckDescForNCHW(static_cast<const armnn::StridedSliceDescriptor&>(descriptor));
                break;
            }
            default:
            {
                m_Result = false;
            }
        }
    }

    /// Returns true if the Layer had a DataLayout and it was NCHW or NCDHW.
    /// Returns false if the Layer either doesn't have a DataLayout or if it
    /// had a DataLayout that was neither NCHW nor NCDHW.
    bool Result()
    {
        return m_Result;
    }

private:
    template<typename Descriptor>
    void CheckDescForNCHW(const Descriptor& descriptor)
    {
        m_Result = (descriptor.m_DataLayout == DataLayout::NCHW) || (descriptor.m_DataLayout == DataLayout::NCDHW);
    }

    bool m_Result = false;
};

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

inline bool IsNCHW(armnn::Layer& layer)
{
    CheckForNCHW check;
    layer.ExecuteStrategy(check);
    return check.Result();
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

/// Checks if the Layer is connected to any Layer that has an NCHW layout.
inline bool ConnectedToLayerWithNCHW(Layer* baseLayer)
{
    Layer& parentLayer = baseLayer->GetInputSlot(0).GetConnectedOutputSlot()->GetOwningLayer();

    if (IsNCHW(parentLayer))
    {
        return true;
    }
    for (unsigned int i = 0; i < baseLayer->GetOutputSlot(0).GetNumConnections(); ++i)
    {
        Layer& nextLayer = baseLayer->GetOutputSlot(0).GetConnection(i)->GetOwningLayer();
        if (IsNCHW(nextLayer))
        {
            return true;
        }
    }
    return false;
}

/// Checks if the Layer is connected to a Splitter Layer through a Tensor that has more than 4 dimensions.
inline bool ConnectedToSplitterWithMoreThan4Dims(Layer* baseLayer)
{
    Layer& parentLayer = baseLayer->GetInputSlot(0).GetConnectedOutputSlot()->GetOwningLayer();
    TensorInfo parentTensorInfo = baseLayer->GetInputSlot(0).GetConnectedOutputSlot()->GetTensorInfo();
    if (parentTensorInfo.GetNumDimensions() > 4 && parentLayer.GetType() == LayerType::Splitter)
    {
        return true;
    }
    for (unsigned int i = 0; i < baseLayer->GetOutputSlot(0).GetNumConnections(); ++i)
    {
        Layer& nextLayer = baseLayer->GetOutputSlot(0).GetConnection(i)->GetOwningLayer();
        TensorInfo nextTensorInfo = baseLayer->GetOutputSlot(0).GetConnection(i)->GetTensorInfo();
        if (nextTensorInfo.GetNumDimensions() > 4 && nextLayer.GetType() == LayerType::Splitter)
        {
            return true;
        }
    }
    return false;
}

inline void RemoveReshapeLayer(ReshapeLayer* baseLayer,
                               std::map<LayerGuid, Layer*>& untouched,
                               OptimizationViews& optimizationViews)
{
    if (baseLayer == nullptr)
    {
        return;
    }
    ReshapeDescriptor reshapeDescriptor = baseLayer->GetParameters();
    Layer& parentLayer = baseLayer->GetInputSlot(0).GetConnectedOutputSlot()->GetOwningLayer();

    // Cannot currently remove the Reshape if it's connected to an Input, Constant or Splitter
    if (parentLayer.GetType() == LayerType::Input || parentLayer.GetType() == LayerType::Constant)
    {
        return;
    }

    // Cannot currently remove the Reshape if it's connected to an OutputSlot or Concat
    for (unsigned int i = 0; i < baseLayer->GetOutputSlot(0).GetNumConnections(); ++i)
    {
        Layer& nextLayer = baseLayer->GetOutputSlot(0).GetConnection(i)->GetOwningLayer();

        if (nextLayer.GetType() == LayerType::Output)
        {
            return;
        }
    }
    auto it = untouched.find(baseLayer->GetGuid());
    if (it == untouched.end())
    {
        // Already removed from map
        return;
    }
    untouched.erase(it);

    // Override the InputSlot TensorInfos for all the layers connected to the Reshape's OutputSlot
    for (unsigned int i = 0; i < baseLayer->GetOutputSlot(0).GetNumConnections(); ++i)
    {
        Layer& nextLayer = baseLayer->GetOutputSlot(0).GetConnection(i)->GetOwningLayer();
        auto inputIndex = baseLayer->GetOutputSlot(0).GetConnection(i)->GetSlotIndex();
        TensorInfo reshapeInfo(baseLayer->GetOutputSlot(0).GetTensorInfo());
        reshapeInfo.SetShape(reshapeDescriptor.m_TargetShape);
        nextLayer.GetInputSlot(inputIndex).SetTensorInfo(reshapeInfo);
    }
    optimizationViews.AddDeletedSubgraph(baseLayer);
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
