//
// Copyright © 2022-2025 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <armnn/StrategyBase.hpp>
#include <armnn/Descriptors.hpp>
#include <optimizations/FoldPadIntoLayer2d.hpp>
#include <type_traits>

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
                auto desc = static_cast<const armnn::BatchMatMulDescriptor&>(descriptor);
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

// Type used to hold the slot numbers to create the lists from. There should
// be a SlotList for each layer in the layers list
typedef std::vector<int> SlotList;

template<typename ILayerType>
SubgraphView::IInputSlots CreateIInputsFromSlotLists(const std::vector<ILayerType*>& layers,
                                                     const std::vector<SlotList>& layersSlotLists)
{
    ARMNN_THROW_INVALIDARG_IF_FALSE(layersSlotLists.size() == layers.size());

    SubgraphView::IInputSlots result;

    for (unsigned int layerIdx = 0; layerIdx < layers.size(); ++layerIdx)
    {
        const SlotList& slotList = layersSlotLists[layerIdx];
        for (unsigned int slotIdx = 0 ; slotIdx < layers[layerIdx]->GetNumInputSlots(); ++slotIdx)
        {
            if (std::find(slotList.begin(), slotList.end(), slotIdx) != slotList.end())
            {
                result.push_back(&(layers[layerIdx]->GetInputSlot(slotIdx)));
            }
        }
    }
    return result;
}

template<typename ILayerType>
SubgraphView::IOutputSlots CreateIOutputsFromSlotLists(const std::vector<ILayerType*>& layers,
                                                       const std::vector<SlotList>& layersSlotLists)
{
    ARMNN_THROW_INVALIDARG_IF_FALSE(layersSlotLists.size() == layers.size());

    SubgraphView::IOutputSlots result;
    for (unsigned int layerIdx = 0; layerIdx < layers.size(); ++layerIdx)
    {
        const SlotList& slotList = layersSlotLists[layerIdx];
        for (unsigned int slotIdx = 0; slotIdx < layers[layerIdx]->GetNumOutputSlots(); ++slotIdx)
        {
            bool foundIt = std::find(slotList.begin(), slotList.end(), slotIdx) != slotList.end();
            if (foundIt)
            {
                result.push_back(&(layers[layerIdx]->GetOutputSlot(slotIdx)));
            }
        }
    }
    return result;
}
}

namespace FoldPadConstraints
{
    // namespace for holding template constraints related to fold pad functions
    // and for static asserts to prevent function misuse
    template <class>
    inline constexpr bool alwaysFalse = false;

    template <typename L, typename D>
    struct IsValidPair : std::false_type {};

    // template specialization of IsValidPair for allowed pairings of layers and descriptors
    template <>
    struct IsValidPair<Pooling2dLayer, Pooling2dDescriptor> : std::true_type {};

    template <>
    struct IsValidPair<Convolution2dLayer, Convolution2dDescriptor> : std::true_type {};

    template <>
    struct IsValidPair<DepthwiseConvolution2dLayer, DepthwiseConvolution2dDescriptor> : std::true_type {};

} // namespace FoldPadConstraints

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
LayerType* ReplaceLayer(OptimizationViews& optimizationViews,
                        LayerType* baseLayer,
                        LayerType* replacementLayer)
{
    SubgraphView substitutionSubgraph({baseLayer},
                                      CreateIInputsFrom({baseLayer}),
                                      CreateIOutputsFrom({baseLayer}));

    SubgraphView replacementSubgraph({replacementLayer},
                                      CreateIInputsFrom({replacementLayer}),
                                      CreateIOutputsFrom({replacementLayer}));


    optimizationViews.AddSubstitution({substitutionSubgraph, replacementSubgraph});

    return replacementLayer;
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

/// Checks the Layer's Connections to see if it's connected to a Layer with the provided layerType. If dimSize is
/// provided will also check if the connecting Tensor has more than that number of dimensions
inline bool ConnectedToLayerType(Layer* baseLayer, LayerType layerType, unsigned int dimSize = 0)
{
    Layer& parentLayer = baseLayer->GetInputSlot(0).GetConnectedOutputSlot()->GetOwningLayer();
    TensorInfo parentTensorInfo = baseLayer->GetInputSlot(0).GetTensorInfo();

    if (parentTensorInfo.GetNumDimensions() > dimSize && parentLayer.GetType() == layerType)
    {
        return true;
    }
    for (unsigned int i = 0; i < baseLayer->GetOutputSlot(0).GetNumConnections(); ++i)
    {
        Layer& nextLayer = baseLayer->GetOutputSlot(0).GetConnection(i)->GetOwningLayer();
        TensorInfo nextTensorInfo = baseLayer->GetOutputSlot(0).GetConnection(i)->GetTensorInfo();

        if (nextTensorInfo.GetNumDimensions() > dimSize && nextLayer.GetType() == layerType)
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


template<typename LayerT, typename Descriptor>
void FoldPadLayer2d(OptimizationViews& optimizationViews,
                    LayerT* baseLayer,
                    Descriptor& descriptor,
                    PadLayer* padLayer)
{

    // Enforce that the function is called with a valid combination of layertype and descriptors
    static_assert(FoldPadConstraints::IsValidPair<LayerT, Descriptor>::value,
                  "FoldPadLayer2d() called with an unsupported (LayerType, Descriptor) combination!");

    IConnectableLayer* replacement = nullptr;
    const std::string name = std::string("folded-") + padLayer->GetName() + "-into-" + baseLayer->GetName();
    if constexpr (std::is_same_v<LayerT, Pooling2dLayer>)
    {
        replacement = optimizationViews.GetINetwork()->AddPooling2dLayer(descriptor, name.c_str());
        LayerT* replacementLayer = PolymorphicDowncast<LayerT*>(replacement);
        FoldPadLayer(optimizationViews,
                    baseLayer,
                    replacementLayer,
                    padLayer);
    }
    else if constexpr (std::is_same_v<LayerT, Convolution2dLayer> ||
                       std::is_same_v<LayerT, DepthwiseConvolution2dLayer>)
    {
        // DepthwiseConv2d and Conv2d pad fold is being done by creating a new layer and subsitituing 
        // the existing conv after updating the padding descriptor with TryFoldPadIntoLayer2d
        // We then mark the pad layer for deletion
        // this prevents a mismatch in the number of expected input slots on the optimized layer
        // i.e. pad has 1 input slot but conv2d has 3 (1 input and 2 constants which show as input slots)
        if constexpr (std::is_same_v<LayerT, Convolution2dLayer>)
        {
            replacement = optimizationViews.GetINetwork()->AddConvolution2dLayer(descriptor, name.c_str());
        }
        else
        {
            replacement = optimizationViews.GetINetwork()->AddDepthwiseConvolution2dLayer(descriptor, name.c_str());
        }
        LayerT* replacementLayer = PolymorphicDowncast<LayerT*>(replacement);
        SubgraphView layerToDelete(padLayer);
        optimizationViews.AddDeletedSubgraph(std::move(layerToDelete));
        ReplaceLayer(optimizationViews,
                     baseLayer,
                     replacementLayer);
    }
    else
    {
        static_assert(FoldPadConstraints::alwaysFalse<LayerT>, 
                     "FoldPadLayer2d() called with an unsupported LayerType");
    }
}

//
// Layer sequence detection such as add + mul + add ( + optional activation )
//

inline bool IsSequenceLayerType(Layer& layer, LayerType type)
{
    return layer.GetType() == type;
}

inline bool IsSequenceLayerType(Layer& layer, BinaryOperation type)
{
    return (layer.GetType() == LayerType::ElementwiseBinary) &&
            (PolymorphicDowncast<ElementwiseBinaryLayer*>(&layer)->GetParameters().m_Operation == type);
}

// Detect a layer sequence and activation if specified. The activation must be at the end of the sequence.
template<typename TYPE>
bool IsLayerSequence(Layer& currentLayer,
                     TYPE first,
                     TYPE second,
                     TYPE third,
                     Layer* layerList[4],
                     bool handleValidActivates,
                     const std::vector<ActivationFunction>& validActivates)
{
    auto PreviousLayer = [](Layer& layer)
    {
        return &layer.GetInputSlot(0).GetConnectedOutputSlot()->GetOwningLayer();
    };

    auto NextLayer = [](Layer& layer)
    {
        return &layer.GetOutputSlot(0).GetConnection(0)->GetOwningLayer();
    };

    auto LayerIncomingConnectionDataType = [](Layer& layer)
    {
        return layer.GetInputSlot(0).GetTensorInfo().GetDataType();
    };

    bool result = false;

    // Match in reverse so there is only 1 connection to check
    if (IsSequenceLayerType(currentLayer, third))
    {
        // Save DataType of third layer
        DataType dataType = LayerIncomingConnectionDataType(currentLayer);

        // Save third layer
        layerList[2] = &currentLayer;

        // Check the layers that proceed this one for the requested grouping
        Layer *prevLayer = PreviousLayer(currentLayer);
        if (prevLayer && IsSequenceLayerType(*prevLayer, second))
        {
            bool dataTypesMatch = (dataType == LayerIncomingConnectionDataType(*prevLayer));
            if (! dataTypesMatch)
            {
                return result;
            }

            layerList[1] = prevLayer;
            prevLayer = PreviousLayer(*prevLayer);
            if (prevLayer && IsSequenceLayerType(*prevLayer, first))
            {
                dataTypesMatch = (dataType == LayerIncomingConnectionDataType(*prevLayer));
                if (! dataTypesMatch)
                {
                    return result;
                }

                layerList[0] = prevLayer;

                // Detected the first 3 layers if we get to this point so now
                // check to see if we have a valid activation. If there is no activation
                // then the sequence still matches.
                if (handleValidActivates)
                {
                    Layer *nextLayer = NextLayer(currentLayer);
                    if (nextLayer)
                    {
                        if (IsSequenceLayerType(*nextLayer, LayerType::Activation))
                        {
                            // This layer is an activation, so it must be a valid type for the sequence
                            ActivationFunction activationFunction =
                                    PolymorphicDowncast<ActivationLayer*>(nextLayer)->GetParameters().m_Function;
                            long count = std::count(validActivates.cbegin(),
                                                    validActivates.cend(),
                                                    activationFunction);
                            if (count > 0)
                            {
                                layerList[3] = nextLayer;
                                result = true;
                            }
                        }
                        else
                        {
                            // Next layer is not an activation so sequence still matches
                            result = true;
                        }
                    }
                }
                else
                {
                    result = true;
                }
            }
        }
    }

    return result;
}

// OpBlockSequencer reorders blocks based on the availability of their input tensors.
// If all of a block’s input tensors are already known, the block is added to the list immediately;
// otherwise, it is queued until its inputs become available.
template<typename LayerT, typename BlockT>
class OpBlockSequencer
{
public:
    struct Pair
    {
        LayerT* layer;
        BlockT* block;
    };

    OpBlockSequencer() = default;
    ~OpBlockSequencer() = default;

    void Add(LayerT* layer, BlockT* block)
    {
        if (HasInputs(block))
        {
            AddReady({layer, block});
            ProcessPending();
        }
        else
        {
            m_Pending.emplace_back(Pair{layer,block});
        }
    }

    std::list<Pair>& Finish()
    {
        ProcessPending();
        if (m_Pending.size())
        {
            std::stringstream stm;
            stm << "[OpBlockSequencer] " <<  m_Pending.size();
            stm << " blocks could not be processed!";
            throw std::invalid_argument(stm.str());
        }
        return m_Ready;
    }
private:
    bool HasInputs(BlockT* block)
    {
        for (auto& inputTensorName : block->GetInputs())
        {
            if (inputTensorName.find("input") != std::string::npos)
            {
                continue;
            }

            if (inputTensorName.find("constant") != std::string::npos)
            {
                continue;
            }

            if (m_TensorMap.find(inputTensorName) == m_TensorMap.end())
            {
                return false;
            }
        }
        return true;
    }

    void AddReady(Pair&& pair)
    {
        m_Ready.emplace_back(pair);
        for (auto & outputTensor : pair.block->GetOutputs())
        {
            m_TensorMap[outputTensor] = 1;
        }
    }

    void ProcessPending()
    {
        auto itr = m_Pending.begin();
        while (itr != m_Pending.end())
        {
            if (HasInputs((*itr).block))
            {
                AddReady(std::move(*itr));
                itr = m_Pending.erase(itr);
            }
            else
            {
                ++itr;
            }
        }
    }
private:
    std::list<Pair> m_Ready;
    std::list<Pair> m_Pending;
    std::unordered_map<std::string, uint32_t> m_TensorMap;
};

} // namespace armnn
