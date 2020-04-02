//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "DynamicQuantizationVisitor.hpp"
#include "NetworkUtils.hpp"

#include <armnn/Descriptors.hpp>
#include <armnn/utility/IgnoreUnused.hpp>
#include <armnn/utility/PolymorphicDowncast.hpp>
#include <armnn/Types.hpp>

#include <limits>

namespace armnn
{

DynamicQuantizationVisitor::DynamicQuantizationVisitor(RangeTracker& rangeTracker, Graph& graph)
        : m_RangeTracker(rangeTracker),
          m_Graph(graph)
{}

void DynamicQuantizationVisitor::SetRange(const IConnectableLayer* layer, unsigned int outputIdx, float min, float max)
{
    m_RangeTracker.SetRange(layer, outputIdx, min, max);
}

void DynamicQuantizationVisitor::ForwardParentParameters(const IConnectableLayer* layer)
{
    for (unsigned int i = 0; i < layer->GetNumInputSlots(); ++i)
    {
        const IOutputSlot *outputSlot = layer->GetInputSlot(i).GetConnection();
        LayerGuid previousLayerId = outputSlot->GetOwningLayerGuid();
        unsigned int ownerIndex = outputSlot->CalculateIndexOnOwner();
        const auto parentRange = m_RangeTracker.GetRange(previousLayerId, ownerIndex);
        SetRange(layer, i, parentRange.first, parentRange.second);
    }
}

void DynamicQuantizationVisitor::AddToCalibratedLayers(const IConnectableLayer* layer)
{
    m_LayersToCalibrate.push_back(layer);
}

void DynamicQuantizationVisitor::AddToNonCalibratedLayers(const IConnectableLayer* layer)
{
    m_LayersNotToCalibrate.push_back(layer);
}

void DynamicQuantizationVisitor::FinishVisit()
{
    for (const IConnectableLayer* layer : m_LayersToCalibrate)
    {
        std::vector<DebugLayer*> newDebugLayers = InsertDebugLayerAfter(
            m_Graph, *PolymorphicDowncast<Layer*>(const_cast<IConnectableLayer*>(layer)));
        // record them so we can take them out again efficiently afterward
        m_DebugLayers.insert(std::end(m_DebugLayers), std::begin(newDebugLayers), std::end(newDebugLayers));
    }
}

void DynamicQuantizationVisitor::RemoveDebugLayers()
{
    for (DebugLayer* debugLayer : m_DebugLayers)
    {
        OutputSlot& proceedingOutputSlot = *debugLayer->GetInputSlot(0).GetConnectedOutputSlot();
        proceedingOutputSlot.Disconnect(debugLayer->GetInputSlot(0));

        for (InputSlot* succeedingInputSlot : debugLayer->GetOutputSlot(0).GetConnections())
        {
            debugLayer->GetOutputSlot(0).Disconnect(*succeedingInputSlot);
            proceedingOutputSlot.Connect(*succeedingInputSlot);
        }
        m_Graph.EraseLayer(debugLayer);
    }
    m_DebugLayers.clear();
}

void DynamicQuantizationVisitor::VisitNonCalibratedLayers() {
    RemoveDebugLayers();
    for (const IConnectableLayer* layer : m_LayersNotToCalibrate)
    {
        ForwardParentParameters(layer);
    }
}

void DynamicQuantizationVisitor::VisitAdditionLayer(const IConnectableLayer* layer,
                                                    const char* name)
{
    IgnoreUnused(name);
    SetRange(layer, 0, -20.f, 20.f);
    AddToCalibratedLayers(layer);
}

void DynamicQuantizationVisitor::VisitAbsLayer(const IConnectableLayer* layer,
                                               const char* name)
{
    IgnoreUnused(name);
    SetRange(layer, 0, -20.f, 20.f);
    AddToCalibratedLayers(layer);
}

void DynamicQuantizationVisitor::VisitArgMinMaxLayer(const IConnectableLayer* layer,
                                                     const ArgMinMaxDescriptor& desc,
                                                     const char* name)
{
    IgnoreUnused(name);
    IgnoreUnused(desc);
    SetRange(layer, 0, -20.f, 20.f);
    AddToCalibratedLayers(layer);
}

void DynamicQuantizationVisitor::VisitBatchNormalizationLayer(const IConnectableLayer* layer,
                                                              const BatchNormalizationDescriptor& desc,
                                                              const ConstTensor& mean,
                                                              const ConstTensor& variance,
                                                              const ConstTensor& beta,
                                                              const ConstTensor& gamma,
                                                              const char* name)
{
    IgnoreUnused(desc);
    IgnoreUnused(mean);
    IgnoreUnused(variance);
    IgnoreUnused(beta);
    IgnoreUnused(gamma);
    IgnoreUnused(name);
    SetRange(layer, 0, -15.0f, 15.0f);
    AddToCalibratedLayers(layer);
}

void DynamicQuantizationVisitor::VisitNormalizationLayer(const IConnectableLayer* layer,
                                 const NormalizationDescriptor& desc,
                                 const char* name)
{
    IgnoreUnused(desc);
    IgnoreUnused(name);
    SetRange(layer, 0, -15.0f, 15.0f);
    AddToCalibratedLayers(layer);
}

void DynamicQuantizationVisitor::VisitConvolution2dLayer(const IConnectableLayer* layer,
                                                         const Convolution2dDescriptor& convolution2dDescriptor,
                                                         const ConstTensor& weights,
                                                         const Optional<ConstTensor>& biases,
                                                         const char* name)
{
    IgnoreUnused(convolution2dDescriptor);
    IgnoreUnused(weights);
    IgnoreUnused(biases);
    IgnoreUnused(name);
    SetRange(layer, 0, -15.0f, 15.0f);
    AddToCalibratedLayers(layer);
}

void DynamicQuantizationVisitor::VisitDepthwiseConvolution2dLayer(const IConnectableLayer* layer,
                                                                  const DepthwiseConvolution2dDescriptor& desc,
                                                                  const ConstTensor& weights,
                                                                  const Optional<ConstTensor>& biases,
                                                                  const char* name)
{
    IgnoreUnused(desc);
    IgnoreUnused(weights);
    IgnoreUnused(biases);
    IgnoreUnused(name);
    SetRange(layer, 0, -15.0f, 15.0f);
    AddToCalibratedLayers(layer);
}

void DynamicQuantizationVisitor::VisitActivationLayer(const IConnectableLayer* layer,
                                                      const ActivationDescriptor& activationDescriptor,
                                                      const char* name)
{
    IgnoreUnused(name, activationDescriptor);
    switch (activationDescriptor.m_Function)
    {
        // Range is 0, 15 for Abs, Linear, ReLu and Soft ReLu
        case ActivationFunction::Abs:
        case ActivationFunction::Linear:
        case ActivationFunction::ReLu:
        case ActivationFunction::SoftReLu:
            SetRange(layer, 0, 0.f, 15.f);
            break;
        case ActivationFunction::BoundedReLu:
            SetRange(layer, 0, 0.f, activationDescriptor.m_A);
            break;
        case ActivationFunction::TanH:
            SetRange(layer, 0, -1.f, 1.f);
            break;
        case ActivationFunction::LeakyReLu:
            SetRange(layer, 0, -5.f, 15.f);
            break;
        default:
            SetRange(layer, 0, -15.f, 15.f);
            break;
    }
    AddToCalibratedLayers(layer);
}

void DynamicQuantizationVisitor::VisitFullyConnectedLayer(const IConnectableLayer *layer,
                                                          const FullyConnectedDescriptor& desc,
                                                          const ConstTensor& weights,
                                                          const Optional<ConstTensor>& biases,
                                                          const char *name)
{
    IgnoreUnused(desc);
    IgnoreUnused(weights);
    IgnoreUnused(biases);
    IgnoreUnused(name);
    SetRange(layer, 0, -15.0f, 15.0f);
    AddToCalibratedLayers(layer);
}

void DynamicQuantizationVisitor::VisitPermuteLayer(const IConnectableLayer* layer,
                                                   const PermuteDescriptor& permuteDescriptor,
                                                   const char* name)
{
    IgnoreUnused(permuteDescriptor);
    IgnoreUnused(name);
    AddToNonCalibratedLayers(layer);
}

void DynamicQuantizationVisitor::VisitSpaceToBatchNdLayer(const IConnectableLayer* layer,
                                                          const SpaceToBatchNdDescriptor& spaceToBatchNdDescriptor,
                                                          const char* name)
{
    IgnoreUnused(spaceToBatchNdDescriptor);
    IgnoreUnused(name);
    AddToNonCalibratedLayers(layer);
}

void DynamicQuantizationVisitor::VisitPooling2dLayer(const IConnectableLayer* layer,
                                                     const Pooling2dDescriptor& pooling2dDescriptor,
                                                     const char* name)
{
    IgnoreUnused(pooling2dDescriptor);
    IgnoreUnused(name);
    AddToNonCalibratedLayers(layer);
}

void DynamicQuantizationVisitor::VisitSoftmaxLayer(const IConnectableLayer* layer,
                                                   const SoftmaxDescriptor& softmaxDescriptor,
                                                   const char* name)
{
    IgnoreUnused(softmaxDescriptor);
    IgnoreUnused(name);
    SetRange(layer, 0, 0.f, 1.f);
    AddToCalibratedLayers(layer);
}

void DynamicQuantizationVisitor::VisitConstantLayer(const IConnectableLayer* layer,
                                                    const ConstTensor& input,
                                                    const char* name)
{
    IgnoreUnused(name);

    if (input.GetDataType() != DataType::Float32)
    {
        throw InvalidArgumentException("Quantization is supported only for FP32 tensors");
    }

    // Work out the range based on the input constants
    unsigned int inputNumElements = input.GetNumElements();
    const float* inputData = reinterpret_cast<const float*>(input.GetMemoryArea());

    float min = std::numeric_limits<float>::max();
    float max = std::numeric_limits<float>::lowest();

    for (unsigned int i = 0; i < inputNumElements; i++)
    {
        const float inputValue = inputData[i];

        min = std::min(min, inputValue);
        max = std::max(max, inputValue);
    }
    SetRange(layer, 0, min, max);
}

void DynamicQuantizationVisitor::VisitConcatLayer(const IConnectableLayer* layer,
                                                  const ConcatDescriptor& originsDescriptor,
                                                  const char* name)
{
    IgnoreUnused(name);
    IgnoreUnused(originsDescriptor);
    float min = std::numeric_limits<float>::max();
    float max = std::numeric_limits<float>::lowest();
    for (unsigned int i = 0; i < layer->GetNumInputSlots(); ++i)
    {
        const IOutputSlot* outputSlot = layer->GetInputSlot(i).GetConnection();
        LayerGuid layerId = outputSlot->GetOwningLayerGuid();
        unsigned int slotIndex = outputSlot->CalculateIndexOnOwner();
        RangeTracker::MinMaxRange range = m_RangeTracker.GetRange(layerId, slotIndex);
        min = std::min(min, range.first);
        max = std::max(max, range.second);
    }
    SetRange(layer, 0, min, max);
    AddToCalibratedLayers(layer);
}

void DynamicQuantizationVisitor::VisitReshapeLayer(const IConnectableLayer* layer,
                                                   const ReshapeDescriptor& reshapeDescriptor,
                                                   const char* name)
{
    IgnoreUnused(reshapeDescriptor);
    IgnoreUnused(name);
    AddToNonCalibratedLayers(layer);
}

void DynamicQuantizationVisitor::VisitSplitterLayer(const IConnectableLayer* layer,
                                                    const SplitterDescriptor& splitterDescriptor,
                                                    const char* name)
{
    IgnoreUnused(splitterDescriptor);
    IgnoreUnused(name);
    AddToNonCalibratedLayers(layer);
}

void DynamicQuantizationVisitor::VisitResizeBilinearLayer(const IConnectableLayer* layer,
                                                          const ResizeBilinearDescriptor& resizeDesc,
                                                          const char* name)
{
    IgnoreUnused(resizeDesc);
    IgnoreUnused(name);
    AddToNonCalibratedLayers(layer);
}

void DynamicQuantizationVisitor::VisitStridedSliceLayer(const IConnectableLayer* layer,
                                                        const StridedSliceDescriptor& stridedSliceDescriptor,
                                                        const char* name)
{
    IgnoreUnused(stridedSliceDescriptor);
    IgnoreUnused(name);
    AddToNonCalibratedLayers(layer);
}

void DynamicQuantizationVisitor::VisitBatchToSpaceNdLayer(const IConnectableLayer* layer,
                                                          const BatchToSpaceNdDescriptor& batchToSpaceNdDescriptor,
                                                          const char* name)
{
    IgnoreUnused(batchToSpaceNdDescriptor);
    IgnoreUnused(name);
    AddToNonCalibratedLayers(layer);
}

void DynamicQuantizationVisitor::VisitInputLayer(const IConnectableLayer* layer, LayerBindingId id, const char* name)
{
    IgnoreUnused(id);
    IgnoreUnused(name);
    SetRange(layer, 0, -0.0f, 0.0f);
    AddToCalibratedLayers(layer);
}

void DynamicQuantizationVisitor::VisitOutputLayer(const IConnectableLayer* layer, LayerBindingId id, const char* name)
{
    IgnoreUnused(id);
    IgnoreUnused(name);
    AddToNonCalibratedLayers(layer);
    m_OutputLayers.push_back(id);
}

const std::vector<LayerBindingId>& DynamicQuantizationVisitor::GetOutputLayers()
{
    return m_OutputLayers;
}

} //namespace armnn
