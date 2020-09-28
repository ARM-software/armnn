//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "NetworkQuantizer.hpp"
#include "NetworkQuantizerUtils.hpp"
#include "Graph.hpp"
#include "Layer.hpp"
#include "Network.hpp"
#include "DynamicQuantizationVisitor.hpp"
#include "StaticRangeVisitor.hpp"
#include "QuantizerVisitor.hpp"
#include "OverrideInputRangeVisitor.hpp"

#include <TensorIOUtils.hpp>

#include <armnn/ILayerVisitor.hpp>
#include <armnn/INetwork.hpp>
#include <armnn/Tensor.hpp>
#include <armnn/Types.hpp>

#include <armnnUtils/TensorUtils.hpp>
#include <armnn/utility/PolymorphicDowncast.hpp>

#include <mapbox/variant.hpp>

#include <vector>
#include <cmath>

namespace armnn
{

using TContainer = mapbox::util::variant<std::vector<float>, std::vector<int>, std::vector<unsigned char>>;

INetworkQuantizer* INetworkQuantizer::CreateRaw(INetwork* inputNetwork, const QuantizerOptions& options)
{
    return new NetworkQuantizer(inputNetwork, options);
}

INetworkQuantizerPtr INetworkQuantizer::Create(INetwork* inputNetwork, const QuantizerOptions& options)
{
    return INetworkQuantizerPtr(CreateRaw(inputNetwork, options), &INetworkQuantizer::Destroy);
}

void INetworkQuantizer::Destroy(INetworkQuantizer *quantizer)
{
    delete PolymorphicDowncast<NetworkQuantizer*>(quantizer);
}

void NetworkQuantizer::OverrideInputRange(LayerBindingId layerId, float min, float max)
{
    const Graph& graph = PolymorphicDowncast<const Network*>(m_InputNetwork)->GetGraph();
    auto inputLayers = graph.GetInputLayers();

    // Walk the input layers of the graph and override the quantization parameters of the one with the given id
    OverrideInputRangeVisitor overrideInputRangeVisitor(m_Ranges, layerId, RangeTracker::MinMaxRange{min, max});
    VisitLayers(inputLayers, overrideInputRangeVisitor);
}

void NetworkQuantizer::Refine(const InputTensors& inputTensors)
{
    // The first time Refine is called the m_Runtime and the DynamicQuantizationVisitor
    // will not have been created. Need to get the environment set up, Runtime loaded,
    // DynamicQuantizationVisitor created and run over the network to initialise itself
    // and the RangeTracker the Debug callback registered and an initial inference
    // done to set up the first min/max values
    if (!m_Runtime)
    {
        m_RefineCount = 0;
        m_Ranges.SetDynamicMode(true);
        const Graph& cGraph = PolymorphicDowncast<const Network*>(m_InputNetwork)->GetGraph().TopologicalSort();

        // need to insert Debug layers in the DynamicQuantizationVisitor
        Graph& graph = const_cast<Graph&>(cGraph);

        // Initialize RangeTracker to the default values for each layer.
        // The default values are overwritten by the min/max that is
        // recorded during the first dataset min/max calibration. This
        // initialisation is only required for the first call of Refine().
        m_DynamicQuantizationVisitor = DynamicQuantizationVisitor(m_Ranges, graph);
        VisitLayers(cGraph, m_DynamicQuantizationVisitor.value());

        IRuntime::CreationOptions options;
        m_Runtime = IRuntime::Create(options);

        // Optimize network - debug already enabled for layers that require quantization
        OptimizerOptions optimizerOptions(false, false);
        std::vector<BackendId> backends = {"CpuRef"};
        IOptimizedNetworkPtr optimizedNet = Optimize(*m_InputNetwork,
                                                     backends,
                                                     m_Runtime->GetDeviceSpec(),
                                                     optimizerOptions);

        m_Runtime->LoadNetwork(m_NetworkId, std::move(optimizedNet));

        // Debug callback function to refine min/max in RangeTracker
        auto rangeTrackerCallback = [&](LayerGuid guid, unsigned int slotIndex, ITensorHandle *tensorHandle) {
            // Get min/max pair from tensor data
            std::pair<float, float> minMax = armnnUtils::FindMinMax(tensorHandle);

            // For first calibration dataset, set min/max range in RangeTracker to
            // min/max ranges gathered during inference
            if (m_RefineCount == 0)
            {
                m_Ranges.ResetMinMax(guid, slotIndex, minMax.first, minMax.second);
            }
            else
            {
                // For every other calibration dataset, only set min/max range if the
                // values gathered are less than / greater than originally recorded.
                m_Ranges.RefineMin(guid, slotIndex, minMax.first);
                m_Ranges.RefineMax(guid, slotIndex, minMax.second);
            }
        };

        m_Runtime->RegisterDebugCallback(m_NetworkId, rangeTrackerCallback);
    }

    // Create output tensor for EnqueueWorkload
    std::vector<armnn::BindingPointInfo> outputBindings;
    auto outputLayers = m_DynamicQuantizationVisitor.value().GetOutputLayers();
    std::vector<TContainer> outputVectors;
    for (auto outputLayerBindingId : outputLayers)
    {
        auto outputTensorInfo = m_Runtime->GetOutputTensorInfo(m_NetworkId, outputLayerBindingId);
        outputBindings.push_back(std::make_pair(outputLayerBindingId, outputTensorInfo));
        outputVectors.push_back(std::vector<float>(outputTensorInfo.GetNumElements(), 0));
    }
    OutputTensors outputTensors = armnnUtils::MakeOutputTensors<TContainer>(outputBindings, outputVectors);

    // Execute EnqueueWorkload with calibration image
    m_Runtime->EnqueueWorkload(m_NetworkId, inputTensors, outputTensors);
    ++m_RefineCount;
}

INetworkPtr NetworkQuantizer::ExportNetwork()
{
    const Graph& graph = PolymorphicDowncast<const Network*>(m_InputNetwork)->GetGraph().TopologicalSort();

    // Step 1) Walk the graph and populate default min/max values for
    // intermediate tensors, only if Runtime does not exist (created
    // if Refine has been called)
    if (!m_Runtime)
    {
        m_Ranges.SetDynamicMode(false);
        StaticRangeVisitor rangeVisitor(m_Ranges);
        VisitLayers(graph, rangeVisitor);
    }
    else
    {
        // Set min/max range of non-calibrated layers to parent layer's range
        m_DynamicQuantizationVisitor.value().VisitNonCalibratedLayers();
        // now tear down the runtime and the dynamic visitor.
        m_Runtime.reset(nullptr);
        m_DynamicQuantizationVisitor = EmptyOptional();
        m_RefineCount = 0;
    }

    // Step 2) Convert input InputNetwork to Quantized InputNetwork
    std::unique_ptr<IQuantizationScheme> quantizationScheme;
    switch (m_Options.m_ActivationFormat)
    {
        case DataType::QAsymmU8:
            quantizationScheme = std::make_unique<QAsymmU8QuantizationScheme>();
            break;
        case DataType::QAsymmS8:
            quantizationScheme = std::make_unique<QAsymmS8QuantizationScheme>();
            break;
        case DataType::QSymmS8:
            quantizationScheme = std::make_unique<QSymmS8QuantizationScheme>();
            break;
        case DataType::QSymmS16:
            quantizationScheme = std::make_unique<QSymm16QuantizationScheme>();
            break;
        default:
            throw InvalidArgumentException("Unsupported quantization target");
    }

    QuantizerVisitor quantizerVisitor(m_Ranges, quantizationScheme.get(), m_Options.m_PreserveType);
    VisitLayers(graph, quantizerVisitor);

    // clear the ranges
    m_Ranges.Reset();

    return quantizerVisitor.RetrieveFinalNetwork();
}

} //namespace armn
