//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "Network.hpp"
#include "Graph.hpp"
#include "Layer.hpp"
#include "DeviceSpec.hpp"
#include "Optimizer.hpp"
#include "SubgraphViewSelector.hpp"
#include "BackendSettings.hpp"
#include "optimizations/All.hpp"

#include <backendsCommon/CpuTensorHandle.hpp>
#include <backendsCommon/WorkloadFactory.hpp>
#include <backendsCommon/BackendRegistry.hpp>
#include <backendsCommon/IBackendInternal.hpp>

#include <armnn/Exceptions.hpp>
#include <armnn/Utils.hpp>
#include <armnn/TypesUtils.hpp>

#include <fcntl.h>
#include <algorithm>
#include <fstream>
#include <memory>
#include <vector>
#include <algorithm>

#include <boost/assert.hpp>
#include <boost/format.hpp>
#include <boost/log/trivial.hpp>
#include <boost/numeric/conversion/converter_policies.hpp>
#include <boost/cast.hpp>

namespace armnn
{

armnn::INetwork* INetwork::CreateRaw()
{
    return new Network();
}

armnn::INetworkPtr INetwork::Create()
{
    return INetworkPtr(CreateRaw(), &INetwork::Destroy);
}

void INetwork::Destroy(INetwork* network)
{
    delete boost::polymorphic_downcast<Network*>(network);
}

Status Network::PrintGraph()
{
    m_Graph->Print();
    return Status::Success;
}

void IOptimizedNetwork::Destroy(IOptimizedNetwork* network)
{
    delete boost::polymorphic_downcast<OptimizedNetwork*>(network);
}

Status OptimizedNetwork::PrintGraph()
{
    m_Graph->Print();
    return Status::Success;
}

Status OptimizedNetwork::SerializeToDot(std::ostream& stream) const
{
    return m_Graph->SerializeToDot(stream);
}

struct OptimizationResult
{
    bool m_Warning;
    bool m_Error;

    OptimizationResult()
        : m_Warning(false)
        , m_Error(false)
    {}
};

void ReportError(const std::string& errorMessage,
                 Optional<std::vector<std::string>&> errorMessages)
{
    std::stringstream fullErrorMessage;
    fullErrorMessage << "ERROR: " << errorMessage;
    BOOST_LOG_TRIVIAL(warning) << fullErrorMessage.str();
    if (errorMessages)
    {
        errorMessages.value().push_back(fullErrorMessage.str());
    }
}

void ReportWarning(const std::string& warningMessage,
                   Optional<std::vector<std::string>&> warningMessages)
{
    std::stringstream fullWarningMessage;
    fullWarningMessage << "WARNING: " << warningMessage;
    BOOST_LOG_TRIVIAL(warning) << fullWarningMessage.str();
    if (warningMessages)
    {
        warningMessages.value().push_back(fullWarningMessage.str());
    }
}

bool CheckScaleSetOnQuantizedType(Layer* layer, Optional<std::vector<std::string>&> errMessages)
{
    bool noErrors = true;
    unsigned int numOutputs = layer->GetNumOutputSlots();
    for (unsigned int i = 0; i < numOutputs; i++) {
        OutputSlot& outputSlot = layer->GetOutputSlot(i);
        TensorInfo info = outputSlot.GetTensorInfo();
        if (DataType::QuantisedAsymm8 == info.GetDataType()) {
            if (0.f == info.GetQuantizationScale()) {
                noErrors = false;
                std::stringstream ss;
                ss << "output " << i << " of layer " << GetLayerTypeAsCString(layer->GetType())
                   << " (" << layer->GetNameStr() << ") is of type"
                   << " Quantized 8 bit but its scale parameter has not been set";
                ReportError(ss.str(), errMessages);
            }
            // Softmax under QuantisedAsymm8 must always be scale (1.0f/256.0f) and offset 0
            if ((info.GetQuantizationScale() != (1.0f / 256.0f) ||
                 info.GetQuantizationOffset() != 0) &&
                 layer->GetType() == armnn::LayerType::Softmax)
            {
                std::stringstream ss;
                ss << "Quantization parameters for Softmax layer (Scale: " <<
                info.GetQuantizationScale() << " and Offset: " << info.GetQuantizationOffset() <<
                ") are incorrect and have been updated to Scale: 0.00390625 and Offset: 0";
                BOOST_LOG_TRIVIAL(warning) << ss.str();
                info.SetQuantizationScale((1.0f /256.0f));
                info.SetQuantizationOffset(0);
                outputSlot.SetTensorInfo(info);
            }
        }
    }
    return noErrors;
}

OptimizationResult AssignBackends(OptimizedNetwork* optNetObjPtr,
                                  BackendSettings& backendSettings,
                                  Graph::Iterator& firstLayer,
                                  Graph::Iterator& lastLayer,
                                  Optional<std::vector<std::string>&> errMessages)
{
    OptimizationResult result;

    // Helper lambda to compose meaningful error message before returning with error
    auto ReturnWithError = [&](const Layer* layer)
    {
        std::stringstream failureMsg;
        failureMsg << "Layer of type " << GetLayerTypeAsCString(layer->GetType())
                   << " is not supported on any preferred backend " << backendSettings.m_PreferredBackends;
        ReportError(failureMsg.str(), errMessages);

        result.m_Error = true;
        return result;
    };

    auto availablePreferredBackends = backendSettings.GetAvailablePreferredBackends();
    if (availablePreferredBackends.empty())
    {
        std::stringstream failureMsg;
        failureMsg << "No preferred backends are available";
        ReportError(failureMsg.str(), errMessages);

        result.m_Error = true;
        return result;
    }

    for (auto it = firstLayer; it != lastLayer; ++it)
    {
        auto layer = *it;
        DataType dataType = layer->GetDataType();
        std::string reasonIfUnsupported;
        bool found = false;
        if (!CheckScaleSetOnQuantizedType(layer, errMessages))
        {
            // don't bomb immediately, find all the quantized outputs
            // which haven't had a scale set and report them all back.
            result.m_Error = true;
        }

        for (const auto& backend : availablePreferredBackends)
        {
            // need to set the compute device on the layer
            // before we can check if it is supported
            layer->SetBackendId(backend);
            if (!IWorkloadFactory::IsLayerSupported(*layer, dataType, reasonIfUnsupported))
            {
                if (dataType == DataType::Float16)
                {
                    if (IWorkloadFactory::IsLayerSupported(*layer, DataType::Float32, reasonIfUnsupported)
                        && layer->GetType() != LayerType::ConvertFp32ToFp16
                        && layer->GetType() != LayerType::ConvertFp16ToFp32)
                    {
                        // Insert FP16 -> FP32 conversion layer before current layer
                        std::vector<ConvertFp16ToFp32Layer*> convertFp16ToFp32Layers =
                            InsertConvertFp16ToFp32LayersBefore(optNetObjPtr->GetGraph(), *layer);

                        // Insert FP32 -> FP16 conversion layer after current layer
                        std::vector<ConvertFp32ToFp16Layer*> convertFp32ToFp16Layers =
                            InsertConvertFp32ToFp16LayersAfter(optNetObjPtr->GetGraph(), *layer);

                        // Assign a supported backend to the newly introduced conversion layers
                        auto AssignFirstSupportedBackend = [&](Layer* layer, BackendId preferredBackend)
                        {
                            bool supportedBackendFound = false;
                            std::string reasonIfUnsupported;

                            // Try preferred backend first
                            layer->SetBackendId(preferredBackend);
                            if (IWorkloadFactory::IsLayerSupported(*layer,
                                                                   EmptyOptional(),
                                                                   reasonIfUnsupported))
                            {
                                supportedBackendFound = true;
                            }
                            else
                            {
                                for (const auto& backend : availablePreferredBackends)
                                {
                                    // Skip preferred backend (we already determined that it is not supported)
                                    if (backend == preferredBackend)
                                    {
                                        continue;
                                    }

                                    layer->SetBackendId(backend);
                                    if (IWorkloadFactory::IsLayerSupported(*layer,
                                                                           EmptyOptional(),
                                                                           reasonIfUnsupported))
                                    {
                                        supportedBackendFound = true;
                                        break;
                                    }
                                }
                            }

                            return supportedBackendFound;
                        };

                        for (ConvertFp16ToFp32Layer* convertLayer : convertFp16ToFp32Layers)
                        {
                            if (!AssignFirstSupportedBackend(convertLayer, backend))
                            {
                                return ReturnWithError(convertLayer);
                            }
                        }

                        for (ConvertFp32ToFp16Layer* convertLayer : convertFp32ToFp16Layers)
                        {
                            if (!AssignFirstSupportedBackend(convertLayer, backend))
                            {
                                return ReturnWithError(convertLayer);
                            }
                        }

                        found = true;
                        break;
                    }
                }
                std::stringstream warningMsg;
                warningMsg << "Layer of type " << GetLayerTypeAsCString(layer->GetType())
                           << " is not supported on requested backend " << layer->GetBackendId().Get()
                           << " for data type " << GetDataTypeName(dataType)
                           << " (reason: " << reasonIfUnsupported
                           << "), falling back to the next backend.";
                ReportWarning(warningMsg.str(), errMessages);
            }
            else
            {
                found = true;
                backendSettings.m_SelectedBackends.insert(backend);
                break;
            }
        }

        // If the layer is unsupported by any devices, log and return a null network.
        if (!found)
        {
            // NOTE: if the layer is not an operation queue type AND we have not got CpuRef as a
            //       fallback we should set the compute device on the layer to CpuRef (these are not
            //       available as accelerated operations, or are only available under certain
            //       conditions, currently they comprise MemCopy, Constant, Permute)
            armnn::LayerType layerType = layer->GetType();
            if (!backendSettings.IsCpuRefUsed() && (layerType == armnn::LayerType::MemCopy ||
                                                    layerType == armnn::LayerType::Constant ||
                                                    layerType == armnn::LayerType::Permute))
            {
                BackendId cpuBackendId(armnn::Compute::CpuRef);
                layer->SetBackendId(cpuBackendId);
                backendSettings.m_SelectedBackends.insert(cpuBackendId);
            }
            else
            {
                return ReturnWithError(layer);
            }
        }
    }

    return result;
}

OptimizationResult AssignBackends(OptimizedNetwork* optNetObjPtr,
                                  BackendSettings& backendSettings,
                                  SubgraphView& subgraph,
                                  Optional<std::vector<std::string>&> errMessages)
{
    Graph::Iterator firstLayer = subgraph.begin();
    Graph::Iterator lastLayer  = subgraph.end();
    return AssignBackends(optNetObjPtr,
                          backendSettings,
                          firstLayer,
                          lastLayer,
                          errMessages);
}

OptimizationResult ApplyBackendOptimizations(OptimizedNetwork* optNetObjPtr,
                                             BackendSettings& backendSettings,
                                             Optional<std::vector<std::string>&> errMessages)
{
    BOOST_ASSERT(optNetObjPtr);

    OptimizationResult result;

    // Get the optimized graph
    Graph& optGraph = optNetObjPtr->GetGraph();

    // Get the entire graph as a sub-graph
    SubgraphView mainSubgraph(optGraph);

    // Run backend specific optimizations
    auto const& backendRegistry = BackendRegistryInstance();
    for (auto&& selectedBackend : backendSettings.m_SelectedBackends)
    {
        auto backendFactory = backendRegistry.GetFactory(selectedBackend);
        auto backendObjPtr  = backendFactory();
        BOOST_ASSERT(backendObjPtr);

        // Select sub-graphs based on backend
        SubgraphViewSelector::Subgraphs subgraphs =
                SubgraphViewSelector::SelectSubgraphs(mainSubgraph,
                                                      // Select layers assigned to the requested backend
                                                      [&backendObjPtr](const Layer& layer)
                                                      {
                                                          return layer.GetType() != LayerType::Input &&
                                                                 layer.GetType() != LayerType::Output &&
                                                                 layer.GetBackendId() == backendObjPtr->GetId();
                                                      });
        if (subgraphs.empty())
        {
            // No sub-graphs found, try with next selected backend
            continue;
        }

        // Try to optimize each sub-graph
        for (auto& subgraph : subgraphs)
        {
            // Try to optimize the current sub-graph
            OptimizationViews optimizationViews = backendObjPtr->OptimizeSubgraphView(*subgraph);
            BOOST_ASSERT(optimizationViews.Validate(*subgraph));

            // Optimization attempted, check the resulting optimized sub-graph
            for (auto& substitution : optimizationViews.GetSubstitutions())
            {
                // Sub-graph optimized, substitute the sub-graph with the new optimized one in the main optimized graph
                SubgraphView& replacementSubgraph   = substitution.m_ReplacementSubgraph;
                SubgraphView& substitutableSubgraph = substitution.m_SubstitutableSubgraph;
                optGraph.SubstituteSubgraph(substitutableSubgraph, replacementSubgraph);

                // Assign the current backend to the optimized sub-graph
                std::for_each(replacementSubgraph.begin(), replacementSubgraph.end(), [&selectedBackend](Layer* l)
                    {
                        BOOST_ASSERT(l);
                        l->SetBackendId(selectedBackend);
                    });
            }

            if (!optimizationViews.GetFailedSubgraphs().empty())
            {
                std::stringstream warningMsg;
                warningMsg << "Some sub-graph(s) failed to optimized on " << backendObjPtr->GetId() << " backend.";
                ReportWarning(warningMsg.str(), errMessages);

                // Failed to optimize the given sub-graph, re-assign the sub-graph layers to other available backends
                BackendSettings settingsCopy(backendSettings);
                if (!backendObjPtr->GetId().IsCpuRef())
                {
                    // Add the current backend to the list of backends to ignore
                    settingsCopy.m_IgnoredBackends.insert(backendObjPtr->GetId());
                }

                int count=0;
                for (auto& failedSubgraph : optimizationViews.GetFailedSubgraphs())
                {
                    // An error occurred: the optimization was attempted but not performed, try different backends
                    std::stringstream subgraphMsg;
                    subgraphMsg << "Re-assigning backends to " << failedSubgraph.GetLayers().size()
                                << " layers inside sub-graph " << count++;
                    ReportWarning(warningMsg.str(), errMessages);

                    OptimizationResult reassignmentResult = AssignBackends(optNetObjPtr,
                                                                           settingsCopy,
                                                                           *subgraph,
                                                                           errMessages);
                    if (reassignmentResult.m_Error)
                    {
                        // Failed to re-assign one of the remaining backends to each layer of the sub-graph
                        result.m_Error = true;
                        return result;
                    }
                }
            }
        }
    }

    return result;
}

IOptimizedNetworkPtr Optimize(const INetwork& inNetwork,
                              const std::vector<BackendId>& backendPreferences,
                              const IDeviceSpec& deviceSpec,
                              const OptimizerOptions& options,
                              Optional<std::vector<std::string>&> errMessages)
{
    if (backendPreferences.empty())
    {
        throw armnn::InvalidArgumentException("Invoked Optimize with no backends specified");
    }

    const Network& network = *boost::polymorphic_downcast<const Network*>(&inNetwork);
    std::unique_ptr<Graph> graph = std::make_unique<Graph>(network.GetGraph());

    auto optNet = IOptimizedNetworkPtr(new OptimizedNetwork(std::move(graph)), &IOptimizedNetwork::Destroy);

    OptimizedNetwork* optNetObjPtr = boost::polymorphic_downcast<OptimizedNetwork*>(optNet.get());

    // Get the optimized graph
    Graph& optGraph = optNetObjPtr->GetGraph();

    // Perform optimisation passes
    using namespace optimizations;
    Optimizer::Pass(optGraph, MakeOptimizations(SquashEqualPermuteSiblings(),
                                                SquashEqualReshapeSiblings(),
                                                OptimizeInversePermutes(),
                                                MovePermuteUp(),
                                                PermuteAsReshape(),
                                                OptimizeConsecutiveReshapes(),
                                                FoldPadIntoConvolution2d()));

    // Infer the tensor infos for all output slots. Throws an exception on failure
    optGraph.InferTensorInfos();

    // If Fp32 to Fp16 optimization is set convert Fp32 network to Fp16
    if (options.m_ReduceFp32ToFp16)
    {
        Optimizer::Pass(optGraph, MakeOptimizations(Fp32NetworkToFp16Converter()));
    }

    // Initialize backend settings
    BackendSettings backendSettings(backendPreferences, deviceSpec);
    if (backendSettings.GetAvailablePreferredBackends().empty())
    {
        std::stringstream failureMsg;
        failureMsg << "None of the preferred backends " << backendPreferences
                   << " are supported. Current platform provides " << backendSettings.m_SupportedBackends;
        ReportError(failureMsg.str(), errMessages);
        return IOptimizedNetworkPtr(nullptr, &IOptimizedNetwork::Destroy);
    }

    // Assign an available backend to each layer
    Graph::Iterator firstLayer = optGraph.begin();
    Graph::Iterator lastLayer  = optGraph.end();
    OptimizationResult assigBackendsResult = AssignBackends(optNetObjPtr,
                                                            backendSettings,
                                                            firstLayer,
                                                            lastLayer,
                                                            errMessages);
    if (assigBackendsResult.m_Error)
    {
        // Failed to assign a backend to each layer
        return IOptimizedNetworkPtr(nullptr, &IOptimizedNetwork::Destroy);
    }

    Optimizer::Pass(optGraph, MakeOptimizations(OptimizeInverseConversionsFp16(),
                                                OptimizeInverseConversionsFp32()));

    // Apply the backend-specific optimizations
    OptimizationResult backendOptimizationResult = ApplyBackendOptimizations(optNetObjPtr,
                                                                             backendSettings,
                                                                             errMessages);
    if (backendOptimizationResult.m_Error)
    {
        // Failed to apply the backend-specific optimizations
        return IOptimizedNetworkPtr(nullptr, &IOptimizedNetwork::Destroy);
    }

    // If the debug flag is set, then insert a DebugLayer after each layer
    // Doing this after applying the backend optimizations as they might have changed some layers
    if (options.m_Debug)
    {
        Optimizer::Pass(optGraph, MakeOptimizations(InsertDebugLayer()));
    }

    optGraph.AddCopyLayers();

    // Convert constants
    Optimizer::Pass(optGraph, MakeOptimizations(ConvertConstantsFloatToHalf()));
    Optimizer::Pass(optGraph, MakeOptimizations(ConvertConstantsHalfToFloat()));

    // Run backend specific optimizations
    for (auto&& chosenBackend : backendSettings.m_SelectedBackends)
    {
        auto factoryFun = BackendRegistryInstance().GetFactory(chosenBackend);
        auto backendPtr = factoryFun();
        BOOST_ASSERT(backendPtr.get() != nullptr);

        auto backendSpecificOptimizations = backendPtr->GetOptimizations();
        if (!backendSpecificOptimizations.empty())
        {
            Optimizer::Pass(optNetObjPtr->GetGraph(), backendSpecificOptimizations);
        }
    }

    return optNet;
}

Network::Network()
: m_Graph(std::make_unique<Graph>())
{
}

Network::~Network()
{
}

IConnectableLayer* Network::AddInputLayer(LayerBindingId id, const char* name)
{
    return m_Graph->AddLayer<InputLayer>(id, name);
}

IConnectableLayer* Network::AddBatchToSpaceNdLayer(const BatchToSpaceNdDescriptor& batchToSpaceNdDescriptor,
                                            const char* name)
{
    return m_Graph->AddLayer<BatchToSpaceNdLayer>(batchToSpaceNdDescriptor, name);
}

IConnectableLayer* Network::AddFullyConnectedLayerImpl(const FullyConnectedDescriptor& fullyConnectedDescriptor,
                                                       const ConstTensor& weights,
                                                       const Optional<ConstTensor>& biases,
                                                       const char* name)
{
    if (fullyConnectedDescriptor.m_BiasEnabled && !biases.has_value())
    {
        throw InvalidArgumentException("AddFullyConnectedLayer: biases cannot be empty");
    }

    const auto layer = m_Graph->AddLayer<FullyConnectedLayer>(fullyConnectedDescriptor, name);

    layer->m_Weight = std::make_unique<ScopedCpuTensorHandle>(weights);

    if (fullyConnectedDescriptor.m_BiasEnabled)
    {
        layer->m_Bias = std::make_unique<ScopedCpuTensorHandle>(biases.value());
    }

    return layer;
}

IConnectableLayer* Network::AddFullyConnectedLayer(const FullyConnectedDescriptor& fullyConnectedDescriptor,
                                                   const ConstTensor& weights,
                                                   const Optional<ConstTensor>& biases,
                                                   const char* name)
{
    return AddFullyConnectedLayerImpl(fullyConnectedDescriptor, weights, biases, name);
}

IConnectableLayer* Network::AddFullyConnectedLayer(const FullyConnectedDescriptor& fullyConnectedDescriptor,
                                                   const ConstTensor& weights,
                                                   const char* name)
{
    Optional<ConstTensor> biases;
    return AddFullyConnectedLayerImpl(fullyConnectedDescriptor, weights, biases, name);
}

IConnectableLayer* Network::AddFullyConnectedLayer(const FullyConnectedDescriptor& fullyConnectedDescriptor,
                                                   const ConstTensor& weights,
                                                   const ConstTensor& biases,
                                                   const char* name)
{
    Optional<ConstTensor> optionalBiases(biases);
    return AddFullyConnectedLayerImpl(fullyConnectedDescriptor, weights, optionalBiases, name);
}

IConnectableLayer* Network::AddConcatLayer(const OriginsDescriptor& mergerDescriptor,
                                           const char* name)
{
    ARMNN_NO_DEPRECATE_WARN_BEGIN
    return AddMergerLayer(mergerDescriptor, name);
    ARMNN_NO_DEPRECATE_WARN_END
}

IConnectableLayer* Network::AddConvolution2dLayerImpl(const Convolution2dDescriptor& convolution2dDescriptor,
                                                      const ConstTensor& weights,
                                                      const Optional<ConstTensor>& biases,
                                                      const char* name)
{
    if (convolution2dDescriptor.m_BiasEnabled && !biases.has_value())
    {
        throw InvalidArgumentException("AddConvolution2dLayer: biases cannot be empty");
    }

    const auto layer = m_Graph->AddLayer<Convolution2dLayer>(convolution2dDescriptor, name);

    layer->m_Weight = std::make_unique<ScopedCpuTensorHandle>(weights);

    if (convolution2dDescriptor.m_BiasEnabled)
    {
        layer->m_Bias = std::make_unique<ScopedCpuTensorHandle>(biases.value());
    }

    return layer;
}

IConnectableLayer* Network::AddConvolution2dLayer(const Convolution2dDescriptor& convolution2dDescriptor,
                                                  const ConstTensor& weights,
                                                  const Optional<ConstTensor>& biases,
                                                  const char* name)
{
    return AddConvolution2dLayerImpl(convolution2dDescriptor, weights, biases, name);
}

IConnectableLayer* Network::AddConvolution2dLayer(const Convolution2dDescriptor& convolution2dDescriptor,
                                                  const ConstTensor& weights,
                                                  const char* name)
{
    Optional<ConstTensor> biases;
    return AddConvolution2dLayerImpl(convolution2dDescriptor, weights, biases, name);
}

IConnectableLayer* Network::AddConvolution2dLayer(const Convolution2dDescriptor& convolution2dDescriptor,
                                                  const ConstTensor& weights,
                                                  const ConstTensor& biases,
                                                  const char* name)
{
    Optional<ConstTensor> optionalBiases(biases);
    return AddConvolution2dLayerImpl(convolution2dDescriptor, weights, optionalBiases, name);
}

IConnectableLayer* Network::AddDepthwiseConvolution2dLayerImpl(
    const DepthwiseConvolution2dDescriptor& convolution2dDescriptor,
    const ConstTensor& weights,
    const Optional<ConstTensor>& biases,
    const char* name)
{
    if (convolution2dDescriptor.m_BiasEnabled && !biases.has_value())
    {
        throw InvalidArgumentException("AddDepthwiseConvolution2dLayer: biases cannot be empty");
    }

    const auto layer = m_Graph->AddLayer<DepthwiseConvolution2dLayer>(convolution2dDescriptor, name);

    layer->m_Weight = std::make_unique<ScopedCpuTensorHandle>(weights);

    if (convolution2dDescriptor.m_BiasEnabled)
    {
        layer->m_Bias = std::make_unique<ScopedCpuTensorHandle>(biases.value());
    }

    return layer;
}

IConnectableLayer* Network::AddDepthwiseConvolution2dLayer(
        const DepthwiseConvolution2dDescriptor& convolution2dDescriptor,
        const ConstTensor& weights,
        const Optional<ConstTensor>& biases,
        const char* name)
{
    return AddDepthwiseConvolution2dLayerImpl(convolution2dDescriptor, weights, biases, name);
}

IConnectableLayer* Network::AddDepthwiseConvolution2dLayer(
    const DepthwiseConvolution2dDescriptor& convolution2dDescriptor,
    const ConstTensor& weights,
    const char* name)
{
    Optional<ConstTensor> biases;
    return AddDepthwiseConvolution2dLayerImpl(convolution2dDescriptor, weights, biases, name);
}

IConnectableLayer* Network::AddDepthwiseConvolution2dLayer(
    const DepthwiseConvolution2dDescriptor& convolution2dDescriptor,
    const ConstTensor& weights,
    const ConstTensor& biases,
    const char* name)
{
    Optional<ConstTensor> optionalBiases(biases);
    return AddDepthwiseConvolution2dLayerImpl(convolution2dDescriptor, weights, optionalBiases, name);
}

IConnectableLayer* Network::AddDetectionPostProcessLayer(const armnn::DetectionPostProcessDescriptor& descriptor,
                                                         const ConstTensor& anchors, const char* name)
{
    const auto layer = m_Graph->AddLayer<DetectionPostProcessLayer>(descriptor, name);

    layer->m_Anchors = std::make_unique<ScopedCpuTensorHandle>(anchors);

    return layer;
}

IConnectableLayer* Network::AddPermuteLayer(const PermuteDescriptor& permuteDescriptor,
                                            const char* name)
{
    return m_Graph->AddLayer<PermuteLayer>(permuteDescriptor, name);
}

IConnectableLayer* Network::AddPooling2dLayer(const Pooling2dDescriptor& pooling2dDescriptor,
    const char* name)
{
    return m_Graph->AddLayer<Pooling2dLayer>(pooling2dDescriptor, name);
}

IConnectableLayer* Network::AddActivationLayer(const ActivationDescriptor& activationDescriptor,
    const char* name)
{
    return m_Graph->AddLayer<ActivationLayer>(activationDescriptor, name);
}

IConnectableLayer* Network::AddNormalizationLayer(const NormalizationDescriptor&
normalizationDescriptor,
    const char* name)
{
    return m_Graph->AddLayer<NormalizationLayer>(normalizationDescriptor, name);
}

IConnectableLayer* Network::AddSoftmaxLayer(const SoftmaxDescriptor& softmaxDescriptor,
    const char* name)
{
    return m_Graph->AddLayer<SoftmaxLayer>(softmaxDescriptor, name);
}

IConnectableLayer* Network::AddSplitterLayer(const ViewsDescriptor& splitterDescriptor,
    const char* name)
{
    return m_Graph->AddLayer<SplitterLayer>(splitterDescriptor, name);
}

IConnectableLayer* Network::AddMaximumLayer(const char* name)
{
    return m_Graph->AddLayer<MaximumLayer>(name);
}

IConnectableLayer* Network::AddMinimumLayer(const char* name)
{
    return m_Graph->AddLayer<MinimumLayer>(name);
}

IConnectableLayer* Network::AddMergerLayer(const OriginsDescriptor& mergerDescriptor,
                                           const char* name)
{
    return m_Graph->AddLayer<MergerLayer>(mergerDescriptor, name);
}

IConnectableLayer* Network::AddAdditionLayer(const char* name)
{
    return m_Graph->AddLayer<AdditionLayer>(name);
}

IConnectableLayer* Network::AddMultiplicationLayer(const char* name)
{
    return m_Graph->AddLayer<MultiplicationLayer>(name);
}

IConnectableLayer* Network::AddOutputLayer(LayerBindingId id, const char* name)
{
    return m_Graph->AddLayer<OutputLayer>(id, name);
}

IConnectableLayer* Network::AddBatchNormalizationLayer(const BatchNormalizationDescriptor& desc,
                                                       const ConstTensor&                  mean,
                                                       const ConstTensor&                  variance,
                                                       const ConstTensor&                  beta,
                                                       const ConstTensor&                  gamma,
                                                       const char*                         name)
{
    const auto layer = m_Graph->AddLayer<BatchNormalizationLayer>(desc, name);

    layer->m_Mean = std::make_unique<ScopedCpuTensorHandle>(mean);
    layer->m_Variance = std::make_unique<ScopedCpuTensorHandle>(variance);
    layer->m_Beta = std::make_unique<ScopedCpuTensorHandle>(beta);
    layer->m_Gamma = std::make_unique<ScopedCpuTensorHandle>(gamma);

    return layer;
}

IConnectableLayer* Network::AddResizeBilinearLayer(const ResizeBilinearDescriptor&
resizeDescriptor, const char* name)
{
    return m_Graph->AddLayer<ResizeBilinearLayer>(resizeDescriptor,name);
}

IConnectableLayer* Network::AddL2NormalizationLayer(const L2NormalizationDescriptor& desc,
                                                    const char* name)
{
    return m_Graph->AddLayer<L2NormalizationLayer>(desc, name);
}

IConnectableLayer* Network::AddConstantLayer(const ConstTensor& input, const char* name)
{
    auto layer = m_Graph->AddLayer<ConstantLayer>(name);

    layer->m_LayerOutput = std::make_unique<ScopedCpuTensorHandle>(input);

    return layer;
}

IConnectableLayer* Network::AddReshapeLayer(const ReshapeDescriptor& reshapeDescriptor,
                                            const char* name)
{
    return m_Graph->AddLayer<ReshapeLayer>(reshapeDescriptor, name);
}

IConnectableLayer* Network::AddSpaceToBatchNdLayer(const SpaceToBatchNdDescriptor& spaceToBatchNdDescriptor,
                                                   const char* name)
{
    return m_Graph->AddLayer<SpaceToBatchNdLayer>(spaceToBatchNdDescriptor, name);
}

IConnectableLayer* Network::AddFloorLayer(const char* name)
{
    return m_Graph->AddLayer<FloorLayer>(name);
}

IConnectableLayer* Network::AddLstmLayer(const LstmDescriptor&  descriptor,
                                         const LstmInputParams& params,
                                         const char* name)
{
    const auto layer = m_Graph->AddLayer<LstmLayer>(descriptor, name);

    //Lstm Basic Parameters
    layer->m_BasicParameters.m_InputToForgetWeights =
        std::make_unique<ScopedCpuTensorHandle>(*(params.m_InputToForgetWeights));
    layer->m_BasicParameters.m_InputToCellWeights =
        std::make_unique<ScopedCpuTensorHandle>(*(params.m_InputToCellWeights));
    layer->m_BasicParameters.m_InputToOutputWeights =
        std::make_unique<ScopedCpuTensorHandle>(*(params.m_InputToOutputWeights));
    layer->m_BasicParameters.m_RecurrentToForgetWeights =
        std::make_unique<ScopedCpuTensorHandle>(*(params.m_RecurrentToForgetWeights));
    layer->m_BasicParameters.m_RecurrentToCellWeights =
        std::make_unique<ScopedCpuTensorHandle>(*(params.m_RecurrentToCellWeights));
    layer->m_BasicParameters.m_RecurrentToOutputWeights =
        std::make_unique<ScopedCpuTensorHandle>(*(params.m_RecurrentToOutputWeights));
    layer->m_BasicParameters.m_ForgetGateBias =
            std::make_unique<ScopedCpuTensorHandle>(*(params.m_ForgetGateBias));
    layer->m_BasicParameters.m_CellBias =
            std::make_unique<ScopedCpuTensorHandle>(*(params.m_CellBias));
    layer->m_BasicParameters.m_OutputGateBias =
            std::make_unique<ScopedCpuTensorHandle>(*(params.m_OutputGateBias));

    //Lstm Cifg parameters
    if(!descriptor.m_CifgEnabled)
    {
        if(params.m_InputToInputWeights == nullptr)
        {
            throw InvalidArgumentException("AddLstmLayer: Input To Input Weights cannot be NULL");
        }
        if(params.m_RecurrentToInputWeights == nullptr)
        {
            throw InvalidArgumentException(
                    "AddLstmLayer: Recurrent To Input Weights cannot be NULL");
        }
        if(params.m_InputGateBias == nullptr)
        {
            throw InvalidArgumentException("AddLstmLayer: Input Gate Bias cannot be NULL");
        }
        layer->m_CifgParameters.m_InputToInputWeights =
            std::make_unique<ScopedCpuTensorHandle>(*(params.m_InputToInputWeights));
        layer->m_CifgParameters.m_RecurrentToInputWeights =
            std::make_unique<ScopedCpuTensorHandle>(*(params.m_RecurrentToInputWeights));
        // In the VTS tests, cell-to-input weights may be null, even if the other CIFG params are not.
        if(params.m_CellToInputWeights != nullptr)
        {
            layer->m_CifgParameters.m_CellToInputWeights =
                    std::make_unique<ScopedCpuTensorHandle>(*(params.m_CellToInputWeights));
        }
        layer->m_CifgParameters.m_InputGateBias =
            std::make_unique<ScopedCpuTensorHandle>(*(params.m_InputGateBias));
    }

    //Lstm projection parameters
    if(descriptor.m_ProjectionEnabled)
    {
        if(params.m_ProjectionWeights == nullptr)
        {
            throw InvalidArgumentException("AddLstmLayer: Projection Weights cannot be NULL");
        }
        layer->m_ProjectionParameters.m_ProjectionWeights =
            std::make_unique<ScopedCpuTensorHandle>(*(params.m_ProjectionWeights));
        if(params.m_ProjectionBias != nullptr)
        {
            layer->m_ProjectionParameters.m_ProjectionBias =
                std::make_unique<ScopedCpuTensorHandle>(*(params.m_ProjectionBias));
        }
    }

    //Lstm Peephole params
    if(descriptor.m_PeepholeEnabled)
    {
        if(params.m_CellToForgetWeights == nullptr)
        {
            throw InvalidArgumentException("AddLstmLayer: Cell To Forget Weights cannot be NULL");
        }
        if(params.m_CellToOutputWeights == nullptr)
        {
            throw InvalidArgumentException("AddLstmLayer: Cell To Output Weights cannot be NULL");
        }
        layer->m_PeepholeParameters.m_CellToForgetWeights =
            std::make_unique<ScopedCpuTensorHandle>(*(params.m_CellToForgetWeights));
        layer->m_PeepholeParameters.m_CellToOutputWeights =
            std::make_unique<ScopedCpuTensorHandle>(*(params.m_CellToOutputWeights));
    }
    return layer;
}

IConnectableLayer* Network::AddDivisionLayer(const char* name)
{
    return m_Graph->AddLayer<DivisionLayer>(name);
}

IConnectableLayer* Network::AddSubtractionLayer(const char* name)
{
    return m_Graph->AddLayer<SubtractionLayer>(name);
}

IConnectableLayer* Network::AddMeanLayer(const MeanDescriptor& meanDescriptor, const char* name)
{
    return m_Graph->AddLayer<MeanLayer>(meanDescriptor,name);
}

IConnectableLayer* Network::AddPadLayer(const PadDescriptor& padDescriptor, const char* name)
{
    return m_Graph->AddLayer<PadLayer>(padDescriptor,name);
}

IConnectableLayer *Network::AddQuantizeLayer(const char *name)
{
    return m_Graph->AddLayer<QuantizeLayer>(name);
}

IConnectableLayer* Network::AddDequantizeLayer(const char* name)
{
    return m_Graph->AddLayer<DequantizeLayer>(name);
}

IConnectableLayer* Network::AddStridedSliceLayer(const StridedSliceDescriptor& stridedSliceDescriptor,
                                                 const char* name)
{
    return m_Graph->AddLayer<StridedSliceLayer>(stridedSliceDescriptor, name);
}

IConnectableLayer* Network::AddGreaterLayer(const char* name)
{
    return m_Graph->AddLayer<GreaterLayer>(name);
}

IConnectableLayer* Network::AddEqualLayer(const char* name)
{
    return m_Graph->AddLayer<EqualLayer>(name);
}

IConnectableLayer* Network::AddRsqrtLayer(const char * name)
{
    return m_Graph->AddLayer<RsqrtLayer>(name);
}

IConnectableLayer* Network::AddGatherLayer(const char* name)
{
    return m_Graph->AddLayer<GatherLayer>(name);
}

IConnectableLayer* Network::AddMergeLayer(const char* name)
{
    return m_Graph->AddLayer<MergeLayer>(name);
}

IConnectableLayer* Network::AddSwitchLayer(const char* name)
{
    return m_Graph->AddLayer<SwitchLayer>(name);
}

void Network::Accept(ILayerVisitor& visitor) const
{
    for (auto layer : GetGraph())
    {
        layer->Accept(visitor);
    };
}

OptimizedNetwork::OptimizedNetwork(std::unique_ptr<Graph> graph)
    : m_Graph(std::move(graph))
{
}

OptimizedNetwork::~OptimizedNetwork()
{
}

} // namespace armnn
