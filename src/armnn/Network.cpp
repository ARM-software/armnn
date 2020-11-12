//
// Copyright © 2017 Arm Ltd and Contributors. All rights reserved.
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
#include <armnn/backends/IBackendInternal.hpp>
#include <backendsCommon/TensorHandleFactoryRegistry.hpp>

#include <armnn/Exceptions.hpp>
#include <armnn/Utils.hpp>
#include <armnn/TypesUtils.hpp>
#include <armnn/BackendRegistry.hpp>
#include <armnn/Logging.hpp>
#include <armnn/utility/Assert.hpp>
#include <armnn/utility/IgnoreUnused.hpp>
#include <armnn/utility/PolymorphicDowncast.hpp>

#include <ProfilingService.hpp>

#include <fcntl.h>
#include <algorithm>
#include <fstream>
#include <memory>
#include <vector>
#include <algorithm>

namespace armnn
{

armnn::INetwork* INetwork::CreateRaw(NetworkOptions networkOptions)
{
    return new Network(networkOptions);
}

armnn::INetworkPtr INetwork::Create(NetworkOptions networkOptions)
{
    return INetworkPtr(CreateRaw(networkOptions), &INetwork::Destroy);
}

void INetwork::Destroy(INetwork* network)
{
    delete PolymorphicDowncast<Network*>(network);
}

void IOptimizedNetwork::Destroy(IOptimizedNetwork* network)
{
    delete PolymorphicDowncast<OptimizedNetwork*>(network);
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

void ReportError(const std::string& errorMessage,
                 Optional<std::vector<std::string>&> errorMessages)
{
    std::stringstream fullErrorMessage;
    fullErrorMessage << "ERROR: " << errorMessage;
    ARMNN_LOG(warning) << fullErrorMessage.str();
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
    ARMNN_LOG(warning) << fullWarningMessage.str();
    if (warningMessages)
    {
        warningMessages.value().push_back(fullWarningMessage.str());
    }
}

OptimizationResult ReturnWithError(OptimizationResult res,
                                   const Layer* layer,
                                   const BackendSettings& backendSettings,
                                   Optional<std::vector<std::string>&> errMessages)
{
    std::stringstream failureMsg;
    failureMsg << "Layer of type " << GetLayerTypeAsCString(layer->GetType())
               << " is not supported on any preferred backend " << backendSettings.m_PreferredBackends;
    ReportError(failureMsg.str(), errMessages);

    res.m_Error = true;
    return res;
}


bool CheckScaleSetOnQuantizedType(Layer* layer, Optional<std::vector<std::string>&> errMessages)
{
    bool noErrors = true;
    unsigned int numOutputs = layer->GetNumOutputSlots();
    for (unsigned int i = 0; i < numOutputs; i++) {
        OutputSlot& outputSlot = layer->GetOutputSlot(i);
        TensorInfo info = outputSlot.GetTensorInfo();
        if (DataType::QAsymmU8 == info.GetDataType()) {
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
                ARMNN_LOG(warning) << ss.str();
                info.SetQuantizationScale((1.0f /256.0f));
                info.SetQuantizationOffset(0);
                outputSlot.SetTensorInfo(info);
            }
        }
    }
    return noErrors;
}

template <typename LayerT>
LayerT* ConvertBf16ToFp32Weight(Layer* l)
{
    LayerT* layer = PolymorphicDowncast<LayerT*>(l);
    if ((layer->GetType() == LayerType::Convolution2d || layer->GetType() == LayerType::FullyConnected)
         && layer->m_Weight)
    {
        const TensorInfo& info = layer->m_Weight->GetTensorInfo();

        if (info.GetDataType() == DataType::BFloat16)
        {
            std::vector<float> newValues(info.GetNumElements());

            armnnUtils::FloatingPointConverter::ConvertBFloat16ToFloat32(
                layer->m_Weight->template GetTensor<armnn::BFloat16>(), info.GetNumElements(), newValues.data());

            TensorInfo newInfo(info.GetShape(), DataType::Float32);
            ConstTensor newInput(newInfo, newValues);
            layer->m_Weight.reset(new ScopedCpuTensorHandle(newInput));
        }
    }
    return layer;
}

OptimizationResult AttemptBackendAssignment(BackendSettings& backendSettings,
                                            Graph& graph,
                                            Layer* layer,
                                            BackendId backend,
                                            DataType dataTypeIn,
                                            DataType dataTypeOut,
                                            const std::vector<BackendId>& availablePreferredBackends,
                                            std::string& reasonIfUnsupported,
                                            Optional<std::vector<std::string>&> errMessages)
{
    OptimizationResult result;

    // Helper lambda to compose meaningful error message before returning with error
    auto ReturnError = [&](const Layer* layer)
        {
            return ReturnWithError(result, layer, backendSettings, errMessages);
        };

    // need to set the compute device on the layer
    // before we can check if it is supported
    layer->SetBackendId(backend);
    if (!IWorkloadFactory::IsLayerSupported(*layer, EmptyOptional(), reasonIfUnsupported))
    {
        if (dataTypeIn == DataType::Float16 || dataTypeOut == DataType::Float16)
        {
            if (IWorkloadFactory::IsLayerSupported(*layer, DataType::Float32, reasonIfUnsupported)
                && layer->GetType() != LayerType::ConvertFp32ToFp16
                && layer->GetType() != LayerType::ConvertFp16ToFp32)
            {
                // Insert FP16 -> FP32 conversion layer before current layer
                std::vector<ConvertFp16ToFp32Layer*> convertFp16ToFp32Layers;
                if (dataTypeIn == DataType::Float16)
                {
                    convertFp16ToFp32Layers =
                        InsertConvertFp16ToFp32LayersBefore(graph, *layer);
                }

                // Insert FP32 -> FP16 conversion layer after current layer
                std::vector<ConvertFp32ToFp16Layer*> convertFp32ToFp16Layers;
                if (dataTypeOut == DataType::Float16)
                {
                    convertFp32ToFp16Layers =
                        InsertConvertFp32ToFp16LayersAfter(graph, *layer);
                }

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
                        return ReturnError(convertLayer);
                    }
                }

                for (ConvertFp32ToFp16Layer* convertLayer : convertFp32ToFp16Layers)
                {
                    if (!AssignFirstSupportedBackend(convertLayer, backend))
                    {
                        return ReturnError(convertLayer);
                    }
                }

                return result;
            }
        }
        else if (dataTypeIn == DataType::BFloat16 || dataTypeOut == DataType::BFloat16)
        {
            if (IWorkloadFactory::IsLayerSupported(*layer, DataType::Float32, reasonIfUnsupported)
                && layer->GetType() != LayerType::ConvertFp32ToBf16
                && layer->GetType() != LayerType::ConvertBf16ToFp32)
            {
                // Insert BF16 -> FP32 conversion layer before current layer
                std::vector<ConvertBf16ToFp32Layer*> convertBf16ToFp32Layers;
                if (dataTypeIn == DataType::BFloat16)
                {
                    convertBf16ToFp32Layers =
                        InsertConvertBf16ToFp32LayersBefore(graph, *layer);
                    if (layer->GetType() == LayerType::Convolution2d)
                    {
                        ConvertBf16ToFp32Weight<Convolution2dLayer>(layer);
                    }
                    else if (layer->GetType() == LayerType::FullyConnected)
                    {
                        ConvertBf16ToFp32Weight<FullyConnectedLayer>(layer);
                    }
                }

                // Insert FP32 -> BF16 conversion layer after current layer
                std::vector<ConvertFp32ToBf16Layer*> convertFp32ToBf16Layers;
                if (dataTypeOut == DataType::BFloat16)
                {
                    convertFp32ToBf16Layers =
                        InsertConvertFp32ToBf16LayersAfter(graph, *layer);
                }

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

                for (ConvertBf16ToFp32Layer* convertLayer : convertBf16ToFp32Layers)
                {
                    if (!AssignFirstSupportedBackend(convertLayer, backend))
                    {
                        return ReturnError(convertLayer);
                    }
                }

                for (ConvertFp32ToBf16Layer* convertLayer : convertFp32ToBf16Layers)
                {
                    if (!AssignFirstSupportedBackend(convertLayer, backend))
                    {
                        return ReturnError(convertLayer);
                    }
                }

                return result;
            }
        }

        std::stringstream warningMsg;
        warningMsg << "Layer of type " << GetLayerTypeAsCString(layer->GetType())
                   << " is not supported on requested backend " << layer->GetBackendId().Get()
                   << " for input data type " << GetDataTypeName(dataTypeIn)
                   << " and output data type " << GetDataTypeName(dataTypeOut)
                   << " (reason: " << reasonIfUnsupported
                   << "), falling back to the next backend.";
        ReportWarning(warningMsg.str(), errMessages);

        return OptimizationResult(true, false);
    }
    else
    {
        return result;
    }
}


OptimizationResult AssignBackends(OptimizedNetwork* optNetObjPtr,
                                  BackendSettings& backendSettings,
                                  Graph::Iterator& firstLayer,
                                  Graph::Iterator& lastLayer,
                                  Optional<std::vector<std::string>&> errMessages)
{
    OptimizationResult result;

    // Helper lambda to compose meaningful error message before returning with error
    auto ReturnError = [&](const Layer* layer)
        {
            return ReturnWithError(result, layer, backendSettings, errMessages);
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

        DataType dataTypeIn  = layer->GetNumInputSlots() == 0 ? DataType::Float32 :
            layer->GetInputSlot(0).GetConnectedOutputSlot()->GetTensorInfo().GetDataType();
        DataType dataTypeOut = layer->GetNumOutputSlots() == 0 ? DataType::Float32 :
            layer->GetOutputSlot(0).GetTensorInfo().GetDataType();

        std::string reasonIfUnsupported;
        bool found = false;
        if (!CheckScaleSetOnQuantizedType(layer, errMessages))
        {
            // don't bomb immediately, find all the quantized outputs
            // which haven't had a scale set and report them all back.
            result.m_Error = true;
        }

        // First try assign layer to hint backend
        if (layer->GetBackendHint().has_value() &&
            backendSettings.IsBackendSupported(layer->GetBackendHint().value()) &&
            AttemptBackendAssignment(backendSettings,
                                     optNetObjPtr->GetGraph(),
                                     layer,
                                     layer->GetBackendHint().value(),
                                     dataTypeIn,
                                     dataTypeOut,
                                     availablePreferredBackends,
                                     reasonIfUnsupported,
                                     errMessages).IsOk())
        {
            found = true;
            backendSettings.m_SelectedBackends.insert(layer->GetBackendHint().value());
        }
        else
        {
            // Try assign layer to prefered list of backends
            for (const auto& backend : availablePreferredBackends)
            {
                if (layer->GetBackendHint().has_value() &&
                    layer->GetBackendHint().value() == backend)
                {
                    continue; //Don't re-test the backend hint
                }

                OptimizationResult res = AttemptBackendAssignment(backendSettings,
                                                                  optNetObjPtr->GetGraph(),
                                                                  layer,
                                                                  backend,
                                                                  dataTypeIn,
                                                                  dataTypeOut,
                                                                  availablePreferredBackends,
                                                                  reasonIfUnsupported,
                                                                  errMessages);

                if (res.IsOk())
                {
                    found = true;
                    backendSettings.m_SelectedBackends.insert(backend);
                    break;
                }
                else if (res.IsError())
                {
                   return res;  // Cannot continue.
                   // Note: we don't need to log the error as it would already
                   // be logged in AttemptBackendAssignment().
                }
                else
                {
                    ARMNN_ASSERT_MSG(res.IsWarningOnly(), "OptimizationResult in unexpected state.");
                }
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
                return ReturnError(layer);
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

BackendsMap CreateSupportedBackends(TensorHandleFactoryRegistry& handleFactoryRegistry,
                                    BackendSettings& backendSettings)
{
    BackendsMap backends;
    auto const& backendRegistry = BackendRegistryInstance();
    for (auto&& selectedBackend : backendSettings.m_SupportedBackends)
    {
        auto backendFactory = backendRegistry.GetFactory(selectedBackend);
        auto backendObjPtr = backendFactory();
        ARMNN_ASSERT(backendObjPtr);

        backendObjPtr->RegisterTensorHandleFactories(handleFactoryRegistry);

        backends[backendObjPtr->GetId()] = std::move(backendObjPtr);
    }

    return backends;
}

OptimizationResult ApplyBackendOptimizations(OptimizedNetwork* optNetObjPtr,
                                             BackendSettings& backendSettings,
                                             BackendsMap& backends,
                                             const ModelOptions& modelOptions,
                                             Optional<std::vector<std::string>&> errMessages)
{
    ARMNN_ASSERT(optNetObjPtr);

    OptimizationResult result;

    // Get the optimized graph
    Graph& optGraph = optNetObjPtr->GetGraph();

    // Run backend specific optimizations
    for (auto&& selectedBackend : backendSettings.m_SelectedBackends)
    {
        auto backendObjPtr = backends.find(selectedBackend)->second.get();
        ARMNN_ASSERT(backendObjPtr);

        // Select sub-graphs based on backend
        SubgraphViewSelector::Subgraphs subgraphs =
                SubgraphViewSelector::SelectSubgraphs(optGraph,
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
            OptimizationViews optimizationViews = backendObjPtr->OptimizeSubgraphView(*subgraph, modelOptions);
            ARMNN_ASSERT(optimizationViews.Validate(*subgraph));

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
                        ARMNN_ASSERT(l);
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
                    ReportWarning(subgraphMsg.str(), errMessages);

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

bool RequiresCopy(ITensorHandleFactory::FactoryId src,
                  ITensorHandleFactory::FactoryId dst,
                  TensorHandleFactoryRegistry& registry)
{
    if (src != dst)
    {
        ITensorHandleFactory* srcFactory = registry.GetFactory(src);
        ITensorHandleFactory* dstFactory = registry.GetFactory(dst);

        if (srcFactory && dstFactory &&
            (srcFactory->GetExportFlags() & dstFactory->GetImportFlags()) != 0)
        {
            return false;
        }
        return true;
    }
    return false;
}

// Find the handle factory for the input layer which results in fewest required copies.
ITensorHandleFactory::FactoryId CalculateSlotOptionForInput(BackendsMap& backends,
                                                            OutputSlot& slot,
                                                            TensorHandleFactoryRegistry& registry)
{
    Layer& layer = slot.GetOwningLayer();
    ARMNN_ASSERT(layer.GetType() == LayerType::Input);

    // Explicitly select the tensorhandle factory for InputLayer because the rules for it are slightly different. It
    // doesn't matter which backend it is assigned to because they all use the same implementation, which
    // requires Map/Unmap support. This means that, so long as the handle type supports map/unmap semantics, we can
    // select a factory with maximum compatibility with the layers connected to the InputLayer.

    // First ensure the from backends can support the TensorHandeAPI
    auto frmBackend = backends.find(layer.GetBackendId());
    if (frmBackend == backends.end() ||
        !frmBackend->second->SupportsTensorAllocatorAPI())
    {
        return ITensorHandleFactory::LegacyFactoryId;
    }

    // Go through all connections to the output slot and determine the TensorHandleFactory which results in the
    // fewest copies.
    std::map<ITensorHandleFactory::FactoryId, int> factoryScores;
    int topScore = 0;
    ITensorHandleFactory::FactoryId topChoice = ITensorHandleFactory::LegacyFactoryId;

    for (auto&& connection : slot.GetConnections())
    {
        const Layer& connectedLayer = connection->GetOwningLayer();

        auto toBackend = backends.find(connectedLayer.GetBackendId());
        ARMNN_ASSERT_MSG(toBackend != backends.end(), "Backend id not found for the connected layer");

        if (!toBackend->second.get()->SupportsTensorAllocatorAPI())
        {
            // The destination backend does not support the tensor allocator API, move to the next one
            continue;
        }

        auto dstPrefs = toBackend->second.get()->GetHandleFactoryPreferences();
        for (auto&& dst : dstPrefs)
        {
            // Input layers use the mem copy workload or import, so the selected factory must
            // support either the map/unmap API or Import API
            ITensorHandleFactory* factory = registry.GetFactory(dst);
            if (!factory->SupportsMapUnmap() &&
                !CheckFlag(factory->GetImportFlags(), MemorySource::Malloc)) // Just support cpu mem imports for now
            {
                // The current tensor handle factory does not support the map/unmap or import
                // strategy, move to the next one
                continue;
            }

            auto it = factoryScores.find(dst);
            if (it == factoryScores.end())
            {
                // Add new score to the table
                factoryScores[dst] = 0;
                if (topChoice == ITensorHandleFactory::LegacyFactoryId)
                {
                    topChoice = dst;
                }
            }
            else
            {
                // Increase the score
                factoryScores[dst]++;

                // Track the best option
                if (factoryScores[dst] > topScore)
                {
                    topScore = factoryScores[dst];
                    topChoice = dst;
                }
            }
        }
    }

    return topChoice;
}

// Find the handle factory for the output layer which results in fewest required copies.
ITensorHandleFactory::FactoryId CalculateSlotOptionForOutput(BackendsMap& backends,
                                                            OutputSlot& slot,
                                                            TensorHandleFactoryRegistry& registry)
{
    IgnoreUnused(backends, slot, registry);
    return ITensorHandleFactory::DeferredFactoryId;
}

// For all handle factories supported on the source backend, we wish to find the one which requires the fewest copies
// when considering all connections.
ITensorHandleFactory::FactoryId CalculateSlotOption(BackendsMap& backends,
                                                    OutputSlot& outputSlot,
                                                    TensorHandleFactoryRegistry& registry)
{
    // First ensure the from backends can support the TensorHandeAPI
    Layer& layer = outputSlot.GetOwningLayer();
    auto frmBackend = backends.find(layer.GetBackendId());
    if (frmBackend == backends.end() ||
        !frmBackend->second->SupportsTensorAllocatorAPI())
    {
        return ITensorHandleFactory::LegacyFactoryId;
    }

    // Connections to Output Layers requires support for map/unmap on the TensorHandle.
    bool requiresMapUnmap = false;
    for (auto&& connection : outputSlot.GetConnections())
    {
        const Layer& connectedLayer = connection->GetOwningLayer();
        if (connectedLayer.GetType() == LayerType::Output)
        {
            requiresMapUnmap = true;
        }
    }

    IBackendInternal* srcBackend = frmBackend->second.get();
    auto srcPrefs = srcBackend->GetHandleFactoryPreferences();

    // Initialize the scores
    std::map<ITensorHandleFactory::FactoryId, int> factoryScores;
    for (auto&& pref : srcPrefs)
    {
        if (requiresMapUnmap) // Only consider factories that support map/unmap if required
        {
            ITensorHandleFactory* factory = registry.GetFactory(pref);
            if (!factory->SupportsMapUnmap())
            {
                // The current tensor handle factory does not support the map/unmap strategy, move to the next one
                continue;
            }
        }

        auto it = factoryScores.find(pref);
        if (it == factoryScores.end())
        {
            // Add new score to the table
            factoryScores[pref] = 0;
        }
    }

    // Score each handle factory based on how many times it requires copies on the slot connections
    for (auto&& connection : outputSlot.GetConnections())
    {
        const Layer& connectedLayer = connection->GetOwningLayer();

        auto toBackend = backends.find(connectedLayer.GetBackendId());
        ARMNN_ASSERT_MSG(toBackend != backends.end(), "Backend id not found for the connected layer");

        auto dstPrefs = toBackend->second.get()->GetHandleFactoryPreferences();
        for (auto&& src : srcPrefs)
        {
            if (factoryScores.find(src) == factoryScores.end()) // Don't consider excluded factories
            {
                continue;
            }

            for (auto&& dst : dstPrefs)
            {
                if (RequiresCopy(src, dst, registry))
                {
                    // Copy avoided, increase the score
                    factoryScores[src]++;
                    break;
                }
            }
        }
    }

    // Find the lowest score
    int minScore = std::numeric_limits<int>::max();
    for (auto it : factoryScores)
    {
        minScore = std::min(minScore, it.second);
    }

    // Collect factories matching the best(lowest) score
    std::vector<ITensorHandleFactory::FactoryId> optimalFactories;
    for (auto it : factoryScores)
    {
        if (it.second == minScore)
        {
            optimalFactories.push_back(it.first);
        }
    }

    // For all compatible Factories matching the best score, find the preferred one for the current layer.
    for (auto&& srcPref : srcPrefs)
    {
        for (auto&& comp : optimalFactories)
        {
            if (comp == srcPref)
            {
                return comp;
            }
        }
    }

    return ITensorHandleFactory::LegacyFactoryId;
}

EdgeStrategy CalculateEdgeStrategy(BackendsMap& backends,
                                   ITensorHandleFactory::FactoryId srcFactoryId,
                                   const Layer& layer,
                                   const Layer& connectedLayer,
                                   TensorHandleFactoryRegistry& registry,
                                   bool importEnabled)
{
    auto toBackend = backends.find(connectedLayer.GetBackendId());
    ARMNN_ASSERT_MSG(toBackend != backends.end(), "Backend id not found for the connected layer");

    auto dstPrefs = toBackend->second.get()->GetHandleFactoryPreferences();

    // Legacy API check for backward compatibility
    if (srcFactoryId == ITensorHandleFactory::LegacyFactoryId || dstPrefs.empty())
    {
        if (layer.GetBackendId() != connectedLayer.GetBackendId())
        {
            return EdgeStrategy::CopyToTarget;
        }
        else
        {
            return EdgeStrategy::DirectCompatibility;
        }
    }

    // TensorHandleFactory API present, so perform more sophisticated strategies.
    // Dst Output layers don't require copy because they use import or map/unmap
    if (connectedLayer.GetType() == LayerType::Output)
    {
        return EdgeStrategy::DirectCompatibility;
    }

    // Search for direct match in prefs
    for (auto&& pref : dstPrefs)
    {
        if (pref == srcFactoryId)
        {
            return EdgeStrategy::DirectCompatibility;
        }
    }

    // Search for export/import options
    ITensorHandleFactory* srcFactory = registry.GetFactory(srcFactoryId);
    if (srcFactory->GetExportFlags() != 0 && importEnabled)
    {
        for (auto&& pref : dstPrefs)
        {
            ITensorHandleFactory* dstFactory = registry.GetFactory(pref);

            // Handles cases when a destPref is not listed in TensorHandleFactoryRegistry
            if (!dstFactory) {
                continue;
            }

            if ((dstFactory->GetImportFlags() & srcFactory->GetExportFlags()) != 0)
            {
                auto srcCapability = srcFactory->GetCapabilities(&layer, &layer, CapabilityClass::PaddingRequired);
                auto dstCapability = dstFactory->GetCapabilities(&connectedLayer,
                                                                 &connectedLayer,
                                                                 CapabilityClass::PaddingRequired);
                // Do not require memory copy if the source and destination do not require padding.
                if (srcCapability.empty() && dstCapability.empty())
                {
                    return EdgeStrategy::ExportToTarget;
                }
            }
        }
    }

    // Search for copy options via map/unmap
    if (srcFactory->SupportsMapUnmap())
    {
        for (auto&& pref : dstPrefs)
        {
            ITensorHandleFactory* dstFactory = registry.GetFactory(pref);
            if (dstFactory && dstFactory->SupportsMapUnmap())
            {
                return EdgeStrategy::CopyToTarget;
            }
        }
    }

    return EdgeStrategy::Undefined;
}

// Select the TensorHandleFactories and the corresponding memory strategy
OptimizationResult SelectTensorHandleStrategy(Graph& optGraph,
                                              BackendsMap& backends,
                                              TensorHandleFactoryRegistry& registry,
                                              bool importEnabled,
                                              Optional<std::vector<std::string>&> errMessages)
{
    OptimizationResult result;

    optGraph.ForEachLayer([&backends, &registry, &result, &errMessages, importEnabled](Layer* layer)
    {
        ARMNN_ASSERT(layer);

        // Lets make sure the backend is in our list of supported backends. Something went wrong during backend
        // assignment if this check fails
        ARMNN_ASSERT(backends.find(layer->GetBackendId()) != backends.end());

        // Check each output separately
        for (unsigned int slotIdx = 0; slotIdx < layer->GetNumOutputSlots(); slotIdx++)
        {
            OutputSlot& outputSlot = layer->GetOutputSlot(slotIdx);

            ITensorHandleFactory::FactoryId slotOption = ITensorHandleFactory::LegacyFactoryId;

            // Calculate the factory to use which results in the fewest copies being made.
            switch(layer->GetType())
            {
                case LayerType::Input:
                    slotOption = CalculateSlotOptionForInput(backends, outputSlot, registry);
                    break;
                case LayerType::Output:
                    slotOption = CalculateSlotOptionForOutput(backends, outputSlot, registry);
                    break;
                default:
                    slotOption = CalculateSlotOption(backends, outputSlot, registry);
                    break;
            }
            outputSlot.SetTensorHandleFactory(slotOption);

            // Now determine the "best" edge strategy for each connection given the slotOption.
            unsigned int connectionIdx = 0;
            for (auto&& connection : outputSlot.GetConnections())
            {
                const Layer& connectedLayer = connection->GetOwningLayer();

                EdgeStrategy strategy = CalculateEdgeStrategy(backends, slotOption, *layer, connectedLayer,
                                                              registry, importEnabled);

                if (strategy == EdgeStrategy::Undefined)
                {
                    result.m_Error = true;
                    if (errMessages)
                    {
                        errMessages.value().emplace_back("Could not find valid strategy required for compatibility"
                                                         " between backends.");
                    }
                    return;
                }

                outputSlot.SetEdgeStrategy(connectionIdx, strategy);

                connectionIdx++;
            }
        }
    });

    return result;
}

IOptimizedNetworkPtr Optimize(const INetwork& inNetwork,
                              const std::vector<BackendId>& backendPreferences,
                              const IDeviceSpec& deviceSpec,
                              const OptimizerOptions& options,
                              Optional<std::vector<std::string>&> messages)
{
    if (backendPreferences.empty())
    {
        throw InvalidArgumentException("Invoked Optimize with no backends specified");
    }

    if (options.m_ReduceFp32ToFp16 && options.m_ReduceFp32ToBf16)
    {
        throw InvalidArgumentException("BFloat16 and Float16 optimization cannot be enabled at the same time.");
    }

    const Network& network = *PolymorphicDowncast<const Network*>(&inNetwork);
    std::unique_ptr<Graph> graph = std::make_unique<Graph>(network.GetGraph());

    auto optNet = IOptimizedNetworkPtr(new OptimizedNetwork(std::move(graph), options.m_ModelOptions),
                                       &IOptimizedNetwork::Destroy);

    OptimizedNetwork* optNetObjPtr = PolymorphicDowncast<OptimizedNetwork*>(optNet.get());

    // Get the optimized graph
    Graph& optGraph = optNetObjPtr->GetGraph();

    // Perform AddBroadcastReshapeLayer optimisation
    using namespace optimizations;
    Optimizer::Pass(optGraph, MakeOptimizations(AddBroadcastReshapeLayer()));

    // Infer the tensor infos for all output slots. Throws an exception on failure
    optGraph.InferTensorInfos();

    // Perform optimisation passes
    Optimizer::Pass(optGraph, MakeOptimizations(SquashEqualPermuteSiblings(),
                                                SquashEqualTransposeSiblings(),
                                                SquashEqualReshapeSiblings(),
                                                OptimizeInversePermutes(),
                                                OptimizeInverseTransposes(),
                                                MovePermuteUp(),
                                                MoveTransposeUp(),
                                                PermuteAsReshape(),
                                                TransposeAsReshape(),
                                                OptimizeConsecutiveReshapes(),
                                                FoldPadIntoConvolution2d(),
                                                PermuteAndBatchToSpaceAsDepthToSpace(),
                                                TransposeAndBatchToSpaceAsDepthToSpace(),
                                                FuseBatchNormIntoConvolution2DFloat32(),
                                                FuseBatchNormIntoConvolution2DFloat16(),
                                                FuseBatchNormIntoDepthwiseConvolution2DFloat32(),
                                                FuseBatchNormIntoDepthwiseConvolution2DFloat16()));

    // If Fp32 to Fp16 optimization is set convert Fp32 network to Fp16
    if (options.m_ReduceFp32ToFp16)
    {
        Optimizer::Pass(optGraph, MakeOptimizations(Fp32NetworkToFp16Converter()));
        Optimizer::Pass(optGraph, MakeOptimizations(ConvertConstantsFloatToHalf()));
    }

    // If Fp32 to Bf16 optimization is set convert Fp32 network to Bf16
    // Convert input of Convolution2d and FullyConnected from Fp32 to Bf16
    // Only Constant weight of Convolution2d and FullyConnected are converted from Fp32 to Bf16
    if (options.m_ReduceFp32ToBf16)
    {
        Optimizer::Pass(optGraph, MakeOptimizations(Fp32NetworkToBf16Converter()));
    }

    // Initialize backend settings
    BackendSettings backendSettings(backendPreferences, deviceSpec);
    if (backendSettings.GetAvailablePreferredBackends().empty())
    {
        std::stringstream failureMsg;
        failureMsg << "None of the preferred backends " << backendPreferences
                   << " are supported. Current platform provides " << backendSettings.m_SupportedBackends;
        ReportError(failureMsg.str(), messages);
        throw InvalidArgumentException(failureMsg.str());
    }

    // Create a map to temporarily hold initialized backend objects
    TensorHandleFactoryRegistry tensorHandleFactoryRegistry;
    BackendsMap backends = CreateSupportedBackends(tensorHandleFactoryRegistry, backendSettings);

    // Assign an available backend to each layer
    Graph::Iterator firstLayer = optGraph.begin();
    Graph::Iterator lastLayer  = optGraph.end();
    OptimizationResult assignBackendsResult = AssignBackends(optNetObjPtr,
                                                             backendSettings,
                                                             firstLayer,
                                                             lastLayer,
                                                             messages);
    if (assignBackendsResult.m_Error)
    {
        // Failed to assign a backend to each layer
        throw InvalidArgumentException("Failed to assign a backend to each layer");
    }

    Optimizer::Pass(optGraph, MakeOptimizations(OptimizeInverseConversionsFp16(),
                                                OptimizeInverseConversionsFp32()));

    // Apply the backend-specific optimizations
    OptimizationResult backendOptimizationResult = ApplyBackendOptimizations(optNetObjPtr,
                                                                             backendSettings,
                                                                             backends,
                                                                             options.m_ModelOptions,
                                                                             messages);
    if (backendOptimizationResult.m_Error)
    {
        // Failed to apply the backend-specific optimizations
        throw InvalidArgumentException("Failed to apply the backend-specific optimizations");
    }

    // If the debug flag is set, then insert a DebugLayer after each layer
    // Doing this after applying the backend optimizations as they might have changed some layers
    if (options.m_Debug)
    {
        Optimizer::Pass(optGraph, MakeOptimizations(InsertDebugLayer()));
    }

    // Calculate the compatibility strategies for tensor handles
    OptimizationResult strategyResult = SelectTensorHandleStrategy(optGraph,
                                                                   backends,
                                                                   tensorHandleFactoryRegistry,
                                                                   options.m_ImportEnabled,
                                                                   messages);
    if (strategyResult.m_Error)
    {
        // Failed to apply the backend-specific optimizations
        return IOptimizedNetworkPtr(nullptr, &IOptimizedNetwork::Destroy);
    }

    // Based on the tensor handle strategy determined above, insert copy layers where required.
    optGraph.AddCompatibilityLayers(backends, tensorHandleFactoryRegistry);

    // Convert constants
    Optimizer::Pass(optGraph, MakeOptimizations(ConvertConstantsFloatToHalf()));
    Optimizer::Pass(optGraph, MakeOptimizations(ConvertConstantsHalfToFloat()));

    // Run backend specific optimizations (deprecated)
    for (auto&& chosenBackend : backendSettings.m_SelectedBackends)
    {
        auto factoryFun = BackendRegistryInstance().GetFactory(chosenBackend);
        auto backendPtr = factoryFun();
        ARMNN_ASSERT(backendPtr.get() != nullptr);

        ARMNN_NO_DEPRECATE_WARN_BEGIN
        auto backendSpecificOptimizations = backendPtr->GetOptimizations();
        ARMNN_NO_DEPRECATE_WARN_END

        if (!backendSpecificOptimizations.empty())
        {
            Optimizer::Pass(optNetObjPtr->GetGraph(), backendSpecificOptimizations);
        }
    }

    return optNet;
}
bool Network::GetShapeInferenceMethod()
{
    if (m_NetworkOptions.size() > 0 && m_NetworkOptions[0].GetBackendId().Get() == "ShapeInferenceMethod")
    {
        return m_NetworkOptions[0].GetOption(0).GetValue().AsBool();
    }

    return false;
}
Network::Network(NetworkOptions networkOptions)
: m_NetworkOptions(networkOptions),
  m_Graph(std::make_unique<Graph>(GetShapeInferenceMethod()))
{}

Network::~Network()
{
}

Status Network::PrintGraph()
{
    m_Graph->Print();
    return Status::Success;
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

IConnectableLayer* Network::AddComparisonLayer(const ComparisonDescriptor& comparisonDescriptor,
                                               const char* name)
{
    return m_Graph->AddLayer<ComparisonLayer>(comparisonDescriptor, name);
}

IConnectableLayer* Network::AddElementwiseUnaryLayer(const ElementwiseUnaryDescriptor& elementwiseUnaryDescriptor,
                                                     const char* name)
{
    return m_Graph->AddLayer<ElementwiseUnaryLayer>(elementwiseUnaryDescriptor, name);
}

IConnectableLayer* Network::AddFillLayer(const FillDescriptor& fillDescriptor,
                                         const char* name)
{
    return m_Graph->AddLayer<FillLayer>(fillDescriptor, name);
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

IConnectableLayer* Network::AddConcatLayer(const ConcatDescriptor& concatDescriptor,
                                           const char* name)
{
    return m_Graph->AddLayer<ConcatLayer>(concatDescriptor, name);
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

IConnectableLayer* Network::AddDepthToSpaceLayer(const DepthToSpaceDescriptor& depthToSpaceDescriptor,
                                                 const char* name)
{
    return m_Graph->AddLayer<DepthToSpaceLayer>(depthToSpaceDescriptor, name);
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

IConnectableLayer* Network::AddArgMinMaxLayer(const ArgMinMaxDescriptor& argMinMaxDescriptor,
                                              const char* name)
{
    return m_Graph->AddLayer<ArgMinMaxLayer>(argMinMaxDescriptor, name);
}

IConnectableLayer* Network::AddNormalizationLayer(const NormalizationDescriptor&
normalizationDescriptor,
    const char* name)
{
    return m_Graph->AddLayer<NormalizationLayer>(normalizationDescriptor, name);
}

IConnectableLayer* Network::AddSliceLayer(const SliceDescriptor& sliceDescriptor, const char* name)
{
    return m_Graph->AddLayer<SliceLayer>(sliceDescriptor, name);
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

IConnectableLayer* Network::AddMergerLayer(const MergerDescriptor& mergerDescriptor,
                                           const char* name)
{
    return AddConcatLayer(mergerDescriptor, name);
}

IConnectableLayer* Network::AddAbsLayer(const char * name)
{
    return AddElementwiseUnaryLayer(ElementwiseUnaryDescriptor(UnaryOperation::Abs), name);
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

IConnectableLayer* Network::AddRankLayer(const char* name)
{
    return m_Graph->AddLayer<RankLayer>(name);
}

IConnectableLayer* Network::AddResizeBilinearLayer(const ResizeBilinearDescriptor& descriptor,
                                                   const char* name)
{
    ResizeDescriptor resizeDescriptor;
    resizeDescriptor.m_Method           = ResizeMethod::Bilinear;
    resizeDescriptor.m_DataLayout       = descriptor.m_DataLayout;
    resizeDescriptor.m_TargetWidth      = descriptor.m_TargetWidth;
    resizeDescriptor.m_TargetHeight     = descriptor.m_TargetHeight;
    resizeDescriptor.m_AlignCorners     = descriptor.m_AlignCorners;
    resizeDescriptor.m_HalfPixelCenters = descriptor.m_HalfPixelCenters;

    return m_Graph->AddLayer<ResizeLayer>(resizeDescriptor, name);
}

IConnectableLayer* Network::AddResizeLayer(const ResizeDescriptor&
resizeDescriptor, const char* name)
{
    return m_Graph->AddLayer<ResizeLayer>(resizeDescriptor, name);
}

IConnectableLayer* Network::AddInstanceNormalizationLayer(const InstanceNormalizationDescriptor& desc,
                                                          const char* name)
{
    return m_Graph->AddLayer<InstanceNormalizationLayer>(desc, name);
}

IConnectableLayer* Network::AddL2NormalizationLayer(const L2NormalizationDescriptor& desc,
                                                    const char* name)
{
    return m_Graph->AddLayer<L2NormalizationLayer>(desc, name);
}

IConnectableLayer* Network::AddLogSoftmaxLayer(const LogSoftmaxDescriptor& desc,
                                               const char* name)
{
    return m_Graph->AddLayer<LogSoftmaxLayer>(desc, name);
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

IConnectableLayer* Network::AddSpaceToDepthLayer(const SpaceToDepthDescriptor& spaceToDepthDescriptor,
                                                 const char* name)
{
    return m_Graph->AddLayer<SpaceToDepthLayer>(spaceToDepthDescriptor, name);
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
            throw InvalidArgumentException("AddLstmLayer: Input To Input Weights cannot be NULL "
                                           "when CIFG is disabled.");
        }
        if(params.m_RecurrentToInputWeights == nullptr)
        {
            throw InvalidArgumentException(
                    "AddLstmLayer: Recurrent To Input Weights cannot be NULL "
                    "when CIFG is disabled.");
        }
        if(params.m_InputGateBias == nullptr)
        {
            throw InvalidArgumentException("AddLstmLayer: Input Gate Bias cannot be NULL "
                                           "when CIFG is disabled.");
        }
        layer->m_CifgParameters.m_InputToInputWeights =
            std::make_unique<ScopedCpuTensorHandle>(*(params.m_InputToInputWeights));
        layer->m_CifgParameters.m_RecurrentToInputWeights =
            std::make_unique<ScopedCpuTensorHandle>(*(params.m_RecurrentToInputWeights));
        layer->m_CifgParameters.m_InputGateBias =
            std::make_unique<ScopedCpuTensorHandle>(*(params.m_InputGateBias));
    }

    //Lstm projection parameters
    if(descriptor.m_ProjectionEnabled)
    {
        if(params.m_ProjectionWeights == nullptr)
        {
            throw InvalidArgumentException("AddLstmLayer: Projection Weights cannot be NULL "
                                           "when projection is enabled.");
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
        if(!descriptor.m_CifgEnabled)
        {
            if(params.m_CellToInputWeights == nullptr)
            {
                throw InvalidArgumentException("AddLstmLayer: Cell To Input Weights cannot be NULL "
                                               "when Peephole is enabled and CIFG disabled.");
            }

            layer->m_PeepholeParameters.m_CellToInputWeights =
                std::make_unique<ScopedCpuTensorHandle>(*(params.m_CellToInputWeights));
        }

        if(params.m_CellToForgetWeights == nullptr)
        {
            throw InvalidArgumentException("AddLstmLayer: Cell To Forget Weights cannot be NULL "
                                           "when Peephole is enabled.");
        }
        if(params.m_CellToOutputWeights == nullptr)
        {
            throw InvalidArgumentException("AddLstmLayer: Cell To Output Weights cannot be NULL "
                                           "when Peephole is enabled.");
        }

        layer->m_PeepholeParameters.m_CellToForgetWeights =
            std::make_unique<ScopedCpuTensorHandle>(*(params.m_CellToForgetWeights));
        layer->m_PeepholeParameters.m_CellToOutputWeights =
            std::make_unique<ScopedCpuTensorHandle>(*(params.m_CellToOutputWeights));
    }

    //Lstm Layer Normalization params
    if(descriptor.m_LayerNormEnabled)
    {
        if(!descriptor.m_CifgEnabled)
        {
            if(params.m_InputLayerNormWeights == nullptr)
            {
                throw InvalidArgumentException("AddLstmLayer: Input layer normalization weights cannot be NULL "
                                               "when layer normalization is enabled and CIFG disabled.");
            }
            layer->m_LayerNormParameters.m_InputLayerNormWeights =
                    std::make_unique<ScopedCpuTensorHandle>(*(params.m_InputLayerNormWeights));
        }

        if(params.m_ForgetLayerNormWeights == nullptr)
        {
            throw InvalidArgumentException("AddLstmLayer: Forget layer normalization weights cannot be NULL "
                                           "when layer normalization is enabled.");
        }
        if(params.m_CellLayerNormWeights == nullptr)
        {
            throw InvalidArgumentException("AddLstmLayer: Cell layer normalization weights cannot be NULL "
                                           "when layer normalization is enabled.");
        }
        if(params.m_OutputLayerNormWeights == nullptr)
        {
            throw InvalidArgumentException("AddLstmLayer: Output layer normalization weights cannot be NULL "
                                           "when layer normalization is enabled.");
        }
        layer->m_LayerNormParameters.m_ForgetLayerNormWeights =
                std::make_unique<ScopedCpuTensorHandle>(*(params.m_ForgetLayerNormWeights));
        layer->m_LayerNormParameters.m_CellLayerNormWeights =
                std::make_unique<ScopedCpuTensorHandle>(*(params.m_CellLayerNormWeights));
        layer->m_LayerNormParameters.m_OutputLayerNormWeights =
                std::make_unique<ScopedCpuTensorHandle>(*(params.m_OutputLayerNormWeights));
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
    return AddComparisonLayer(ComparisonDescriptor(ComparisonOperation::Greater), name);
}

IConnectableLayer* Network::AddEqualLayer(const char* name)
{
    return AddComparisonLayer(ComparisonDescriptor(ComparisonOperation::Equal), name);
}

IConnectableLayer* Network::AddRsqrtLayer(const char * name)
{
    return AddElementwiseUnaryLayer(ElementwiseUnaryDescriptor(UnaryOperation::Rsqrt), name);
}

IConnectableLayer* Network::AddGatherLayer(const char* name)
{
    GatherDescriptor gatherDescriptor{};
    return AddGatherLayer(gatherDescriptor, name);
}

IConnectableLayer* Network::AddGatherLayer(const GatherDescriptor& gatherDescriptor,
                                           const char* name)
{
    return m_Graph->AddLayer<GatherLayer>(gatherDescriptor, name);
}

IConnectableLayer* Network::AddMergeLayer(const char* name)
{
    return m_Graph->AddLayer<MergeLayer>(name);
}

IConnectableLayer* Network::AddSwitchLayer(const char* name)
{
    return m_Graph->AddLayer<SwitchLayer>(name);
}

IConnectableLayer* Network::AddPreluLayer(const char* name)
{
    return m_Graph->AddLayer<PreluLayer>(name);
}

IConnectableLayer* Network::AddTransposeConvolution2dLayer(const TransposeConvolution2dDescriptor& descriptor,
                                                           const ConstTensor& weights,
                                                           const Optional<ConstTensor>& biases,
                                                           const char* name)
{
    if (descriptor.m_BiasEnabled && !biases.has_value())
    {
        throw InvalidArgumentException("AddTransposeConvolution2dLayer: Biases cannot be empty");
    }

    const auto layer = m_Graph->AddLayer<TransposeConvolution2dLayer>(descriptor, name);

    layer->m_Weight = std::make_unique<ScopedCpuTensorHandle>(weights);

    if (descriptor.m_BiasEnabled)
    {
        layer->m_Bias = std::make_unique<ScopedCpuTensorHandle>(biases.value());
    }

    return layer;
}

IConnectableLayer* Network::AddTransposeLayer(const TransposeDescriptor& transposeDescriptor,
                                              const char* name)
{
    return m_Graph->AddLayer<TransposeLayer>(transposeDescriptor, name);
}

IConnectableLayer* Network::AddStackLayer(const StackDescriptor& stackDescriptor,
                                          const char* name)
{
    return m_Graph->AddLayer<StackLayer>(stackDescriptor, name);
}


IConnectableLayer* Network::AddStandInLayer(const StandInDescriptor& desc,
                                            const char* name)
{
    return m_Graph->AddLayer<StandInLayer>(desc, name);
}

IConnectableLayer* Network::AddQuantizedLstmLayer(const QuantizedLstmInputParams& params,
                                                  const char* name)
{
    const auto layer = m_Graph->AddLayer<QuantizedLstmLayer>(name);

    // InputToX weights
    layer->m_QuantizedLstmParameters.m_InputToInputWeights =
            std::make_unique<ScopedCpuTensorHandle>(params.GetInputToInputWeights());
    layer->m_QuantizedLstmParameters.m_InputToForgetWeights =
            std::make_unique<ScopedCpuTensorHandle>(params.GetInputToForgetWeights());
    layer->m_QuantizedLstmParameters.m_InputToCellWeights =
            std::make_unique<ScopedCpuTensorHandle>(params.GetInputToCellWeights());
    layer->m_QuantizedLstmParameters.m_InputToOutputWeights =
            std::make_unique<ScopedCpuTensorHandle>(params.GetInputToOutputWeights());

    // RecurrentToX weights
    layer->m_QuantizedLstmParameters.m_RecurrentToInputWeights =
            std::make_unique<ScopedCpuTensorHandle>(params.GetRecurrentToInputWeights());
    layer->m_QuantizedLstmParameters.m_RecurrentToForgetWeights =
            std::make_unique<ScopedCpuTensorHandle>(params.GetRecurrentToForgetWeights());
    layer->m_QuantizedLstmParameters.m_RecurrentToCellWeights =
            std::make_unique<ScopedCpuTensorHandle>(params.GetRecurrentToCellWeights());
    layer->m_QuantizedLstmParameters.m_RecurrentToOutputWeights =
            std::make_unique<ScopedCpuTensorHandle>(params.GetRecurrentToOutputWeights());

    // Bias
    layer->m_QuantizedLstmParameters.m_InputGateBias =
            std::make_unique<ScopedCpuTensorHandle>(params.GetInputGateBias());
    layer->m_QuantizedLstmParameters.m_ForgetGateBias =
            std::make_unique<ScopedCpuTensorHandle>(params.GetForgetGateBias());
    layer->m_QuantizedLstmParameters.m_CellBias =
            std::make_unique<ScopedCpuTensorHandle>(params.GetCellBias());
    layer->m_QuantizedLstmParameters.m_OutputGateBias =
            std::make_unique<ScopedCpuTensorHandle>(params.GetOutputGateBias());

    return layer;
}

IConnectableLayer* Network::AddQLstmLayer(const QLstmDescriptor&  descriptor,
                                          const LstmInputParams& params,
                                          const char* name)
{
    const auto layer = m_Graph->AddLayer<QLstmLayer>(descriptor, name);

    // QLstm Basic Parameters
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

    // QLstm Cifg parameters
    if(!descriptor.m_CifgEnabled)
    {
        if(params.m_InputToInputWeights == nullptr)
        {
            throw InvalidArgumentException("AddQLstmLayer: Input To Input Weights cannot be NULL");
        }

        if(params.m_RecurrentToInputWeights == nullptr)
        {
            throw InvalidArgumentException(
                    "AddQLstmLayer: Recurrent To Input Weights cannot be NULL");
        }

        if(params.m_InputGateBias == nullptr)
        {
            throw InvalidArgumentException("AddQLstmLayer: Input Gate Bias cannot be NULL");
        }

        layer->m_CifgParameters.m_InputToInputWeights =
                std::make_unique<ScopedCpuTensorHandle>(*(params.m_InputToInputWeights));
        layer->m_CifgParameters.m_RecurrentToInputWeights =
                std::make_unique<ScopedCpuTensorHandle>(*(params.m_RecurrentToInputWeights));
        layer->m_CifgParameters.m_InputGateBias =
                std::make_unique<ScopedCpuTensorHandle>(*(params.m_InputGateBias));
    }

    // QLstm Projection parameters
    if(descriptor.m_ProjectionEnabled)
    {
        if(params.m_ProjectionWeights == nullptr)
        {
            throw InvalidArgumentException("AddQLstmLayer: Projection Weights cannot be NULL");
        }

        layer->m_ProjectionParameters.m_ProjectionWeights =
                std::make_unique<ScopedCpuTensorHandle>(*(params.m_ProjectionWeights));

        // Projection bias is optional even if projection is enabled
        if(params.m_ProjectionWeights != nullptr)
        {
            layer->m_ProjectionParameters.m_ProjectionBias =
                    std::make_unique<ScopedCpuTensorHandle>(*(params.m_ProjectionBias));
        }

    }

    // QLstm Peephole params
    if(descriptor.m_PeepholeEnabled)
    {
        if(params.m_CellToForgetWeights == nullptr)
        {
            throw InvalidArgumentException("AddQLstmLayer: Cell To Forget Weights cannot be NULL");
        }

        if(params.m_CellToOutputWeights == nullptr)
        {
            throw InvalidArgumentException("AddQLstmLayer: Cell To Output Weights cannot be NULL");
        }

        if(!descriptor.m_CifgEnabled)
        {
            if(params.m_CellToInputWeights == nullptr)
            {
                throw InvalidArgumentException("AddQLstmLayer: Cell To Input Weights cannot be NULL");
            }

            layer->m_PeepholeParameters.m_CellToInputWeights =
                    std::make_unique<ScopedCpuTensorHandle>(*(params.m_CellToInputWeights));
        }

        layer->m_PeepholeParameters.m_CellToForgetWeights =
                std::make_unique<ScopedCpuTensorHandle>(*(params.m_CellToForgetWeights));
        layer->m_PeepholeParameters.m_CellToOutputWeights =
                std::make_unique<ScopedCpuTensorHandle>(*(params.m_CellToOutputWeights));
    }

    // QLstm Layer Normalization params
    if(descriptor.m_LayerNormEnabled)
    {
        if(params.m_ForgetLayerNormWeights == nullptr)
        {
            throw InvalidArgumentException("AddQLstmLayer: Forget layer normalization weights cannot be NULL");
        }

        if(params.m_CellLayerNormWeights == nullptr)
        {
            throw InvalidArgumentException("AddQLstmLayer: Cell layer normalization weights cannot be NULL");
        }

        if(params.m_OutputLayerNormWeights == nullptr)
        {
            throw InvalidArgumentException("AddQLstmLayer: Output layer normalization weights cannot be NULL");
        }

        if(!descriptor.m_CifgEnabled)
        {
            if(params.m_InputLayerNormWeights == nullptr)
            {
                throw InvalidArgumentException("AddQLstmLayer: Input layer normalization weights cannot be NULL");
            }

            layer->m_LayerNormParameters.m_InputLayerNormWeights =
                    std::make_unique<ScopedCpuTensorHandle>(*(params.m_InputLayerNormWeights));
        }

        layer->m_LayerNormParameters.m_ForgetLayerNormWeights =
                std::make_unique<ScopedCpuTensorHandle>(*(params.m_ForgetLayerNormWeights));
        layer->m_LayerNormParameters.m_CellLayerNormWeights =
                std::make_unique<ScopedCpuTensorHandle>(*(params.m_CellLayerNormWeights));
        layer->m_LayerNormParameters.m_OutputLayerNormWeights =
                std::make_unique<ScopedCpuTensorHandle>(*(params.m_OutputLayerNormWeights));
    }
    return layer;
}

IConnectableLayer* Network::AddLogicalBinaryLayer(const LogicalBinaryDescriptor& logicalBinaryDescriptor,
                                                  const char* name)
{
    return m_Graph->AddLayer<LogicalBinaryLayer>(logicalBinaryDescriptor, name);
}

void Network::Accept(ILayerVisitor& visitor) const
{
    for (auto layer : GetGraph())
    {
        layer->Accept(visitor);
    };
}

OptimizedNetwork::OptimizedNetwork(std::unique_ptr<Graph> graph)
    : m_Graph(std::move(graph)), m_Guid(profiling::ProfilingService::GetNextGuid())
{
}

OptimizedNetwork::OptimizedNetwork(std::unique_ptr<Graph> graph, const ModelOptions& modelOptions)
    : m_Graph(std::move(graph)), m_Guid(profiling::ProfilingService::GetNextGuid()), m_ModelOptions(modelOptions)
{
}

OptimizedNetwork::~OptimizedNetwork()
{
}

} // namespace armnn
