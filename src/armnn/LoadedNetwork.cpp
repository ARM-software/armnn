//
// Copyright © 2017-2024 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "LoadedNetwork.hpp"
#include "Layer.hpp"
#include "Graph.hpp"
#include "Markers.hpp"
#include "Profiling.hpp"
#include "HeapProfiling.hpp"

#include <armnn/BackendHelper.hpp>
#include <armnn/BackendRegistry.hpp>
#include <armnn/Logging.hpp>

#include <armnn/backends/TensorHandle.hpp>
#include <armnn/backends/IBackendInternal.hpp>
#include <armnn/backends/IMemoryManager.hpp>
#include <armnn/backends/MemCopyWorkload.hpp>

#include <armnn/profiling/ArmNNProfiling.hpp>

#include <backendsCommon/MemSyncWorkload.hpp>

#include <common/include/Processes.hpp>

#include <fmt/format.h>

namespace armnn
{

using namespace std;
using namespace arm::pipe;

namespace
{

template <typename ExceptionType>
std::string ToErrorMessage(const char * prefix, const ExceptionType & error)
{
    std::stringstream ss;
    ss << prefix << " " << error.what();
    return ss.str();
}

void AddLayerStructure(std::unique_ptr<TimelineUtilityMethods>& timelineUtils,
                       const Layer& layer,
                       ProfilingGuid networkGuid)
{
    // Add layer to the post-optimisation network structure
    std::string layerName = layer.GetNameStr().empty() ? "<Unnamed>" : layer.GetNameStr();
    timelineUtils->CreateNamedTypedChildEntity(layer.GetGuid(),
                                               networkGuid,
                                               layerName,
                                               LabelsAndEventClasses::LAYER_GUID);
    for (auto&& input : layer.GetInputSlots())
    {
        const IOutputSlot* source = input.GetConnectedOutputSlot();
        if (!source)
        {
            throw armnn::NullPointerException("Null source found on input to layer \"" + layerName + "\".");
        }
        timelineUtils->CreateConnectionRelationship(ProfilingRelationshipType::RetentionLink,
                                                    source->GetOwningLayerGuid(),
                                                    layer.GetGuid());
    }
}

void AddWorkloadStructure(std::unique_ptr<TimelineUtilityMethods>& timelineUtils,
                          std::unique_ptr<IWorkload>& workload,
                          const Layer& layer)
{
    // Add workload to the post-optimisation network structure
    timelineUtils->CreateTypedEntity(workload->GetGuid(), LabelsAndEventClasses::WORKLOAD_GUID);
    timelineUtils->MarkEntityWithLabel(workload->GetGuid(),
                                       layer.GetBackendId().Get(),
                                       LabelsAndEventClasses::BACKENDID_GUID);

    // Link the workload to the layer
    timelineUtils->CreateRelationship(ProfilingRelationshipType::RetentionLink,
                                      layer.GetGuid(),
                                      workload->GetGuid(),
                                      LabelsAndEventClasses::CHILD_GUID);
}

} // anonymous

/**
 * This function performs a sanity check to ensure that the combination of input and output memory source matches the
 * values for importEnabled and exportEnabled that were specified during optimization. During optimization the tensor
 * handle factories are chosen based on whether import and export are enabled. If the user then specifies something
 * incompatible here it can lead to problems.
 *
 * @param optimizedOptions
 * @param networkProperties
 */
void ValidateSourcesMatchOptimizedNetwork(std::vector<BackendOptions> optimizedOptions,
                                          const INetworkProperties& networkProperties)
{
    // Find the "Global" backend options. During the optimize phase the values of importEnabled and exportEnabled are
    // added as backend options.
    const vector<BackendOptions>::iterator& backendItr =
        find_if(optimizedOptions.begin(), optimizedOptions.end(), [](const BackendOptions& backend) {
            if (backend.GetBackendId().Get() == "Global")
            {
                return true;
            }
            else
            {
                return false;
            }
        });
    bool importEnabled = false;
    bool exportEnabled = false;
    if (backendItr != optimizedOptions.end())
    {
        // Find the importEnabled and exportEnabled values.
        for (size_t i = 0; i < backendItr->GetOptionCount(); i++)
        {
            const BackendOptions::BackendOption& option = backendItr->GetOption(i);
            if (option.GetName() == "ImportEnabled")
            {
                importEnabled = option.GetValue().AsBool();
            }
            if (option.GetName() == "ExportEnabled")
            {
                exportEnabled = option.GetValue().AsBool();
            }
        }
    }

    // Now that we have values for import and export compare them to the MemorySource variables.
    // Any value of MemorySource that's not "Undefined" implies that we need to do an import of some kind.
    if ((networkProperties.m_InputSource == MemorySource::Undefined && importEnabled) ||
        (networkProperties.m_InputSource != MemorySource::Undefined && !importEnabled))
    {
        auto message = fmt::format("The input memory source specified, '{0}',", networkProperties.m_InputSource);
        if (!importEnabled)
        {
            message.append(" requires that memory import be enabled. However, "
                           "it was disabled when this network was optimized.");
        }
        else
        {
            message.append(" requires that memory import be disabled. However, "
                           "it was enabled when this network was optimized.");
        }
        throw InvalidArgumentException(message);
    }

    if ((networkProperties.m_OutputSource == MemorySource::Undefined && exportEnabled) ||
        (networkProperties.m_OutputSource != MemorySource::Undefined && !exportEnabled))
    {
        auto message = fmt::format("The output memory source specified, '{0}',", networkProperties.m_OutputSource);
        if (!exportEnabled)
        {
            message.append(" requires that memory export be enabled. However, "
                           "it was disabled when this network was optimized.");
        }
        else
        {
            message.append(" requires that memory export be disabled. However, "
                           "it was enabled when this network was optimized.");
        }
        throw InvalidArgumentException(message);
    }
} // anonymous

std::unique_ptr<LoadedNetwork> LoadedNetwork::MakeLoadedNetwork(std::unique_ptr<IOptimizedNetwork> net,
                                                                std::string& errorMessage,
                                                                const INetworkProperties& networkProperties,
                                                                arm::pipe::IProfilingService* profilingService)
{
    std::unique_ptr<LoadedNetwork> loadedNetwork;

    auto Fail = [&](const std::exception& error) -> std::unique_ptr<LoadedNetwork>
    {
        errorMessage = ToErrorMessage("An error occurred when preparing the network workloads: ", error);
        ARMNN_LOG(error) << errorMessage;

        return std::unique_ptr<LoadedNetwork>();
    };

    try
    {
        loadedNetwork.reset(new LoadedNetwork(std::move(net), networkProperties, profilingService));
    }
    catch (const armnn::RuntimeException& error)
    {
        return Fail(error);
    }
    catch (const armnn::Exception& error)
    {
        return Fail(error);
    }
    catch (const std::runtime_error& error)
    {
        return Fail(error);
    }

    return loadedNetwork;
}

LoadedNetwork::LoadedNetwork(std::unique_ptr<IOptimizedNetwork> net,
                             const INetworkProperties& networkProperties,
                             arm::pipe::IProfilingService* profilingService) :
                             m_OptimizedNetwork(std::move(net)),
                             m_NetworkProperties(networkProperties),
                             m_TensorHandleFactoryRegistry(),
                             m_ProfilingService(profilingService)
{
    ARMNN_SCOPED_PROFILING_EVENT(Compute::Undefined, "LoadedNetwork");
    // Get the profiler and register it for the current thread.
    const std::shared_ptr<IProfiler>& profiler = m_OptimizedNetwork->GetProfiler();
    ProfilerManager::GetInstance().RegisterProfiler(profiler.get());

    profiler->EnableProfiling(networkProperties.m_ProfilingEnabled);

    profiler->EnableNetworkDetailsToStdOut(networkProperties.m_OutputNetworkDetailsMethod);

    // We need to check that the memory sources match up with the values of import and export specified during the
    // optimize phase. If they don't this will throw an exception.
    ValidateSourcesMatchOptimizedNetwork(m_OptimizedNetwork.get()->pOptimizedNetworkImpl->GetModelOptions(),
                                         m_NetworkProperties);

    //First create tensor handlers, backends and workload factories.
    //Handlers are created before workloads are.
    //Because workload creation can modify some of the handlers,
    //(for example the splitter and concat layers).

    bool useExternalMemoryManager = false;
    bool useInternalMemoryManager = false;
    Graph& order = m_OptimizedNetwork->pOptimizedNetworkImpl->GetGraph();
    // Ensure Topological order
    order.SetLayersOutOfOrder();
    order.TopologicalSort();

    m_IsInputImported = std::vector<bool>(order.GetNumInputs(), false);
    m_IsOutputImported = std::vector<bool>(order.GetNumOutputs(), false);

    for (auto&& layer : order)
    {
        auto const& backendId = layer->GetBackendId();
        if (m_Backends.count(backendId) == 0)
        {
            auto createBackend = BackendRegistryInstance().GetFactory(backendId);
            auto it = m_Backends.emplace(std::make_pair(backendId, createBackend()));

            IBackendInternal* backend = it.first->second.get();

            m_SupportsExternallyManagedMemory[backend->GetId()] = false;
            useInternalMemoryManager = true;

            if (HasMatchingCapability(BackendOptions::BackendOption{"ExternallyManagedMemory", true},
                                      backend->GetCapabilities())
                && (m_NetworkProperties.m_ExternalMemoryManagementEnabled))
            {
                m_SupportsExternallyManagedMemory[backend->GetId()] = true;
                useExternalMemoryManager = true;
                useInternalMemoryManager = false;
            }

            IBackendInternal::IWorkloadFactoryPtr workloadFactory;
            if (backend->SupportsTensorAllocatorAPI())
            {
                workloadFactory = backend->CreateWorkloadFactory(
                    m_TensorHandleFactoryRegistry,
                    m_OptimizedNetwork->pOptimizedNetworkImpl->GetModelOptions(),
                    static_cast<MemorySourceFlags>(m_NetworkProperties.m_InputSource),
                    static_cast<MemorySourceFlags>(m_NetworkProperties.m_OutputSource));
            }
            else
            {
                m_BackendMemoryMangers.emplace_back(backend->CreateMemoryManager());
                workloadFactory = backend->CreateWorkloadFactory(
                        m_BackendMemoryMangers.back(), m_OptimizedNetwork->pOptimizedNetworkImpl->GetModelOptions());
            }
            m_WorkloadFactories[backendId ] = std::move(workloadFactory);
        }
    }

    for (auto&& layer : order)
    {
        auto& workloadFactory = GetWorkloadFactory(*layer);
        bool supportsExternalManager = m_SupportsExternallyManagedMemory[layer->GetBackendId()];

        switch (layer->GetType())
        {
            case LayerType::Input:
            case LayerType::MemImport:
            {
                // If IsImportEnabled is true then we need to set IsMemoryManaged
                // to false when creating TensorHandles
                layer->CreateTensorHandles(m_TensorHandleFactoryRegistry,
                                           workloadFactory,
                                           !supportsExternalManager &&
                                           (m_NetworkProperties.m_InputSource == MemorySource::Undefined));
                break;
            }
            case LayerType::Constant:
            {
                layer->CreateTensorHandles(m_TensorHandleFactoryRegistry, workloadFactory, true);
                break;
            }
            default:
            {
                // Look for a layer with 1 OutputSlot which has 1 connection and that connection is an Output Layer
                // If Export is enabled disable memory management so we can export, otherwise we do a copy
                if ((layer->GetNumOutputSlots() == 1) &&
                   (layer->GetOutputSlots()[0].GetNumConnections() == 1) &&
                   (layer->GetOutputSlots()[0].GetConnection(0)->GetOwningLayer().GetType() == LayerType::Output))
                {
                    layer->CreateTensorHandles(m_TensorHandleFactoryRegistry,
                                               workloadFactory,
                                               !supportsExternalManager &&
                                               (m_NetworkProperties.m_OutputSource == MemorySource::Undefined));
                }
                else
                {
                    layer->CreateTensorHandles(m_TensorHandleFactoryRegistry,
                                               workloadFactory,
                                               !supportsExternalManager);
                }
            }
        }
    }

    ProfilingGuid networkGuid = m_OptimizedNetwork->GetGuid();
    std::unique_ptr<TimelineUtilityMethods> timelineUtils =
        TimelineUtilityMethods::GetTimelineUtils(*m_ProfilingService);
    if (timelineUtils)
    {
        timelineUtils->CreateTypedEntity(networkGuid, LabelsAndEventClasses::NETWORK_GUID);
        // Mark the network with a start of life event
        timelineUtils->RecordEvent(networkGuid, LabelsAndEventClasses::ARMNN_PROFILING_SOL_EVENT_CLASS);
        // and with the process ID
        int processID = arm::pipe::GetCurrentProcessId();
        std::stringstream ss;
        ss << processID;
        timelineUtils->MarkEntityWithLabel(networkGuid, ss.str(), LabelsAndEventClasses::PROCESS_ID_GUID);
    }

    std::vector<IWorkload*> ConstWorkloads;

    //Then create workloads.
    {
        ARMNN_SCOPED_PROFILING_EVENT(Compute::Undefined, "LoadNetwork_CreateWorkloads");
        for (auto&& layer: order)
        {
            if (timelineUtils)
            {
                // Add layer to the post-optimisation network structure
                AddLayerStructure(timelineUtils, *layer, networkGuid);
            }

            const IWorkloadFactory& workloadFactory = GetWorkloadFactory(*layer);

            switch (layer->GetType())
            {
                case LayerType::Input:
                case LayerType::Output:
                {
                    // Inputs and outputs are treated in a special way - see EnqueueInput() and EnqueueOutput().
                    break;
                }
                default:
                {
                    auto workload = layer->CreateWorkload(workloadFactory);

                    if (!workload)
                    {
                        const char* const layerName =
                                layer->GetNameStr().length() != 0 ? layer->GetName() : "<Unnamed>";
                        throw InvalidArgumentException(
                                fmt::format("No workload created for layer (name: '{0}' type: '{1}') (compute '{2}')",
                                            layerName, static_cast<int>(layer->GetType()), layer->GetBackendId().Get()
                                ));
                    }

                    if (timelineUtils)
                    {
                        // Add workload to the post-optimisation network structure
                        AddWorkloadStructure(timelineUtils, workload, *layer);
                    }

                    m_WorkloadQueue.emplace_back(std::move(workload));

                    if (layer->GetType() == LayerType::Constant)
                    {
                        // Place the Constant Workloads into a queue so that they can be executed first
                        ConstWorkloads.emplace_back(m_WorkloadQueue.back().get());
                    }

                    // release the constant data in the layer.
                    layer->ReleaseConstantData();
                    break;
                }
            }
        }
    }

    // Gather information about workloads for inputs & outputs
    if (m_WorkloadQueue.size() != 0)
    {
        const int noOfInputs = armnn::numeric_cast<int>(order.GetNumInputs());

        // Get indices of all workloads connected to each input and
        // check if they support tensor handle replacement
        for (const BindableLayer* layer: order.GetInputLayers())
        {
            const auto bindingId = layer->GetBindingId();

            bool supportsReplacement = true;

            for (const auto inputSlot: layer->GetOutputSlot(0).GetConnections())
            {
                auto workloadIndex = std::distance(order.begin(), order.GetPosInGraph(inputSlot->GetOwningLayer()));
                workloadIndex -= noOfInputs;

                m_InputWorkloadSlotPairs[bindingId].emplace_back(WorkloadIndices{
                        armnn::numeric_cast<unsigned int>(workloadIndex), inputSlot->GetSlotIndex()});

                // Avoid if input is connected directly to an output
                if (inputSlot->GetOwningLayer().GetType() != LayerType::Output)
                {
                    auto workload = m_WorkloadQueue[m_InputWorkloadSlotPairs[bindingId].back().m_WorkloadIndex].get();
                    supportsReplacement &= workload->SupportsTensorHandleReplacement();
                }
            }

            ITensorHandleFactory::FactoryId factoryId = layer->GetOutputSlot(0).GetTensorHandleFactoryId();
            // Get matching import factory Id
            ITensorHandleFactory::FactoryId importFactoryId =
                    m_TensorHandleFactoryRegistry.GetMatchingImportFactoryId(factoryId);

            ITensorHandleFactory *importFactory = m_TensorHandleFactoryRegistry.GetFactory(importFactoryId);

            if (supportsReplacement && importFactory)
            {
                m_PreImportedInputHandles.emplace_back(
                        bindingId, importFactory->CreateTensorHandle(layer->GetOutputSlot(0).GetTensorInfo(), false));
            }
            else
            {
                m_PreImportedInputHandles.emplace_back(bindingId, nullptr);
            }
        }

        // Get indices of all workloads connected to each output and
        // check if they support tensor handle replacement
        for (const BindableLayer* layer: order.GetOutputLayers())
        {
            const auto bindingId = layer->GetBindingId();

            const auto outputSlot = layer->GetInputSlot(0).GetConnectedOutputSlot();
            auto& indices = m_OutputWorkloadSlotPairs[bindingId];

            // Avoid if output is connected directly to an input
            if (outputSlot->GetOwningLayer().GetType() != LayerType::Input)
            {
                auto workloadIndex = std::distance(order.begin(), order.GetPosInGraph(outputSlot->GetOwningLayer()));
                workloadIndex -= noOfInputs;

                indices.m_OutputSlotIndices = WorkloadIndices{numeric_cast<unsigned int>(workloadIndex),
                                                              outputSlot->CalculateIndexOnOwner()};

                bool supportsReplacement = true;
                auto outputWorkload = m_WorkloadQueue[indices.m_OutputSlotIndices.m_WorkloadIndex].get();
                supportsReplacement &= outputWorkload->SupportsTensorHandleReplacement();

                for (auto &inputSlot: outputSlot->GetConnections())
                {
                    if (inputSlot->GetOwningLayer().GetType() != LayerType::Output)
                    {
                        auto inWorkloadIndex = std::distance(order.begin(),
                                                             order.GetPosInGraph(inputSlot->GetOwningLayer()));
                        inWorkloadIndex -= noOfInputs;
                        indices.m_InputSlotIndices.emplace_back(
                                WorkloadIndices{numeric_cast<unsigned int>(inWorkloadIndex),
                                                inputSlot->GetSlotIndex()});
                        auto inputWorkload = m_WorkloadQueue[indices.m_InputSlotIndices.back().m_WorkloadIndex].get();
                        supportsReplacement &= inputWorkload->SupportsTensorHandleReplacement();
                    }
                }

                ITensorHandleFactory::FactoryId factoryId = outputSlot->GetTensorHandleFactoryId();
                // Get matching import factory Id
                ITensorHandleFactory::FactoryId importFactoryId =
                        m_TensorHandleFactoryRegistry.GetMatchingImportFactoryId(factoryId);
                ITensorHandleFactory *importFactory = m_TensorHandleFactoryRegistry.GetFactory(importFactoryId);

                if (supportsReplacement && importFactory)
                {
                    m_PreImportedOutputHandles.emplace_back(
                            bindingId, importFactory->CreateTensorHandle(outputSlot->GetTensorInfo(), false));
                }
                else
                {
                    m_PreImportedOutputHandles.emplace_back(bindingId, nullptr);
                }
            }
        }
    }

    for (auto&& workloadFactory : m_WorkloadFactories)
    {
        workloadFactory.second->AfterWorkloadsCreated();
    }

    if (timelineUtils)
    {
        // Commit to send the post-optimisation network structure
        timelineUtils->Commit();
    }

    if (useExternalMemoryManager)
    {
        CreateMemoryProfile();
        auto backendStrategyMap = BackendRegistryInstance().GetMemoryOptimizerStrategies();
        for (auto& backendMemoryProfile : m_MemBlockMap)
        {
            const BackendId& backendId = backendMemoryProfile.first;
            if (backendStrategyMap.find(backendId) != backendStrategyMap.end())
            {
                m_MemBinMap[backendId] = backendStrategyMap[backendId]->Optimize(backendMemoryProfile.second);
            }
            else
            {
                m_MemBinMap[backendId] = m_ConstantStrategy->Optimize(backendMemoryProfile.second);
            }
        }
        m_ExternalMemoryManager = CreateExternalMemoryManger(m_TensorMemory);

        // Sort m_TensorMemory, so it's order matches m_Tensorhandles
        std::sort(m_TensorMemory.begin(), m_TensorMemory.end(),
                  [](const std::pair<std::shared_ptr<TensorMemory>, MemorySource>& lhs,
                     const std::pair<std::shared_ptr<TensorMemory>, MemorySource>& rhs)
                  {
                      return lhs.first->m_OutputSlotId < rhs.first->m_OutputSlotId;
                  });
    }


    // Now that the intermediate tensor memory has been set-up,
    // do any post allocation configuration for each workload.

    if (useInternalMemoryManager)
    {
        // Set up memory.
        m_OptimizedNetwork->pOptimizedNetworkImpl->GetGraph().AllocateDynamicBuffers();
    }

    if (useExternalMemoryManager)
    {
        AllocateAndExecuteConstantWorkloads();
    }

    for (const auto& workload : m_WorkloadQueue)
    {
        workload->PostAllocationConfigure();
    }

    // Execute all constant layer workloads
    for (auto workload: ConstWorkloads)
    {
        workload->Execute();
    }
    MARK_OPTIMIZED_NETWORK_LOADED()
}

void LoadedNetwork::AllocateAndExecuteConstantWorkloads()
{
    ARMNN_SCOPED_PROFILING_EVENT(Compute::Undefined, "LoadNetwork_AllocateAndExecuteConstants");
    for (auto& pair : m_ConstantWorkloads)
    {
        auto tensorHandle = m_ConstantTensorHandles[pair.first];
        tensorHandle->Allocate();
        pair.second->Execute();
    }
}

void LoadedNetwork::SendNetworkStructure(arm::pipe::IProfilingService& profilingService)
{
    ARMNN_SCOPED_PROFILING_EVENT(Compute::Undefined, "LoadNetwork_SendNetworkStructure");
    Graph& order = m_OptimizedNetwork->pOptimizedNetworkImpl->GetGraph().TopologicalSort();
    ProfilingGuid networkGuid = m_OptimizedNetwork->GetGuid();

    std::unique_ptr<TimelineUtilityMethods> timelineUtils =
        TimelineUtilityMethods::GetTimelineUtils(profilingService);

    timelineUtils->CreateTypedEntity(networkGuid, LabelsAndEventClasses::NETWORK_GUID);

    for (auto&& layer : order)
    {
        // Add layer to the post-optimisation network structure
        AddLayerStructure(timelineUtils, *layer, networkGuid);
        switch (layer->GetType())
        {
            case LayerType::Input:
            case LayerType::Output:
            {
                // Inputs and outputs are treated in a special way - see EnqueueInput() and EnqueueOutput().
                break;
            }
            default:
            {
                for (auto& workload : m_WorkloadQueue)
                {
                    // Add workload to the post-optimisation network structure
                    AddWorkloadStructure(timelineUtils, workload, *layer);
                }
            break;
            }
        }
    }
    // Commit to send the post-optimisation network structure
    timelineUtils->Commit();
}

ProfilingGuid LoadedNetwork::GetNetworkGuid()
{
    return m_OptimizedNetwork->GetGuid();
}

TensorInfo LoadedNetwork::GetInputTensorInfo(LayerBindingId layerId) const
{
    for (auto&& inputLayer : m_OptimizedNetwork->pOptimizedNetworkImpl->GetGraph().GetInputLayers())
    {
        if (inputLayer->GetNumOutputSlots() != 1)
        {
            throw armnn::GraphValidationException("Input layer should have exactly 1 output slot");
        }

        if (inputLayer->GetBindingId() == layerId)
        {
            return inputLayer->GetOutputSlot(0).GetTensorInfo();
        }
    }

    throw InvalidArgumentException(fmt::format("No input layer is associated with id {}", layerId));
}

TensorInfo LoadedNetwork::GetOutputTensorInfo(LayerBindingId layerId) const
{
    for (auto&& outputLayer : m_OptimizedNetwork->pOptimizedNetworkImpl->GetGraph().GetOutputLayers())
    {
        if (outputLayer->GetNumInputSlots() != 1)
        {
            throw armnn::GraphValidationException("Output layer should have exactly 1 input slot");
        }

        if (!outputLayer->GetInputSlot(0).GetConnection())
        {
            throw armnn::GraphValidationException("Input slot on Output layer must be connected");
        }

        if (outputLayer->GetBindingId() == layerId)
        {
            return outputLayer->GetInputSlot(0).GetTensorInfo();
        }
    }

    throw InvalidArgumentException(fmt::format("No output layer is associated with id {}", layerId));
}

const IWorkloadFactory& LoadedNetwork::GetWorkloadFactory(const Layer& layer) const
{
    const IWorkloadFactory* workloadFactory = nullptr;

    auto it = m_WorkloadFactories.find(layer.GetBackendId());
    if (it ==  m_WorkloadFactories.end())
    {
        throw RuntimeException(fmt::format("No workload factory for {0} to be used for layer: {1}",
                                           layer.GetBackendId().Get(),
                                           layer.GetNameStr()),
                                           CHECK_LOCATION());
    }

    workloadFactory = it->second.get();

    if (!workloadFactory)
    {
        throw armnn::NullPointerException("No workload factory");
    }

    return *workloadFactory;
}

namespace {

// Non-copyable class owning accelerator-specific tensor data.
class TensorPin
{
public:
    TensorPin(std::unique_ptr<ITensorHandle> handle, const TensorInfo& info, LayerBindingId id)
        : m_TensorHandle(std::move(handle))
        , m_TensorInfo(info)
        , m_Id(id)
    {
    }

    ITensorHandle* GetTensorHandle() const { return m_TensorHandle.get(); }
    const TensorInfo& GetTensorInfo() const { return m_TensorInfo; }
    LayerBindingId GetBindingId() const { return m_Id; }

private:
    std::unique_ptr<ITensorHandle> m_TensorHandle;
    TensorInfo m_TensorInfo;
    LayerBindingId m_Id;
};

static const TensorPin& GetTensorPin(LayerBindingId id,
    const std::vector<TensorPin>& pins,
    char const* bindingPointDesc)
{
    auto it = std::find_if(pins.begin(), pins.end(),
        [id](const TensorPin& pin)
    {
        return pin.GetBindingId() == id;
    });

    if (it != pins.end())
    {
        return *it;
    }
    else
    {
        throw InvalidArgumentException(fmt::format("No tensor supplied for {0} {1}", bindingPointDesc, id));
    }
}

// Stores data that needs to be kept accessible for the entire execution of a workload.
class WorkloadData
{
public:
    WorkloadData(const InputTensors& inputTensors, const OutputTensors& outputTensors)
    {
        m_InputTensorPins.reserve(inputTensors.size());
        m_OutputTensorPins.reserve(outputTensors.size());

        for (auto inputTensorPair : inputTensors)
        {
            auto inputTensor = inputTensorPair.second;

            std::unique_ptr<ITensorHandle> tensorHandle =
                std::make_unique<ConstPassthroughTensorHandle>(inputTensor.GetInfo(),inputTensor.GetMemoryArea());
            LayerBindingId layerId = inputTensorPair.first;

            m_InputTensorPins.emplace_back(std::move(tensorHandle), inputTensor.GetInfo(), layerId);
        }

        for (auto outputTensorPair : outputTensors)
        {
            auto outputTensor = outputTensorPair.second;

            std::unique_ptr<ITensorHandle> tensorHandle =
                std::make_unique<PassthroughTensorHandle>(outputTensor.GetInfo(), outputTensor.GetMemoryArea());
            LayerBindingId layerId = outputTensorPair.first;

            m_OutputTensorPins.emplace_back(std::move(tensorHandle), outputTensor.GetInfo(), layerId);
        }
    }

    const TensorPin& GetInputTensorPin(LayerBindingId id) const
    {
        return GetTensorPin(id, m_InputTensorPins, "input");
    }

    const TensorPin& GetOutputTensorPin(LayerBindingId id) const
    {
        return GetTensorPin(id, m_OutputTensorPins, "output");
    }

private:

    std::vector<TensorPin> m_InputTensorPins;
    std::vector<TensorPin> m_OutputTensorPins;
};

}

Status LoadedNetwork::EnqueueWorkload(const InputTensors& inputTensors,
                                      const OutputTensors& outputTensors,
                                      std::vector<ImportedInputId> preImportedInputIds,
                                      std::vector<ImportedOutputId> preImportedOutputIds)
{
    const Graph& graph = m_OptimizedNetwork->pOptimizedNetworkImpl->GetGraph();

    // Walk graph to determine the order of execution.
    if (graph.GetNumLayers() < 2)
    {
        ARMNN_LOG(warning) << "IRuntime::EnqueueWorkload()::Less than two nodes in graph";
        return Status::Failure;
    }

    // Data that must be kept alive for the entire execution of the workload.
    WorkloadData workloadData(inputTensors, outputTensors);

    // Input tensors can be provided as parameters or pre imported. Either way the number of
    // tensors should match the number of inputs.
    if (graph.GetNumInputs() != (inputTensors.size() + preImportedInputIds.size()))
    {
        throw InvalidArgumentException("Number of inputs provided does not match network.");
    }

    // For each input to the network, call EnqueueInput with the data passed by the user.
    {
        ARMNN_SCOPED_PROFILING_EVENT(Compute::Undefined, "PrepareInputs");
        m_InputQueue.clear();
        m_InputQueue.reserve(graph.GetNumInputs());

        unsigned int inputIndex = 0;
        unsigned int importedInputIdIndex = 0;
        std::sort(preImportedInputIds.begin(), preImportedInputIds.end());
        for (const BindableLayer* inputLayer : graph.GetInputLayers())
        {
            if (importedInputIdIndex < preImportedInputIds.size() &&
                inputIndex == preImportedInputIds[importedInputIdIndex])
            {
                // Only replace tensorhandles if they have not already been replaced
                if (!m_IsInputImported[inputIndex])
                {
                    auto outputTensorHandle = m_PreImportedInputHandles[inputIndex].m_TensorHandle.get();

                    for (const auto& workloadInfo: m_InputWorkloadSlotPairs[inputLayer->GetBindingId()])
                    {
                        auto workload = m_WorkloadQueue[workloadInfo.m_WorkloadIndex].get();
                        workload->ReplaceInputTensorHandle(outputTensorHandle, workloadInfo.m_SlotIndex);
                    }
                    m_IsInputImported[inputIndex] = true;
                }
                importedInputIdIndex++;
            }
            else
            {
                if (m_IsInputImported[inputIndex])
                {
                    OutputHandler& handler = const_cast<OutputHandler&>(inputLayer->GetOutputHandler(0));

                    for (const auto& workloadInfo: m_InputWorkloadSlotPairs[inputLayer->GetBindingId()])
                    {
                        auto workload = m_WorkloadQueue[workloadInfo.m_WorkloadIndex].get();
                        workload->ReplaceInputTensorHandle(handler.GetData(), workloadInfo.m_SlotIndex);
                    }

                    m_IsInputImported[inputIndex] = false;
                }

                // InputTensorHandle is not imported yet, process to enqueue input
                const TensorPin& pin = workloadData.GetInputTensorPin(inputLayer->GetBindingId());
                EnqueueInput(*inputLayer, pin.GetTensorHandle(), pin.GetTensorInfo());
            }
            inputIndex++;
        }
    }
    // For each output to the network, call EnqueueOutput with the data passed by the user.
    {
        ARMNN_SCOPED_PROFILING_EVENT(Compute::Undefined, "PrepareOutputs");
        m_OutputQueue.clear();
        m_OutputQueue.reserve(graph.GetNumOutputs());

        if (preImportedOutputIds.size() > graph.GetNumOutputs())
        {
            throw InvalidArgumentException("Invalid number of preImportedOutputIds");
        }

        unsigned int outputIndex = 0;
        unsigned int importedOutputIdIndex = 0;
        std::sort(preImportedOutputIds.begin(), preImportedOutputIds.end());
        for (const BindableLayer* outputLayer : graph.GetOutputLayers())
        {
            if (importedOutputIdIndex < preImportedOutputIds.size() &&
                outputIndex == preImportedOutputIds[importedOutputIdIndex])
            {
                // Only replace tensorhandles if they have not already been replaced
                ITensorHandle* inputTensorHandle = m_PreImportedOutputHandles[outputIndex].m_TensorHandle.get();

                if (!m_IsOutputImported[outputIndex])
                {
                    const auto bindingId = outputLayer->GetBindingId();
                    const auto& indices = m_OutputWorkloadSlotPairs[bindingId];

                    auto outputWorkload = m_WorkloadQueue[indices.m_OutputSlotIndices.m_WorkloadIndex].get();

                    outputWorkload->ReplaceOutputTensorHandle(inputTensorHandle,
                                                              indices.m_OutputSlotIndices.m_SlotIndex);

                    for (const auto& workloadInfo: indices.m_InputSlotIndices)
                    {
                        auto inputWorkload = m_WorkloadQueue[workloadInfo.m_WorkloadIndex].get();
                        inputWorkload->ReplaceInputTensorHandle(inputTensorHandle, workloadInfo.m_SlotIndex);
                    }
                    m_IsOutputImported[outputIndex] = true;
                }

                if (!inputTensorHandle)
                {
                    throw armnn::NullPointerException("Data should have been allocated.");
                }

                MemSyncQueueDescriptor syncDesc;
                syncDesc.m_Inputs.push_back(inputTensorHandle);
                WorkloadInfo info;
                info.m_InputTensorInfos.push_back(outputLayer->GetInputSlot(0).GetTensorInfo());

                auto syncWorkload = std::make_unique<SyncMemGenericWorkload>(syncDesc, info);
                if (!syncWorkload)
                {
                    throw armnn::NullPointerException("No sync workload created");
                }

                m_OutputQueue.push_back(std::move(syncWorkload));
                importedOutputIdIndex++;
            }
            else
            {
                if (m_IsOutputImported[outputIndex])
                {
                    const auto bindingId = outputLayer->GetBindingId();
                    const auto& indices = m_OutputWorkloadSlotPairs[bindingId];

                    auto outputWorkload = m_WorkloadQueue[indices.m_OutputSlotIndices.m_WorkloadIndex].get();
                    const OutputHandler& outputHandler =
                            outputLayer->GetInputSlot(0).GetConnectedOutputSlot()->GetOutputHandler();

                    outputWorkload->ReplaceOutputTensorHandle(
                            outputHandler.GetData(), indices.m_OutputSlotIndices.m_SlotIndex);

                    for (const auto& workloadInfo: indices.m_InputSlotIndices)
                    {
                        auto inputWorkload = m_WorkloadQueue[workloadInfo.m_WorkloadIndex].get();
                        inputWorkload->ReplaceInputTensorHandle(outputHandler.GetData(), workloadInfo.m_SlotIndex);
                    }
                    m_IsOutputImported[outputIndex] = false;
                }

                const TensorPin& pin = workloadData.GetOutputTensorPin(outputLayer->GetBindingId());
                // OutputTensorHandle is not imported yet, process to enqueue Output
                EnqueueOutput(*outputLayer, pin.GetTensorHandle(), pin.GetTensorInfo());
            }
            outputIndex++;
        }
    }

    std::unique_ptr<TimelineUtilityMethods> timelineUtils =
                        TimelineUtilityMethods::GetTimelineUtils(*m_ProfilingService);
    ProfilingGuid inferenceGuid = m_ProfilingService->GetNextGuid();
    if (timelineUtils)
    {
        // Add inference timeline trace if profiling is enabled.
        ProfilingGuid networkGuid = m_OptimizedNetwork->GetGuid();
        timelineUtils->CreateTypedEntity(inferenceGuid, LabelsAndEventClasses::INFERENCE_GUID);
        timelineUtils->CreateRelationship(ProfilingRelationshipType::RetentionLink,
                                          networkGuid,
                                          inferenceGuid,
                                          LabelsAndEventClasses::EXECUTION_OF_GUID);
        timelineUtils->RecordEvent(inferenceGuid, LabelsAndEventClasses::ARMNN_PROFILING_SOL_EVENT_CLASS);
    }

    bool executionSucceeded = true;

    {
        if (m_ProfilingService->IsProfilingEnabled())
        {
            m_ProfilingService->IncrementCounterValue(INFERENCES_RUN);
        }
        ARMNN_SCOPED_PROFILING_EVENT(Compute::Undefined, "Execute");
        ARMNN_SCOPED_HEAP_PROFILING("Executing");
        executionSucceeded = Execute(timelineUtils, inferenceGuid);
    }

    if (timelineUtils)
    {
        // Add end of life of the inference timeline if profiling is enabled.
        timelineUtils->RecordEvent(inferenceGuid, LabelsAndEventClasses::ARMNN_PROFILING_EOL_EVENT_CLASS);
        timelineUtils->Commit();
    }

    return executionSucceeded ? Status::Success : Status::Failure;
}

void LoadedNetwork::EnqueueInput(const BindableLayer& layer, ITensorHandle* tensorHandle, const TensorInfo& tensorInfo)
{
    if (layer.GetType() != LayerType::Input)
    {
        throw InvalidArgumentException("EnqueueInput: given layer not an InputLayer");
    }

    if (tensorHandle == nullptr)
    {
        throw InvalidArgumentException("EnqueueInput: tensorHandle must not be NULL");
    }

    InputQueueDescriptor inputQueueDescriptor;
    WorkloadInfo info;

    inputQueueDescriptor.m_Inputs.push_back(tensorHandle);
    info.m_InputTensorInfos.push_back(tensorInfo);

    if (layer.GetNumOutputSlots() != 1)
    {
        throw armnn::GraphValidationException("Can only handle Input Layer with one output");
    }

    const OutputHandler& handler = layer.GetOutputHandler();
    const TensorInfo& outputTensorInfo = handler.GetTensorInfo();
    ITensorHandle* outputTensorHandle = handler.GetData();

    if (!outputTensorHandle)
    {
        throw armnn::NullPointerException("Data should have been allocated.");
    }

    inputQueueDescriptor.m_Outputs.push_back(outputTensorHandle);
    info.m_OutputTensorInfos.push_back(outputTensorInfo);

    MemorySourceFlags importFlags = outputTensorHandle->GetImportFlags();
    bool needMemCopy = true;
    if ((m_NetworkProperties.m_InputSource != MemorySource::Undefined))  // Try import the input tensor
    {
        if(CheckFlag(importFlags, m_NetworkProperties.m_InputSource))
        {
            needMemCopy = false;
            // This assumes a CPU Tensor handle
            void* mem = tensorHandle->Map(false);
            if (outputTensorHandle->Import(mem, m_NetworkProperties.m_InputSource))
            {
                tensorHandle->Unmap();
                return; // No need for a workload since the import has been done.
            }
            tensorHandle->Unmap();
            throw MemoryImportException("EnqueueInput: Memory Import failed");
        }
    }
    if (needMemCopy)
    {
        // Create a mem copy workload for input since we did not import
        std::unique_ptr<IWorkload> inputWorkload = std::make_unique<CopyMemGenericWorkload>(inputQueueDescriptor, info);

        if (!inputWorkload)
        {
            throw armnn::NullPointerException("No input workload created");
        }

        std::unique_ptr<TimelineUtilityMethods> timelineUtils =
                            TimelineUtilityMethods::GetTimelineUtils(*m_ProfilingService);
        if (timelineUtils)
        {
            // Add Input Workload to the post-optimisation network structure
            AddWorkloadStructure(timelineUtils, inputWorkload, layer);
            timelineUtils->Commit();
        }

        m_InputQueue.push_back(std::move(inputWorkload));
    }
}

void LoadedNetwork::EnqueueOutput(const BindableLayer& layer, ITensorHandle* tensorHandle, const TensorInfo& tensorInfo)
{
    if (layer.GetType() != LayerType::Output)
    {
        throw InvalidArgumentException("EnqueueOutput: given layer not an OutputLayer");
    }

    if (tensorHandle == nullptr)
    {
        throw InvalidArgumentException("EnqueueOutput: tensorHandle must not be NULL");
    }

    OutputQueueDescriptor outputQueueDescriptor;
    WorkloadInfo info;

    outputQueueDescriptor.m_Outputs.push_back(tensorHandle);
    info.m_OutputTensorInfos.push_back(tensorInfo);

    if (layer.GetNumInputSlots() != 1)
    {
        throw armnn::GraphValidationException("Output Layer should have exactly one input.");
    }

    // Gets the output handler from the previous node.
    const OutputHandler& outputHandler = layer.GetInputSlots()[0].GetConnectedOutputSlot()->GetOutputHandler();

    const TensorInfo& inputTensorInfo = outputHandler.GetTensorInfo();
    ITensorHandle* inputTensorHandle = outputHandler.GetData();
    if (!inputTensorHandle)
    {
        throw armnn::NullPointerException("Data should have been allocated.");
    }

    // Try import the output tensor.
    // Note: We can only import the output pointer if all of the following  hold true:
    // a) The imported pointer is aligned sufficiently
    // b) The tensor has zero padding
    // c) There is only one connection to the OutputSlot and it is to an OutputLayer.
    // d) The output pointer is allocated via malloc. (Other types will be supported in a later release)
    // e) m_NetworkProperties.m_OutputSource != MemorySource::Undefined
    bool needMemCopy = true;
    if (m_NetworkProperties.m_OutputSource != MemorySource::Undefined &&
        (layer.GetInputSlots()[0].GetConnectedOutputSlot()->GetNumConnections() == 1))
    {
        if(layer.GetInputSlots()[0].GetConnectedOutputSlot()->GetOwningLayer().GetType() != LayerType::Input)
        {
            MemorySourceFlags importFlags = inputTensorHandle->GetImportFlags();
            if (CheckFlag(importFlags, m_NetworkProperties.m_OutputSource))
            {
                needMemCopy = false;
                void *mem = tensorHandle->Map(false);
                bool importOk = inputTensorHandle->Import(mem, m_NetworkProperties.m_OutputSource);
                tensorHandle->Unmap();

                if (importOk)
                {
                    // Insert synchronization workload
                    MemSyncQueueDescriptor syncDesc;
                    syncDesc.m_Inputs.push_back(inputTensorHandle);
                    info.m_InputTensorInfos.push_back(inputTensorInfo);
                    auto syncWorkload = std::make_unique<SyncMemGenericWorkload>(syncDesc, info);
                    if (!syncWorkload)
                    {
                        throw armnn::NullPointerException("No sync workload created");
                    }
                    m_OutputQueue.push_back(std::move(syncWorkload));
                }
                else
                {
                    throw MemoryExportException("EnqueueOutput: Memory Export failed");
                }
            }
        }
    }
    if (needMemCopy)
    {
        // If we got here then we didn't export the memory, so add an output workload which performs a memcopy.
        outputQueueDescriptor.m_Inputs.push_back(inputTensorHandle);
        info.m_InputTensorInfos.push_back(inputTensorInfo);

        std::unique_ptr<IWorkload> outputWorkload =
            std::make_unique<CopyMemGenericWorkload>(outputQueueDescriptor, info);
        if (!outputWorkload)
        {
            throw armnn::NullPointerException("No output workload created");
        }

        std::unique_ptr<TimelineUtilityMethods> timelineUtils =
            TimelineUtilityMethods::GetTimelineUtils(*m_ProfilingService);
        if (timelineUtils)
        {
            // Add Output Workload to the post-optimisation network structure
            AddWorkloadStructure(timelineUtils, outputWorkload, layer);
            timelineUtils->Commit();
        }

        m_OutputQueue.push_back(std::move(outputWorkload));
    }
}

void LoadedNetwork::AllocateWorkingMemory(
#if !defined(ARMNN_DISABLE_THREADS)
     std::lock_guard<std::mutex>& lock
#endif
    )
{
    ARMNN_SCOPED_PROFILING_EVENT(Compute::Undefined, "Working Memory Allocation");

#if !defined(ARMNN_DISABLE_THREADS)
    // this unused parameter makes sure we can only call this function with a valid lock
    IgnoreUnused(lock);
#endif
    if (m_IsWorkingMemAllocated)
    {
        return;
    }

    if (m_ExternalMemoryManager)
    {
        m_ExternalMemoryManager->Allocate();

        for (unsigned int i = 0; i < m_TensorMemory.size(); ++i)
        {
            m_Tensorhandles[i]->Import(m_TensorMemory[i].first->m_Data, m_TensorMemory[i].second);
        }
    }

    for (auto&& memoryManager : m_BackendMemoryMangers)
    {
        if (memoryManager)
        {
            memoryManager->Acquire();
        }
    }
    m_TensorHandleFactoryRegistry.AquireMemory();
    m_IsWorkingMemAllocated = true;
}

void LoadedNetwork::FreeWorkingMemory()
{
#if !defined(ARMNN_DISABLE_THREADS)
    std::lock_guard<std::mutex> lockGuard(m_WorkingMemMutex);
#endif

    if (!m_IsWorkingMemAllocated)
    {
        return;
    }

    if (m_ExternalMemoryManager)
    {
        m_ExternalMemoryManager->Deallocate();
    }

    // Informs the memory managers to release memory in its respective memory group
    for (auto&& memoryManager : m_BackendMemoryMangers)
    {
        if (memoryManager)
        {
            memoryManager->Release();
        }
    }
    m_TensorHandleFactoryRegistry.ReleaseMemory();
    m_IsWorkingMemAllocated = false;
}

bool LoadedNetwork::Execute(std::unique_ptr<TimelineUtilityMethods>& timelineUtils,
                           ProfilingGuid inferenceGuid)
{
    bool success = true;

    auto Fail = [&](const std::exception& error)
    {
        ARMNN_LOG(error) << "An error occurred attempting to execute a workload: " << error.what();
        success = false;
    };

    try
    {
#if !defined(ARMNN_DISABLE_THREADS)
        std::lock_guard<std::mutex> lockGuard(m_WorkingMemMutex);
        AllocateWorkingMemory(lockGuard);
#else
        AllocateWorkingMemory();
#endif

        ProfilingDynamicGuid workloadInferenceID(0);
        auto ExecuteQueue = [&timelineUtils, &workloadInferenceID, &inferenceGuid](WorkloadQueue& queue)
        {
            for (auto& workload : queue)
            {
                if(timelineUtils)
                {
                    workloadInferenceID = timelineUtils->RecordWorkloadInferenceAndStartOfLifeEvent(workload->GetGuid(),
                                                                                                    inferenceGuid);
                }

                MARK_WORKLOAD_EXECUTION_BEGIN();
                workload->Execute();
                MARK_WORKLOAD_EXECUTION_END();
                if(timelineUtils)
                {
                    timelineUtils->RecordEndOfLifeEvent(workloadInferenceID);
                }
            }
        };

        MARK_INFERENCE_EXECUTION_BEGIN();
        ExecuteQueue(m_InputQueue);
        ExecuteQueue(m_WorkloadQueue);
        ExecuteQueue(m_OutputQueue);
        MARK_INFERENCE_EXECUTION_END();
    }
    catch (const RuntimeException& error)
    {
        Fail(error);
    }
    catch (const std::runtime_error& error)
    {
        Fail(error);
    }

    return success;
}

void LoadedNetwork::EnqueueInput(const ConstTensor& inputTensor, ITensorHandle* inputTensorHandle)
{
    if (m_NetworkProperties.m_InputSource != MemorySource::Undefined)  // Try import the input tensor
    {
        MemorySourceFlags importFlags = inputTensorHandle->GetImportFlags();
        if (CheckFlag(importFlags, m_NetworkProperties.m_InputSource) )
        {
            std::unique_ptr<ITensorHandle> tensorHandle =
                    std::make_unique<ConstPassthroughTensorHandle>(inputTensor.GetInfo(),
                                                                   inputTensor.GetMemoryArea());
            void* mem = tensorHandle->Map(false);

            if (inputTensorHandle->Import(mem, m_NetworkProperties.m_InputSource))
            {
                tensorHandle->Unmap();
                return;
            }
            tensorHandle->Unmap();
            throw MemoryImportException("EnqueueInput: Memory Import failed");
        }
        else
        {
            throw MemoryImportException("EnqueueInput: Memory Import failed, backend does not support Import");
        }
    }
    else
    {
        ARMNN_SCOPED_PROFILING_EVENT(Compute::Undefined, "CopyInput");
        std::unique_ptr<ITensorHandle> tensorHandle =
                std::make_unique<ConstPassthroughTensorHandle>(inputTensor.GetInfo(), inputTensor.GetMemoryArea());

        auto copyFunc = [](void* dst, const void* src, size_t size)
        {
            memcpy(dst, src, size);
        };

        CopyTensorContentsGeneric(tensorHandle.get(), inputTensorHandle, copyFunc);
    }
}

// Note: We can only import the output pointer if all of the following  hold true:
// a) The imported pointer is aligned sufficiently
// b) The tensor has zero padding
// c) There is only one connection to the OutputSlot and it is to an OutputLayer.
// d) The output pointer is allocated via malloc. (Other types will be supported in a later release)
// e) m_IsExportEnabled must be set to true
void LoadedNetwork::ImportOutputTensor(const Tensor& outputTensor, ITensorHandle* outputTensorHandle)
{
    if (!outputTensorHandle)
    {
        throw armnn::NullPointerException("Data should have been allocated.");
    }

    MemorySourceFlags importFlags = outputTensorHandle->GetImportFlags();
    if (CheckFlag(importFlags, m_NetworkProperties.m_OutputSource))
    {
        std::unique_ptr<ITensorHandle> tensorHandle =
                std::make_unique<PassthroughTensorHandle>(outputTensor.GetInfo(),
                                                          outputTensor.GetMemoryArea());

        void* mem = tensorHandle->Map(false);
        bool importOk = outputTensorHandle->Import(mem, m_NetworkProperties.m_OutputSource);
        tensorHandle->Unmap();

        if (!importOk)
        {
            throw MemoryExportException("ImportOutputTensor: Memory Export failed");
        }
    }
    else
    {
        throw MemoryExportException("ImportOutputTensor: Memory Export failed, attempting to export Input Layer");
    }

}

void CopyToOutputTensor(const Tensor& outputTensor, ITensorHandle* outputTensorHandle)
{
    ARMNN_SCOPED_PROFILING_EVENT(Compute::Undefined, "CopyOutput");
    auto copyFunc = [](void* dst, const void* src, size_t size)
    {
        memcpy(dst, src, size);
    };

    std::unique_ptr<ITensorHandle> tensorHandle =
            std::make_unique<PassthroughTensorHandle>(outputTensor.GetInfo(),
                                                      outputTensor.GetMemoryArea());

    CopyTensorContentsGeneric(outputTensorHandle, tensorHandle.get(), copyFunc);
}


const armnn::ConstTensor GetInputTensor(const LayerBindingId layerId, const InputTensors& inputTensors)
{
    for (auto inputTensorPair : inputTensors)
    {
        LayerBindingId id = inputTensorPair.first;
        if (id == layerId)
        {
            return inputTensorPair.second;
        }
    }
    throw InvalidArgumentException("Input does not exist.");
}

const armnn::Tensor GetOutputTensor(const LayerBindingId layerId, const OutputTensors& outputTensors)
{
    for (auto outputTensorPair : outputTensors)
    {
        LayerBindingId id = outputTensorPair.first;
        if (id == layerId)
        {
            return outputTensorPair.second;
        }
    }
    throw InvalidArgumentException("Output does not exist.");
}

std::vector<ImportedInputId> LoadedNetwork::ImportInputs(const InputTensors& inputTensors,
                                                         MemorySource forceImportMemorySource)
{
    // Cannot import if import is not enabled and forceImportMemorySource is undefined
    if (forceImportMemorySource == MemorySource::Undefined)
    {
        throw MemoryImportException("ImportInputs: Memory Import failed, NetworkProperties.m_ImportEnabled");
    }
    // The number of pre imported tensors should not exceed the number of inputs.
    if (inputTensors.size() > m_OptimizedNetwork->pOptimizedNetworkImpl->GetGraph().GetNumInputs())
    {
        throw MemoryImportException("ImportInputs: The number of tensors provided exceeds the number of inputs.");
    }

    std::vector<ImportedInputId> importedInputs;
    Graph& graph = m_OptimizedNetwork->pOptimizedNetworkImpl->GetGraph().TopologicalSort();
    unsigned int inputIndex = 0;
    for (const BindableLayer* inputLayer : graph.GetInputLayers())
    {
        auto outputTensorHandle = m_PreImportedInputHandles[inputIndex].m_TensorHandle.get();

        if (!outputTensorHandle)
        {
            inputIndex++;
            continue;
        }

        auto layerBindingId = inputLayer->GetBindingId();
        auto it = std::find_if(inputTensors.begin(), inputTensors.end(), [=](const auto& inputTensor)
        {
            return inputTensor.first == layerBindingId;
        });

        if (it == inputTensors.end())
        {
            inputIndex++;
            continue;
        }

        const auto& inputTensor = *it;
        std::unique_ptr<ITensorHandle> passThroughTensorHandle =
                std::make_unique<ConstPassthroughTensorHandle>(inputTensor.second.GetInfo(),
                                                               inputTensor.second.GetMemoryArea());

        try
        {
            if (outputTensorHandle->CanBeImported(passThroughTensorHandle->Map(), forceImportMemorySource)
                && (outputTensorHandle->Import(passThroughTensorHandle->Map(), forceImportMemorySource)))
            {
                importedInputs.push_back(inputIndex);
            }
            passThroughTensorHandle->Unmap();
        }
        catch(const MemoryImportException& exception)
        {
            ARMNN_LOG(error) << "An error occurred attempting to import input_"
                                       << inputIndex << " : " << exception.what();
            passThroughTensorHandle->Unmap();
        }
        inputIndex++;
    }

    return importedInputs;
}

std::vector<ImportedOutputId> LoadedNetwork::ImportOutputs(const OutputTensors& outputTensors,
                                                           MemorySource forceImportMemorySource)
{
    // Cannot import if import is not enabled and forceImportMemorySource is undefined
    if (forceImportMemorySource == MemorySource::Undefined)
    {
        throw MemoryImportException("ImportOutputs: Memory Import failed, NetworkProperties.m_ImportEnabled");
    }
    // If forceImportMemorySource is defined, try import if memory is aligned
    if (outputTensors.size() != m_OptimizedNetwork->pOptimizedNetworkImpl->GetGraph().GetNumOutputs())
    {
        throw MemoryImportException("ImportOutputs: Force Import failed, incorrect number of tensors");
    }
    std::vector<ImportedOutputId> importedOutputs;
    Graph& graph = m_OptimizedNetwork->pOptimizedNetworkImpl->GetGraph().TopologicalSort();

    unsigned int outputIndex = 0;
    for (const BindableLayer* const outputLayer : graph.GetOutputLayers())
    {
        auto inputTensorHandle = m_PreImportedOutputHandles[outputIndex].m_TensorHandle.get();
        if (!inputTensorHandle)
        {
            outputIndex++;
            continue;
        }

        auto layerBindingId = outputLayer->GetBindingId();
        auto it = std::find_if(outputTensors.begin(), outputTensors.end(), [=] (const auto& outputTensor)
        {
            return outputTensor.first == layerBindingId;
        });

        if (it == outputTensors.end())
        {
            outputIndex++;
            continue;
        }

        const auto outputTensor = *it;
        try
        {
            // Check if the output memory can be imported
            if (inputTensorHandle->CanBeImported(outputTensor.second.GetMemoryArea(), forceImportMemorySource)
                && inputTensorHandle->Import(outputTensor.second.GetMemoryArea(), forceImportMemorySource))
            {
                importedOutputs.push_back(outputIndex);
            }
        }
        catch(const MemoryImportException& exception)
        {
            ARMNN_LOG(error) << "An error occurred attempting to import output_"
                             << outputIndex << " : " << exception.what();
        }
        outputIndex++;
    }
    return importedOutputs;
}

void LoadedNetwork::ClearImportedInputs(const std::vector<ImportedInputId> inputIds)
{
    for (auto id : inputIds)
    {
        if (id > m_PreImportedInputHandles.size())
        {
            throw InvalidArgumentException(fmt::format("ClearImportedInputs::Unknown ImportedInputId: {}", id));
        }

        auto& importedTensorHandle = m_PreImportedInputHandles[id].m_TensorHandle;
        if (!importedTensorHandle)
        {
            throw InvalidArgumentException(
                    fmt::format("ClearImportedInputs::ImportedInput with id: {} has already been deleted", id));
        }
        // Call Unimport then destroy the tensorHandle
        importedTensorHandle->Unimport();
        importedTensorHandle = {};
    }
}

void LoadedNetwork::ClearImportedOutputs(const std::vector<ImportedOutputId> outputIds)
{
    for (auto id : outputIds)
    {
        if (id > m_PreImportedOutputHandles.size())
        {
            throw InvalidArgumentException(fmt::format("ClearImportedOutputs::Unknown ImportedOutputId: {}", id));
        }

        auto& importedTensorHandle = m_PreImportedOutputHandles[id].m_TensorHandle;
        if (!importedTensorHandle)
        {
            throw InvalidArgumentException(
                    fmt::format("ClearImportedOutputs::ImportedOutput with id: {} has already been deleted", id));
        }
        // Call Unimport then destroy the tensorHandle
        importedTensorHandle->Unimport();
        importedTensorHandle = {};
    }
}

void LoadedNetwork::RegisterDebugCallback(const DebugCallbackFunction& func)
{
    for (auto&& workloadPtr: m_WorkloadQueue)
    {
        workloadPtr.get()->RegisterDebugCallback(func);
    }
}

void LoadedNetwork::CreateMemoryProfile()
{
    // Finds the first TensorHandle ancestor of a SubTensorHandle. If the ITensorHandle provided
    // is a TensorHandle, the function just returns it
    auto TraceSubTensorHandleAncestry = [](ITensorHandle* const subTensorHandle)
    {
        ITensorHandle* ancestor = subTensorHandle;
        while (ancestor && ancestor->GetParent())
        {
            ancestor = ancestor->GetParent();
        }
        return ancestor;
    };

    struct PartialBlock
    {
        unsigned int m_StartOfLife;
        unsigned int m_Lifetime;

        size_t m_MemSize;
        unsigned int m_Index;

        BackendId m_BackendId;
    };

    auto align = [](size_t numToAlign)
    {
        const size_t alignment = sizeof(float);
        return ((numToAlign + alignment - 1) / alignment) * alignment;
    };

    std::unordered_map<ITensorHandle*, PartialBlock> memBlockTrackerMap;

    const bool inputImportingEnabled = m_NetworkProperties.m_InputSource != MemorySource::Undefined;
    const bool outputImportingEnabled = m_NetworkProperties.m_OutputSource != MemorySource::Undefined;

    unsigned int timestep = 0;
    unsigned int outputIndex = 0;
    Graph& order = m_OptimizedNetwork->pOptimizedNetworkImpl->GetGraph().TopologicalSort();

    for (auto&& layer : order)
    {
        const LayerType& layerType = layer->GetType();
        // Don't manage memory if importing.
        if (layerType == LayerType::Input && inputImportingEnabled)
        {
            continue;
        }
        // Don't manage memory if importing.
        if (layerType == LayerType::Output && outputImportingEnabled
            && layer->GetInputSlot(0).GetConnectedOutputSlot()->GetNumConnections() == 1)
        {
            continue;
        }
        // Because Constant Layer memory can not be shared, the memory must persist for the lifetime of execution,
        // management is done separately.
        if (layerType == LayerType::Constant)
        {
            continue;
        }

        BackendId backendId = layer->GetBackendId();
        for (auto& outputSlot : layer->GetOutputSlots())
        {
            if (!m_SupportsExternallyManagedMemory[backendId])
            {
                continue;
            }

            ITensorHandle* tensorHandle = outputSlot.GetOutputHandler().GetData();
            tensorHandle = TraceSubTensorHandleAncestry(tensorHandle);

            if (memBlockTrackerMap.find(tensorHandle) == memBlockTrackerMap.end())
            {
                PartialBlock partialBlock;

                partialBlock.m_StartOfLife = timestep;

                size_t alignedSize = align(outputSlot.GetOutputHandler().GetTensorInfo().GetNumBytes());
                partialBlock.m_MemSize = alignedSize;
                partialBlock.m_Index = outputIndex++;
                partialBlock.m_Lifetime = outputSlot.GetNumConnections();
                partialBlock.m_BackendId = backendId;

                if (partialBlock.m_Lifetime == 0)
                {
                    m_MemBlockMap[partialBlock.m_BackendId].emplace_back(partialBlock.m_StartOfLife,
                                                                         partialBlock.m_StartOfLife,
                                                                         partialBlock.m_MemSize,
                                                                         0,
                                                                         partialBlock.m_Index);
                }
                else
                {
                    memBlockTrackerMap[tensorHandle] = partialBlock;
                }
                m_Tensorhandles.push_back(tensorHandle);

            }
            else
            {
                memBlockTrackerMap.at(tensorHandle).m_Lifetime += outputSlot.GetNumConnections();
            }
        }

        for (auto& inputSlot : layer->GetInputSlots())
        {
            const Layer& connectedInputLayer = inputSlot.GetConnectedOutputSlot()->GetOwningLayer();
            const LayerType& owningLayerType = connectedInputLayer.GetType();

            if (owningLayerType == LayerType::Constant)
            {
                continue;
            }
            if (inputImportingEnabled && owningLayerType == LayerType::Input)
            {
                continue;
            }
            if (!m_SupportsExternallyManagedMemory[connectedInputLayer.GetBackendId()])
            {
                continue;
            }

            auto outputSlot = inputSlot.GetConnectedOutputSlot();

            ITensorHandle* tensorHandle = outputSlot->GetOutputHandler().GetData();
            tensorHandle = TraceSubTensorHandleAncestry(tensorHandle);

            PartialBlock& partialBlock = memBlockTrackerMap.at(tensorHandle);

            auto& lifetime = partialBlock.m_Lifetime;
            --lifetime;

            if (lifetime == 0)
            {
                m_MemBlockMap[partialBlock.m_BackendId].emplace_back(partialBlock.m_StartOfLife,
                                                                     timestep,
                                                                     partialBlock.m_MemSize,
                                                                     0,
                                                                     partialBlock.m_Index);
            }
        }
        ++timestep;
    }

}

std::unique_ptr<MemoryManager> LoadedNetwork::CreateExternalMemoryManger(
        std::vector<std::pair<std::shared_ptr<TensorMemory>, MemorySource>>& tensorMemoryVec)
{
    std::unique_ptr<MemoryManager> memoryManager = std::make_unique<MemoryManager>();
    auto allocatorMap = BackendRegistryInstance().GetAllocators();

    for (auto& backend : m_MemBinMap)
    {
        std::vector<BufferStorage> bufferStorageVec;

        std::shared_ptr<ICustomAllocator> backendAllocator;
        if (allocatorMap.find(backend.first) != allocatorMap.end())
        {
            backendAllocator = allocatorMap[backend.first];
        }
        else
        {
            backendAllocator = m_Backends[backend.first]->GetDefaultAllocator();
        }

        for (auto& memBin : backend.second)
        {
            BufferStorage bufferStorage;
            bufferStorage.m_BufferSize = memBin.m_MemSize;
            bufferStorage.m_TensorMemoryVector.reserve(memBin.m_MemBlocks.size());

            for (auto& memBlock : memBin.m_MemBlocks)
            {
                auto tensorMemory = std::make_shared<TensorMemory>(TensorMemory{memBlock.m_Offset, memBlock.m_Index});

                tensorMemoryVec.emplace_back(tensorMemory, backendAllocator->GetMemorySourceType());
                bufferStorage.m_TensorMemoryVector.emplace_back(tensorMemory);
            }

            bufferStorageVec.emplace_back(std::move(bufferStorage));
        }

        memoryManager->StoreMemToAllocate(bufferStorageVec, backendAllocator, 4);
    }

    return memoryManager;
}

LayerBindingId LoadedNetwork::ValidateImportedInputID(ImportedInputId id)
{
    try
    {
        const auto& importedTensorHandlePin = m_PreImportedInputHandles.at(id);
        if (!importedTensorHandlePin.m_TensorHandle)
        {
            throw InvalidArgumentException(fmt::format("LoadedNetwork::Execute:"
                                                       "PreImportedInput: {} has been deleted", id));
        }
        return importedTensorHandlePin.m_LayerBindingId;
    }
    catch (const std::out_of_range&)
    {
        throw InvalidArgumentException(fmt::format("LoadedNetwork::Execute: Unknown ImportedInputId: {}", id));
    }
}

LayerBindingId LoadedNetwork::ValidateImportedOutputID(ImportedOutputId id)
{
    try
    {
        const auto& importedTensorHandlePin = m_PreImportedOutputHandles.at(id);
        if (!importedTensorHandlePin.m_TensorHandle)
        {
            throw InvalidArgumentException(fmt::format("LoadedNetwork::Execute: "
                                                       "PreImportedOutput: {} has been deleted", id));
        }
        return importedTensorHandlePin.m_LayerBindingId;
    }
    catch (const std::out_of_range&)
    {
        throw InvalidArgumentException(fmt::format("LoadedNetwork::Execute: Unknown ImportedOutputId: {}", id));
    }
}

}
