//
// Copyright © 2017 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "LoadedNetwork.hpp"
#include "Layer.hpp"
#include "Graph.hpp"
#include <Processes.hpp>
#include "Profiling.hpp"
#include "HeapProfiling.hpp"
#include "WorkingMemHandle.hpp"

#include <armnn/BackendRegistry.hpp>
#include <armnn/Logging.hpp>
#include <armnn/utility/Assert.hpp>

#include <backendsCommon/TensorHandle.hpp>
#include <armnn/backends/IMemoryManager.hpp>
#include <backendsCommon/MemCopyWorkload.hpp>
#include <backendsCommon/MemSyncWorkload.hpp>
#include <armnn/BackendHelper.hpp>

#include <fmt/format.h>

namespace armnn
{

using namespace std;
using namespace armnn::profiling;

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
        ARMNN_ASSERT(source != NULL);
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

std::unique_ptr<LoadedNetwork> LoadedNetwork::MakeLoadedNetwork(std::unique_ptr<IOptimizedNetwork> net,
                                                                std::string& errorMessage,
                                                                const INetworkProperties& networkProperties,
                                                                profiling::ProfilingService&  profilingService)
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
                             profiling::ProfilingService&  profilingService) :
                             m_OptimizedNetwork(std::move(net)),
                             m_NetworkProperties(networkProperties),
                             m_TensorHandleFactoryRegistry(),
                             m_ProfilingService(profilingService)
{
    // Create a profiler and register it for the current thread.
    m_Profiler = std::make_shared<IProfiler>();
    ProfilerManager::GetInstance().RegisterProfiler(m_Profiler.get());

    m_Profiler->EnableProfiling(networkProperties.m_ProfilingEnabled);

    m_Profiler->EnableNetworkDetailsToStdOut(networkProperties.m_OutputNetworkDetailsMethod);

    Graph& order = m_OptimizedNetwork->pOptimizedNetworkImpl->GetGraph().TopologicalSort();
    //First create tensor handlers, backends and workload factories.
    //Handlers are created before workloads are.
    //Because workload creation can modify some of the handlers,
    //(for example the splitter and concat layers).
    for (auto&& layer : order)
    {
        auto const& backendId = layer->GetBackendId();
        if (m_Backends.count(backendId) == 0)
        {
            auto createBackend = BackendRegistryInstance().GetFactory(backendId);
            auto it = m_Backends.emplace(std::make_pair(backendId, createBackend()));

            IBackendInternal* backend = it.first->second.get();

            if (networkProperties.m_AsyncEnabled &&
                !HasCapability(BackendOptions::BackendOption{"AsyncExecution", true}, backend->GetCapabilities()))
            {
                std::string er = backend->GetId();
                er += " does not support AsyncExecution";
                throw BackendCapabilityException(er);
            }

            if (backend->SupportsTensorAllocatorAPI())
            {
                auto workloadFactory = backend->CreateWorkloadFactory(
                    m_TensorHandleFactoryRegistry,
                    m_OptimizedNetwork->pOptimizedNetworkImpl->GetModelOptions(),
                    static_cast<MemorySourceFlags>(m_NetworkProperties.m_InputSource),
                    static_cast<MemorySourceFlags>(m_NetworkProperties.m_OutputSource));
                m_WorkloadFactories.emplace(
                    std::make_pair(backendId, std::make_pair(std::move(workloadFactory), nullptr)));
            }
            else
            {
                IBackendInternal::IMemoryManagerSharedPtr memoryManager = backend->CreateMemoryManager();
                auto workloadFactory = backend->CreateWorkloadFactory(
                    memoryManager, m_OptimizedNetwork->pOptimizedNetworkImpl->GetModelOptions());

                m_WorkloadFactories.emplace(
                    std::make_pair(backendId, std::make_pair(std::move(workloadFactory), memoryManager)));
            }
        }
    }

    if (!networkProperties.m_AsyncEnabled)
    {
        for (auto&& layer : order)
        {
            auto& workloadFactory = GetWorkloadFactory(*layer);

            switch (layer->GetType())
            {
                case LayerType::Input:
                case LayerType::MemImport:
                {
                    // If IsImportEnabled is true then we need to set IsMemoryManaged
                    // to false when creating TensorHandles
                    layer->CreateTensorHandles(m_TensorHandleFactoryRegistry,
                                               workloadFactory,
                                               !m_NetworkProperties.m_ImportEnabled);
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
                                                   !m_NetworkProperties.m_ExportEnabled);
                    }
                    else
                    {
                        layer->CreateTensorHandles(m_TensorHandleFactoryRegistry, workloadFactory);
                    }
                }
            }
        }
    }

    ProfilingGuid networkGuid = m_OptimizedNetwork->GetGuid();
    std::unique_ptr<TimelineUtilityMethods> timelineUtils =
                        TimelineUtilityMethods::GetTimelineUtils(m_ProfilingService);
    if (timelineUtils)
    {
        timelineUtils->CreateTypedEntity(networkGuid, LabelsAndEventClasses::NETWORK_GUID);
        // Mark the network with a start of life event
        timelineUtils->RecordEvent(networkGuid, LabelsAndEventClasses::ARMNN_PROFILING_SOL_EVENT_CLASS);
        // and with the process ID
        int processID = armnnUtils::Processes::GetCurrentId();
        std::stringstream ss;
        ss << processID;
        timelineUtils->MarkEntityWithLabel(networkGuid, ss.str(), LabelsAndEventClasses::PROCESS_ID_GUID);
    }

    //Then create workloads.
    for (auto&& layer : order)
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

                // For async networks ConstantWorkloads are managed exclusively by LoadedNetwork
                // and are separated out from the other workloads
                if (networkProperties.m_AsyncEnabled && layer->GetType() == LayerType::Constant)
                {
                    m_ConstantWorkloads[layer->GetGuid()] = std::move(workload);
                }
                else
                {
                    m_WorkloadQueue.push_back(move(workload));
                }

                // release the constant data in the layer..
                layer->ReleaseConstantData();
                break;
            }
        }
    }

    for (auto&& workloadFactory : m_WorkloadFactories)
    {
        workloadFactory.second.first->AfterWorkloadsCreated();
    }

    if (timelineUtils)
    {
        // Commit to send the post-optimisation network structure
        timelineUtils->Commit();
    }

    if (!networkProperties.m_AsyncEnabled)
    {
        // Set up memory.
        m_OptimizedNetwork->pOptimizedNetworkImpl->GetGraph().AllocateDynamicBuffers();

        // Now that the intermediate tensor memory has been set-up,
        // do any post allocation configuration for each workload.
        for (auto &workload : m_WorkloadQueue)
        {
            workload->PostAllocationConfigure();
        }
    }
    else
    {
        AllocateAndExecuteConstantWorkloads();
    }
}

void LoadedNetwork::AllocateAndExecuteConstantWorkloads()
{
    Graph& order = m_OptimizedNetwork->pOptimizedNetworkImpl->GetGraph();
    for (auto&& layer : order)
    {
        if (layer->GetType() == LayerType::Constant)
        {
            const auto& outSlot = layer->GetOutputSlots()[0];
            const auto factoryId = outSlot.GetTensorHandleFactoryId();
            ARMNN_ASSERT(factoryId != ITensorHandleFactory::LegacyFactoryId);
            auto& workloadFactory = GetWorkloadFactory(*layer);

            layer->CreateTensorHandles(m_TensorHandleFactoryRegistry, workloadFactory);
            ITensorHandle* tensorHandle = outSlot.GetOutputHandler().GetData();

            m_ConstantTensorHandles[layer->GetGuid()] = tensorHandle;
            tensorHandle->Allocate();

            WorkingMemDescriptor memDesc;
            memDesc.m_Outputs.push_back(tensorHandle);
            m_ConstantWorkloads[layer->GetGuid()]->ExecuteAsync(memDesc);
        }
    }
}


void LoadedNetwork::SendNetworkStructure()
{
    Graph& order = m_OptimizedNetwork->pOptimizedNetworkImpl->GetGraph().TopologicalSort();
    ProfilingGuid networkGuid = m_OptimizedNetwork->GetGuid();

    std::unique_ptr<TimelineUtilityMethods> timelineUtils =
                        TimelineUtilityMethods::GetTimelineUtils(m_ProfilingService);

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

profiling::ProfilingGuid LoadedNetwork::GetNetworkGuid()
{
    return m_OptimizedNetwork->GetGuid();
}

TensorInfo LoadedNetwork::GetInputTensorInfo(LayerBindingId layerId) const
{
    for (auto&& inputLayer : m_OptimizedNetwork->pOptimizedNetworkImpl->GetGraph().GetInputLayers())
    {
        ARMNN_ASSERT_MSG(inputLayer->GetNumOutputSlots() == 1, "Input layer should have exactly 1 output slot");
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
        ARMNN_ASSERT_MSG(outputLayer->GetNumInputSlots() == 1, "Output layer should have exactly 1 input slot");
        ARMNN_ASSERT_MSG(outputLayer->GetInputSlot(0).GetConnection(), "Input slot on Output layer must be connected");
        if (outputLayer->GetBindingId() == layerId)
        {
            return outputLayer->GetInputSlot(0).GetConnection()->GetTensorInfo();
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

    workloadFactory = it->second.first.get();

    ARMNN_ASSERT_MSG(workloadFactory, "No workload factory");

    std::string reasonIfUnsupported;
    ARMNN_ASSERT_MSG(IWorkloadFactory::IsLayerSupported(layer,
                                                        {},
                                                        reasonIfUnsupported,
                                                        m_OptimizedNetwork->pOptimizedNetworkImpl->GetModelOptions()),
        "Factory does not support layer");
    IgnoreUnused(reasonIfUnsupported);
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
                                      const OutputTensors& outputTensors)
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

    if (graph.GetNumInputs() != inputTensors.size())
    {
        throw InvalidArgumentException("Number of inputs provided does not match network.");
    }

    // For each input to the network, call EnqueueInput with the data passed by the user.
    {
        ARMNN_SCOPED_PROFILING_EVENT(Compute::Undefined, "PrepareInputs");
        m_InputQueue.clear();
        m_InputQueue.reserve(graph.GetNumInputs());
        for (const BindableLayer* inputLayer : graph.GetInputLayers())
        {
            const TensorPin& pin = workloadData.GetInputTensorPin(inputLayer->GetBindingId());
            EnqueueInput(*inputLayer, pin.GetTensorHandle(), pin.GetTensorInfo());
        }
    }

    // For each output to the network, call EnqueueOutput with the data passed by the user.
    {
        ARMNN_SCOPED_PROFILING_EVENT(Compute::Undefined, "PrepareOutputs");
        m_OutputQueue.clear();
        m_OutputQueue.reserve(graph.GetNumOutputs());
        for (const BindableLayer* outputLayer : graph.GetOutputLayers())
        {
            const TensorPin& pin = workloadData.GetOutputTensorPin(outputLayer->GetBindingId());
            EnqueueOutput(*outputLayer, pin.GetTensorHandle(), pin.GetTensorInfo());
        }
    }

    std::unique_ptr<TimelineUtilityMethods> timelineUtils =
                        TimelineUtilityMethods::GetTimelineUtils(m_ProfilingService);
    ProfilingGuid inferenceGuid = m_ProfilingService.GetNextGuid();
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
        if (m_ProfilingService.IsProfilingEnabled())
        {
            m_ProfilingService.IncrementCounterValue(armnn::profiling::INFERENCES_RUN);
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

    ARMNN_ASSERT_MSG(layer.GetNumOutputSlots() == 1, "Can only handle Input Layer with one output");
    const OutputHandler& handler = layer.GetOutputHandler();
    const TensorInfo& outputTensorInfo = handler.GetTensorInfo();
    ITensorHandle* outputTensorHandle = handler.GetData();
    ARMNN_ASSERT_MSG(outputTensorHandle != nullptr,
                     "Data should have been allocated.");
    inputQueueDescriptor.m_Outputs.push_back(outputTensorHandle);
    info.m_OutputTensorInfos.push_back(outputTensorInfo);

    MemorySourceFlags importFlags = outputTensorHandle->GetImportFlags();
    bool needMemCopy = true;
    if (m_NetworkProperties.m_ImportEnabled)  // Try import the input tensor
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

        ARMNN_ASSERT_MSG(inputWorkload, "No input workload created");

        std::unique_ptr<TimelineUtilityMethods> timelineUtils =
                            TimelineUtilityMethods::GetTimelineUtils(m_ProfilingService);
        if (timelineUtils)
        {
            // Add Input Workload to the post-optimisation network structure
            AddWorkloadStructure(timelineUtils, inputWorkload, layer);
            timelineUtils->Commit();
        }

        m_InputQueue.push_back(move(inputWorkload));
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

    ARMNN_ASSERT_MSG(layer.GetNumInputSlots() == 1, "Output Layer should have exactly one input.");

    // Gets the output handler from the previous node.
    const OutputHandler& outputHandler = layer.GetInputSlots()[0].GetConnectedOutputSlot()->GetOutputHandler();

    const TensorInfo& inputTensorInfo = outputHandler.GetTensorInfo();
    ITensorHandle* inputTensorHandle = outputHandler.GetData();
    ARMNN_ASSERT_MSG(inputTensorHandle != nullptr, "Data should have been allocated.");

    // Try import the output tensor.
    // Note: We can only import the output pointer if all of the following  hold true:
    // a) The imported pointer is aligned sufficiently
    // b) The tensor has zero padding
    // c) There is only one connection to the OutputSlot and it is to an OutputLayer.
    // d) The output pointer is allocated via malloc. (Other types will be supported in a later release)
    // e) m_IsExportEnabled must be set to true
    bool needMemCopy = true;
    if (m_NetworkProperties.m_ExportEnabled &&
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
                    ARMNN_ASSERT_MSG(syncWorkload, "No sync workload created");
                    m_OutputQueue.push_back(move(syncWorkload));
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
        ARMNN_ASSERT_MSG(outputWorkload, "No output workload created");

        std::unique_ptr<TimelineUtilityMethods> timelineUtils =
            TimelineUtilityMethods::GetTimelineUtils(m_ProfilingService);
        if (timelineUtils)
        {
            // Add Output Workload to the post-optimisation network structure
            AddWorkloadStructure(timelineUtils, outputWorkload, layer);
            timelineUtils->Commit();
        }

        m_OutputQueue.push_back(move(outputWorkload));
    }
}

void LoadedNetwork::AllocateWorkingMemory(std::lock_guard<std::mutex>& lock)
{
    ARMNN_SCOPED_PROFILING_EVENT(Compute::Undefined, "Working Memory Allocation");

    // this unused parameter makes sure we can only call this function with a valid lock
    IgnoreUnused(lock);

    if (m_IsWorkingMemAllocated)
    {
        return;
    }
    for (auto&& workloadFactory : m_WorkloadFactories)
    {
        IBackendInternal::IMemoryManagerSharedPtr memoryManager = workloadFactory.second.second;
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
    std::lock_guard<std::mutex> lockGuard(m_WorkingMemMutex);
    if (!m_IsWorkingMemAllocated)
    {
        return;
    }
    // Informs the memory managers to release memory in it's respective memory group
    for (auto&& workloadFactory : m_WorkloadFactories)
    {
        IBackendInternal::IMemoryManagerSharedPtr memoryManager = workloadFactory.second.second;
        if (memoryManager)
        {
            memoryManager->Release();
        }
    }
    m_TensorHandleFactoryRegistry.ReleaseMemory();
    m_IsWorkingMemAllocated = false;
}

bool LoadedNetwork::Execute(std::unique_ptr<TimelineUtilityMethods>& timelineUtils,
                            profiling::ProfilingGuid inferenceGuid)
{
    bool success = true;

    auto Fail = [&](const std::exception& error)
    {
        ARMNN_LOG(error) << "An error occurred attempting to execute a workload: " << error.what();
        success = false;
    };

    try
    {
        std::lock_guard<std::mutex> lockGuard(m_WorkingMemMutex);
        AllocateWorkingMemory(lockGuard);

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
                workload->Execute();
                if(timelineUtils)
                {
                    timelineUtils->RecordEndOfLifeEvent(workloadInferenceID);
                }
            }
        };

        ExecuteQueue(m_InputQueue);
        ExecuteQueue(m_WorkloadQueue);
        ExecuteQueue(m_OutputQueue);
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
    if (m_NetworkProperties.m_ImportEnabled)  // Try import the input tensor
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
    ARMNN_ASSERT_MSG(outputTensorHandle != nullptr, "Data should have been allocated.");
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

std::vector<ImportedInputId> LoadedNetwork::ImportInputs(const InputTensors& inputTensors)
{
    if (!m_NetworkProperties.m_ImportEnabled)  // Try import the input tensor
    {
        throw MemoryImportException("ImportInputs: Memory Import failed, NetworkProperties.m_ImportEnabled");
    }

    std::vector<ImportedInputId> importedInputs;
    Graph& graph = m_OptimizedNetwork->pOptimizedNetworkImpl->GetGraph().TopologicalSort();

    for (auto inputTensor : inputTensors)
    {
        auto layerBindingId = inputTensor.first;
        auto it = std::find_if(graph.GetInputLayers().begin(), graph.GetInputLayers().end(), [=](auto* layer)
        {
            return layer->GetBindingId() == layerBindingId;
        });

        if (it == graph.GetInputLayers().end())
        {
            throw MemoryImportException(fmt::format("ImportInputs: Memory Import failed, unknown LayerBindingId: {}",
                                                    layerBindingId));
        }

        const Layer* layer = *it;
        if (layer->GetType() != LayerType::Input)
        {
            throw InvalidArgumentException("ImportInputs: given layer not an InputLayer");
        }

        auto& backend = m_Backends.at(layer->GetBackendId());
        if (!HasCapability(BackendOptions::BackendOption{"PreImportIOTensors", true}, backend->GetCapabilities()))
        {
            std::string er = backend->GetId();
            er += " does not have PreImportIOTensors capability";
            throw BackendCapabilityException(er);
        }

        const OutputSlot& outputSlot = layer->GetOutputSlots()[0];

        ITensorHandleFactory::FactoryId factoryId = outputSlot.GetTensorHandleFactoryId();
        const TensorInfo& tensorInfo = outputSlot.GetTensorInfo();

        ITensorHandleFactory* handleFactory = m_TensorHandleFactoryRegistry.GetFactory(factoryId);
        ARMNN_ASSERT(handleFactory);

        ImportedTensorHandlePin importedTensorHandlePin{layerBindingId,
                                                        handleFactory->CreateTensorHandle(tensorInfo, false)};

        ITensorHandle* tensorHandle = importedTensorHandlePin.m_TensorHandle.get();

        if (!CheckFlag(tensorHandle->GetImportFlags(), m_NetworkProperties.m_InputSource))
        {
            throw MemoryImportException(
                fmt::format("ImportInputs: Memory Import failed, backend: {} does not support importing from source {}"
                            , factoryId, m_NetworkProperties.m_InputSource));
        }

        std::unique_ptr<ITensorHandle> passThroughTensorHandle =
                std::make_unique<ConstPassthroughTensorHandle>(inputTensor.second.GetInfo(),
                                                               inputTensor.second.GetMemoryArea());

        if (tensorHandle->Import(passThroughTensorHandle->Map(), m_NetworkProperties.m_InputSource))
        {
            importedInputs.push_back(m_CurImportedInputId++);
            passThroughTensorHandle->Unmap();
        }
        else
        {
            passThroughTensorHandle->Unmap();
            throw MemoryImportException("ImportInputs: Memory Import failed");
        }

        m_PreImportedInputHandles.push_back(std::move(importedTensorHandlePin));
    }

    return importedInputs;
}

std::vector<ImportedOutputId> LoadedNetwork::ImportOutputs(const OutputTensors& outputTensors)
{
    if (!m_NetworkProperties.m_ExportEnabled)  // Try import the output tensor
    {
        throw MemoryImportException("ImportOutputs: Memory Import failed, NetworkProperties.m_ImportEnabled");
    }

    std::vector<ImportedOutputId> importedOutputs;
    Graph& graph = m_OptimizedNetwork->pOptimizedNetworkImpl->GetGraph().TopologicalSort();

    for (const auto& outputTensor : outputTensors)
    {
        auto layerBindingId = outputTensor.first;
        auto it = std::find_if(graph.GetOutputLayers().begin(), graph.GetOutputLayers().end(), [=](auto* layer)
        {
            return layer->GetBindingId() == layerBindingId;
        });

        if (it == graph.GetOutputLayers().end())
        {
            throw MemoryImportException(fmt::format("ImportOutputs: Memory Import failed, unknown LayerBindingId: {}",
                                                     layerBindingId));
        }

        const Layer* layer = *it;
        if (layer->GetType() != LayerType::Output)
        {
            throw InvalidArgumentException("ImportOutputs: given layer not an OutputLayer");
        }

        auto& backend = m_Backends.at(layer->GetBackendId());
        if (!HasCapability(BackendOptions::BackendOption{"PreImportIOTensors", true}, backend->GetCapabilities()))
        {
            std::string er = backend->GetId();
            er += " does not have PreImportIOTensors capability";
            throw BackendCapabilityException(er);
        }

        const InputSlot& inputSlot = layer->GetInputSlots()[0];
        ITensorHandleFactory::FactoryId factoryId = inputSlot.GetConnectedOutputSlot()->GetTensorHandleFactoryId();
        const TensorInfo& tensorInfo = inputSlot.GetConnectedOutputSlot()->GetTensorInfo();

        ITensorHandleFactory* handleFactory = m_TensorHandleFactoryRegistry.GetFactory(factoryId);
        ARMNN_ASSERT(handleFactory);

        ImportedTensorHandlePin importedTensorHandlePin{layerBindingId,
                                                        handleFactory->CreateTensorHandle(tensorInfo, false)};

        ITensorHandle* tensorHandle = importedTensorHandlePin.m_TensorHandle.get();

        if (!CheckFlag(tensorHandle->GetImportFlags(), m_NetworkProperties.m_OutputSource))
        {
            throw MemoryImportException(fmt::format("ImportInputs: Memory Import failed, backend: "
                                                    "{} does not support importing from source {}"
                                                    , factoryId, m_NetworkProperties.m_OutputSource));
        }

        if (tensorHandle->Import(outputTensor.second.GetMemoryArea(), m_NetworkProperties.m_OutputSource))
        {
            importedOutputs.push_back(m_CurImportedOutputId++);
        }
        else
        {
            throw MemoryImportException("ImportInputs: Memory Import failed");
        }

        m_PreImportedOutputHandles.push_back(std::move(importedTensorHandlePin));
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

Status LoadedNetwork::Execute(const InputTensors& inputTensors,
                              const OutputTensors& outputTensors,
                              IWorkingMemHandle& iWorkingMemHandle,
                              std::vector<ImportedInputId> preImportedInputs,
                              std::vector<ImportedOutputId> preImportedOutputs)
{
    const Graph& graph = m_OptimizedNetwork->pOptimizedNetworkImpl->GetGraph();

    if (inputTensors.size() + preImportedInputs.size() != graph.GetNumInputs())
    {
        if (preImportedInputs.empty())
        {
            throw InvalidArgumentException("LoadedNetwork::Execute: Number of inputs provided does not match network.");
        }
        else
        {
            throw InvalidArgumentException("LoadedNetwork::Execute: "
                                           "Number of inputs + preImportedInputs provided does not match network.");
        }
    }

    if (outputTensors.size() + preImportedOutputs.size() != graph.GetNumOutputs())
    {
        if (preImportedOutputs.empty())
        {
            throw InvalidArgumentException("LoadedNetwork::Execute: "
                                           "Number of outputs provided does not match network.");
        }
        else
        {
            throw InvalidArgumentException("LoadedNetwork::Execute: "
                                           "Number of outputs + preImportedOutputs provided does not match network.");
        }
    }

    WorkingMemHandle& workingMemHandle = dynamic_cast<WorkingMemHandle&>(iWorkingMemHandle);
    // Collect all the given LayerBindingIds and check them for duplicates and unknowns.
    std::vector<LayerBindingId>& bindingIds = workingMemHandle.GetBindingIdVector();
    unsigned int index = 0;
    for (auto pair : inputTensors)
    {
        bindingIds[index++] = pair.first;
    }
    for (ImportedInputId id : preImportedInputs)
    {
        bindingIds[index++] = ValidateImportedInputID(id);
    }
    for (auto pair : outputTensors)
    {
        bindingIds[index++] = pair.first;
    }
    for (ImportedOutputId id : preImportedOutputs)
    {
        bindingIds[index++] = ValidateImportedOutputID(id);
    }

    workingMemHandle.ValidateBindingIds();

    auto resetMemHandle = [&]()
    {
        for (ImportedInputId id: preImportedInputs)
        {
            const LayerBindingId layerBindingId = m_PreImportedInputHandles[id].m_LayerBindingId;

            auto inputHandle = workingMemHandle.GetInputHandle(layerBindingId);
            auto inputConnections = workingMemHandle.GetInputConnections(layerBindingId);
            for (auto it : inputConnections)
            {
                *it = inputHandle;
            }
        }

        for (ImportedOutputId id: preImportedOutputs)
        {
            const LayerBindingId layerBindingId = m_PreImportedOutputHandles[id].m_LayerBindingId;

            auto outputHandle = workingMemHandle.GetOutputHandle(layerBindingId);
            auto outputConnections = workingMemHandle.GetOutputConnection(layerBindingId);

            for (auto it : outputConnections)
            {
                *it = outputHandle;
            }
        }
    };

    std::unique_ptr<profiling::TimelineUtilityMethods> timelineUtils =
            profiling::TimelineUtilityMethods::GetTimelineUtils(m_ProfilingService);
    profiling::ProfilingGuid inferenceGuid = m_ProfilingService.GetNextGuid();
    if (timelineUtils)
    {
        // Add inference timeline trace if profiling is enabled.
        profiling::ProfilingGuid networkGuid = m_OptimizedNetwork->GetGuid();
        timelineUtils->CreateTypedEntity(inferenceGuid, profiling::LabelsAndEventClasses::INFERENCE_GUID);
        timelineUtils->CreateRelationship(profiling::ProfilingRelationshipType::RetentionLink,
                                          networkGuid,
                                          inferenceGuid,
                                          profiling::LabelsAndEventClasses::EXECUTION_OF_GUID);
        timelineUtils->RecordEvent(inferenceGuid, profiling::LabelsAndEventClasses::ARMNN_PROFILING_SOL_EVENT_CLASS);
    }

    bool executionSucceeded = true;

    if (timelineUtils)
    {
        // Add end of life of the inference timeline if profiling is enabled.
        timelineUtils->RecordEvent(inferenceGuid, profiling::LabelsAndEventClasses::ARMNN_PROFILING_EOL_EVENT_CLASS);
        timelineUtils->Commit();
    }

    if (!workingMemHandle.IsAllocated())
    {
        workingMemHandle.Allocate();
    }

    {
        ARMNN_SCOPED_PROFILING_EVENT(Compute::Undefined, "PrepareInputs");
        for (auto pair : inputTensors)
        {
            EnqueueInput(pair.second, workingMemHandle.GetInputHandle(pair.first));
        }

        // Swap in the pre-imported inputs if any
        for (ImportedInputId id : preImportedInputs)
        {
            const ImportedTensorHandlePin& importedInputPin = m_PreImportedInputHandles[id];
            const LayerBindingId layerBindingId = m_PreImportedInputHandles[id].m_LayerBindingId;
            const auto& preimportedHandle = importedInputPin.m_TensorHandle;

            auto inputConnections = workingMemHandle.GetInputConnections(layerBindingId);
            for (auto it : inputConnections)
            {
                *it = preimportedHandle.get();
            }
        }
    }
    {
        ARMNN_SCOPED_PROFILING_EVENT(Compute::Undefined, "PrepareOutputs");
        if (m_NetworkProperties.m_ExportEnabled)
        {
            for (auto pair: outputTensors)
            {
                ImportOutputTensor(pair.second, workingMemHandle.GetOutputHandle(pair.first));
            }
        }

        for (ImportedOutputId id : preImportedOutputs)
        {
            const ImportedTensorHandlePin& importedOutputPin = m_PreImportedOutputHandles[id];
            const LayerBindingId layerBindingId = m_PreImportedOutputHandles[id].m_LayerBindingId;
            const auto& preimportedHandle = importedOutputPin.m_TensorHandle;

            auto outputConnections = workingMemHandle.GetOutputConnection(layerBindingId);

            for (auto it : outputConnections)
            {
                *it = preimportedHandle.get();
            }
        }
    }

    auto Fail = [&](const std::exception& error)
    {
        ARMNN_LOG(error) << "An error occurred attempting to execute a workload: " << error.what();
        executionSucceeded = false;
    };
    profiling::ProfilingDynamicGuid workloadInferenceID(0);

    try
    {
        for (unsigned int i = 0; i < m_WorkloadQueue.size(); ++i)
        {
            auto& workload = m_WorkloadQueue[i];
            if (timelineUtils)
            {
                workloadInferenceID = timelineUtils->RecordWorkloadInferenceAndStartOfLifeEvent(workload->GetGuid(),
                                                                                                inferenceGuid);
            }
            workload->ExecuteAsync(workingMemHandle.GetWorkingMemDescriptorAt(i));

            if (timelineUtils)
            {
                timelineUtils->RecordEndOfLifeEvent(workloadInferenceID);
            }
        }
    }
    catch (const RuntimeException& error)
    {
        resetMemHandle();
        Fail(error);
    }
    catch (const std::runtime_error& error)
    {
        resetMemHandle();
        Fail(error);
    }
    catch (...)
    {
        resetMemHandle();
        throw;
    }

    if (!m_NetworkProperties.m_ExportEnabled)
    {
        for (auto pair: outputTensors)
        {
            CopyToOutputTensor(pair.second, workingMemHandle.GetOutputHandle(pair.first));
        }
    }
    else
    {
       ARMNN_SCOPED_PROFILING_EVENT(Compute::Undefined, "SyncMemGeneric_Execute");
       workingMemHandle.MemSyncOutputs();
    }

    resetMemHandle();

    return executionSucceeded ? Status::Success : Status::Failure;
}

/// Create a new unique WorkingMemHandle object. Create multiple handles if you wish to have
/// overlapped Execution by calling this function from different threads.
std::unique_ptr<IWorkingMemHandle> LoadedNetwork::CreateWorkingMemHandle(NetworkId networkId)
{
    Graph& order = m_OptimizedNetwork->pOptimizedNetworkImpl->GetGraph();
    std::unordered_map<LayerGuid, std::vector<std::unique_ptr<ITensorHandle> > > tensorHandleMap;
    std::vector<WorkingMemDescriptor> workingMemDescriptors;
    std::unordered_map<LayerGuid, WorkingMemDescriptor> workingMemDescriptorMap;
    TensorHandleFactoryRegistry tensorHandleFactoryRegistry;
    WorkloadFactoryMap workloadFactoryMap;

    std::vector<std::shared_ptr<IMemoryManager>> memoryManagers;

    for (auto const& backend : m_Backends)
    {
        if (backend.second->SupportsTensorAllocatorAPI())
        {
            backend.second->RegisterTensorHandleFactories(
                tensorHandleFactoryRegistry,
                static_cast<MemorySourceFlags>(m_NetworkProperties.m_InputSource),
                static_cast<MemorySourceFlags>(m_NetworkProperties.m_OutputSource));
            memoryManagers.emplace_back(tensorHandleFactoryRegistry.GetMemoryManagers().back());
        }
        else
        {
            std::shared_ptr<IMemoryManager> memoryManager = backend.second->CreateMemoryManager();
            auto workloadFactory = backend.second->CreateWorkloadFactory(
                    memoryManager, m_OptimizedNetwork->pOptimizedNetworkImpl->GetModelOptions());

            workloadFactoryMap.emplace(
                    std::make_pair(backend.first, std::make_pair(std::move(workloadFactory), memoryManager)));
            memoryManagers.emplace_back(memoryManager);
        }
    }

    auto GetTensorHandle = [&](Layer* layer, const OutputSlot& outputSlot, bool isMemoryManaged)
    {
        ITensorHandleFactory::FactoryId factoryId = outputSlot.GetTensorHandleFactoryId();
        const TensorInfo& tensorInfo = outputSlot.GetTensorInfo();

        if (factoryId == ITensorHandleFactory::LegacyFactoryId)
        {
            BackendId id = layer->GetBackendId();
            ARMNN_NO_DEPRECATE_WARN_BEGIN
            return workloadFactoryMap.at(id).first->CreateTensorHandle(tensorInfo, isMemoryManaged);
            ARMNN_NO_DEPRECATE_WARN_END
        }
        else
        {
            ITensorHandleFactory* handleFactory = tensorHandleFactoryRegistry.GetFactory(factoryId);
            ARMNN_ASSERT(handleFactory);
            return handleFactory->CreateTensorHandle(tensorInfo, isMemoryManaged);
        }
    };

    struct HandleInfo
    {
        unsigned int m_ReferenceCount = 0;
        bool isInputLayerHandle = false;
        bool isOutputLayerHandle = false;

        WorkingMemHandle::InputMemDescriptorCoords m_InputMemDescriptorCoords;
        WorkingMemHandle::OutputMemDescriptorCoords m_OutputMemDescriptorCoords;
    };

    std::unordered_map<const ITensorHandle*, HandleInfo> handleReferenceCounts;
    unsigned int layerIndex = 0;
    for (auto&& layer : order)
    {
        // Constant layers execution and management is handled during loaded network construction
        if (layer->GetType() == LayerType::Constant)
        {
            continue;
        }

        WorkingMemDescriptor workingMemDescriptor;

        bool isMemoryManaged = true;
        bool isInputLayer = false;
        bool isOutputLayer = false;
        bool isConnectedToOutputLayer = false;

        if (layer->GetType() == LayerType::Input || layer->GetType() == LayerType::MemImport)
        {
            // Input layers/workloads will not be executed so the descriptor is not added to workingMemDescriptors
            // However we will still need to manage the tensorHandle
            isInputLayer = true;
            isMemoryManaged = !m_NetworkProperties.m_ImportEnabled;
        }
        else if (layer->GetType() == LayerType::Output)
        {
            isOutputLayer = true;
        }

        unsigned int slotIndex = 0;
        // Create a tensor handle for each output slot of a layer
        // Once we create it, we start managing its lifetime
        for (auto& slot : layer->GetOutputSlots())
        {
            for (unsigned int i = 0; i < slot.GetNumConnections(); ++i)
            {
                if ((slot.GetConnection(i)->GetOwningLayer().GetType() == LayerType::Output))
                {
                    if (!isConnectedToOutputLayer)
                    {
                        isConnectedToOutputLayer = true;
                        // If Export is enabled disable memory management, so we can export, otherwise we do a copy
                        isMemoryManaged = !m_NetworkProperties.m_ExportEnabled;
                    }
                    else
                    {
                        // Importing in this case would likely cause unexpected behaviour, so we disallow it.
                        ARMNN_LOG(warning) <<
                           fmt::format("Layer name: '{0}' guid: '{1}' has two or more OutputLayers connected to it. "
                                       "This will prevent importing on the connected OutputLayers.",
                                        layer->GetName(), layer->GetGuid());
                        isMemoryManaged = true;
                    }
                }
            }

            tensorHandleMap[layer->GetGuid()].emplace_back(GetTensorHandle(layer, slot, isMemoryManaged));
            ITensorHandle* tensorHandle = tensorHandleMap[layer->GetGuid()].back().get();

            workingMemDescriptor.m_Outputs.push_back(tensorHandle);
            tensorHandle->Manage();
            unsigned int numConnections = slot.GetNumConnections();
            ARMNN_ASSERT(numConnections != 0);

            HandleInfo& handleInfo = handleReferenceCounts[tensorHandle];
            handleInfo.m_ReferenceCount = numConnections;

            // Store the coordinates of the current layer's OutputSlot that is connected to the OutputLayer
            if (isConnectedToOutputLayer)
            {
                handleInfo.isOutputLayerHandle = true;
                handleInfo.m_OutputMemDescriptorCoords.m_OutputSlotCoords = {layerIndex, slotIndex};
            }
            // Store the LayerBindingId of the InputLayer
            if (isInputLayer)
            {
                handleInfo.isInputLayerHandle = true;
                LayerBindingId bindingId = static_cast<BindableLayer*>(layer)->GetBindingId();
                handleInfo.m_InputMemDescriptorCoords.m_LayerBindingId = bindingId;
            }
            slotIndex++;
        }
        // Loop through the input slots in the same layer and decrement the reference counter associated
        // to each tensor handle we encounter.
        // Once it reaches zero, the lifetime of the tensor handle has ended, and we mark its memory as available
        // so that the next tensor handle with a non overlapping lifetime can share its memory.
        for (auto& slot : layer->GetInputSlots())
        {
            ARMNN_ASSERT(slot.GetConnection());
            auto outputSlot = slot.GetConnectedOutputSlot();
            auto key = outputSlot->GetOwningLayer().GetGuid();

            // Constant layers execution and management is handled during loaded network construction
            auto found = m_ConstantTensorHandles.find(key);
            if (found != m_ConstantTensorHandles.end())
            {
                workingMemDescriptor.m_Inputs.push_back(found->second);
                continue;
            }

            auto search = tensorHandleMap.find(key);
            unsigned int index = outputSlot->CalculateIndexOnOwner();
            ITensorHandle* inputTensorHandle = search->second[index].get();
            workingMemDescriptor.m_Inputs.push_back(inputTensorHandle);

            HandleInfo& handleInfo = handleReferenceCounts.at(inputTensorHandle);

            // Store the LayerBindingId of the OutputLayer
            if (isOutputLayer)
            {
                LayerBindingId bindingId = static_cast<BindableLayer*>(layer)->GetBindingId();
                handleInfo.m_OutputMemDescriptorCoords.m_LayerBindingIds.push_back(bindingId);
                handleInfo.m_OutputMemDescriptorCoords.m_InputSlotCoords.push_back({layerIndex, slot.GetSlotIndex()});
            }
            // In this case the layer is not an Output Layer but shares it's input tensorhandle with an OutputLayer
            // It will need to be updated as well, if we swap out the tensorhandle
            else if (handleInfo.isOutputLayerHandle)
            {
                handleInfo.m_OutputMemDescriptorCoords.m_InputSlotCoords.push_back({layerIndex, slot.GetSlotIndex()});
            }

            // Store the coordinates of the InputSlots connected to the InputLayer
            // There can be more than one InputSlot connected to an InputLayer, so we use a vector
            if (handleInfo.isInputLayerHandle)
            {
                std::pair<LayerGuid, unsigned int> connectionLocation{layerIndex, slot.GetSlotIndex()};
                handleInfo.m_InputMemDescriptorCoords.m_InputSlotCoords.emplace_back(connectionLocation);
            }

            --handleInfo.m_ReferenceCount;
            if (handleInfo.m_ReferenceCount == 0u)
            {
                // Stop managing lifetime of tensor handle
                inputTensorHandle->Allocate();
            }
        }
        workingMemDescriptorMap.insert({layer->GetGuid(), workingMemDescriptor});

        // Input/Output layers/workloads will not be executed, so the descriptor is not added to workingMemDescriptors
        // However we will still need to manage the tensorHandle
        if (!isInputLayer)
        {
            workingMemDescriptors.push_back(workingMemDescriptor);
            layerIndex++;
        }
    }

    std::vector<WorkingMemHandle::InputMemDescriptorCoords> inputConnectionsInfo;
    std::vector<WorkingMemHandle::OutputMemDescriptorCoords> outputConnectionsInfo;

    for (const auto& handleInfo: handleReferenceCounts)
    {
        if (handleInfo.second.isOutputLayerHandle)
        {
            outputConnectionsInfo.emplace_back(handleInfo.second.m_OutputMemDescriptorCoords);
        }

        if (handleInfo.second.isInputLayerHandle)
        {
            inputConnectionsInfo.emplace_back(handleInfo.second.m_InputMemDescriptorCoords);
        }
    }

    return std::make_unique<WorkingMemHandle>(networkId,
                                              inputConnectionsInfo,
                                              outputConnectionsInfo,
                                              workingMemDescriptors,
                                              workingMemDescriptorMap,
                                              memoryManagers,
                                              std::move(tensorHandleMap));
}

void LoadedNetwork::RegisterDebugCallback(const DebugCallbackFunction& func)
{
    for (auto&& workloadPtr: m_WorkloadQueue)
    {
        workloadPtr.get()->RegisterDebugCallback(func);
    }
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
