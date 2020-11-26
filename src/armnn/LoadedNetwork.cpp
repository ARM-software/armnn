//
// Copyright © 2017 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "LoadedNetwork.hpp"
#include "Layer.hpp"
#include "Graph.hpp"
#include "Network.hpp"
#include <Processes.hpp>
#include "Profiling.hpp"
#include "HeapProfiling.hpp"

#include <armnn/BackendRegistry.hpp>
#include <armnn/Logging.hpp>
#include <armnn/utility/Assert.hpp>

#include <backendsCommon/CpuTensorHandle.hpp>
#include <armnn/backends/IMemoryManager.hpp>
#include <backendsCommon/MemCopyWorkload.hpp>
#include <backendsCommon/MemSyncWorkload.hpp>

#include <LabelsAndEventClasses.hpp>

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

std::unique_ptr<LoadedNetwork> LoadedNetwork::MakeLoadedNetwork(std::unique_ptr<OptimizedNetwork> net,
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

LoadedNetwork::LoadedNetwork(std::unique_ptr<OptimizedNetwork> net,
                             const INetworkProperties& networkProperties,
                             profiling::ProfilingService&  profilingService) :
                             m_OptimizedNetwork(std::move(net)),
                             m_IsImportEnabled(networkProperties.m_ImportEnabled),
                             m_IsExportEnabled(networkProperties.m_ExportEnabled),
                             m_TensorHandleFactoryRegistry(),
                             m_ProfilingService(profilingService)
{
    // Create a profiler and register it for the current thread.
    m_Profiler = std::make_shared<Profiler>();
    ProfilerManager::GetInstance().RegisterProfiler(m_Profiler.get());

    Graph& order = m_OptimizedNetwork->GetGraph().TopologicalSort();
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

            if (backend->SupportsTensorAllocatorAPI())
            {
                auto workloadFactory = backend->CreateWorkloadFactory(
                    m_TensorHandleFactoryRegistry, m_OptimizedNetwork->GetModelOptions());
                m_WorkloadFactories.emplace(
                    std::make_pair(backendId, std::make_pair(std::move(workloadFactory), nullptr)));
            }
            else
            {
                IBackendInternal::IMemoryManagerSharedPtr memoryManager = backend->CreateMemoryManager();
                auto workloadFactory = backend->CreateWorkloadFactory(
                    memoryManager, m_OptimizedNetwork->GetModelOptions());

                m_WorkloadFactories.emplace(
                    std::make_pair(backendId, std::make_pair(std::move(workloadFactory), memoryManager)));
            }
        }
    }

    for (auto&& layer : order)
    {
        auto& workloadFactory = GetWorkloadFactory(*layer);

        switch (layer->GetType())
        {
        case LayerType::Input:
        case LayerType::MemImport:
            {
                // If IsImportEnabled is true then we need to set IsMemoryManaged to false when creating TensorHandles
                layer->CreateTensorHandles(m_TensorHandleFactoryRegistry, workloadFactory, !m_IsImportEnabled);
                break;
            }
        default:
            {
                // Look for the layer with 1 OutputSlot which has 1 connection and that connection is an Output Layer
                // If Export is enabled disable memory management so we can export, otherwise we do a copy
                if((layer->GetNumOutputSlots() == 1) &&
                   (layer->GetOutputSlots()[0].GetNumConnections() == 1) &&
                   (layer->GetOutputSlots()[0].GetConnection(0)->GetOwningLayer().GetType() == LayerType::Output))
                {
                    layer->CreateTensorHandles(m_TensorHandleFactoryRegistry, workloadFactory, !m_IsExportEnabled);
                }
                else
                {
                    layer->CreateTensorHandles(m_TensorHandleFactoryRegistry, workloadFactory);
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

                m_WorkloadQueue.push_back(move(workload));
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

    // Set up memory.
    m_OptimizedNetwork->GetGraph().AllocateDynamicBuffers();

    // Now that the intermediate tensor memory has been set-up, do any post allocation configuration for each workload.
    for (auto& workload : m_WorkloadQueue)
    {
        workload->PostAllocationConfigure();
    }
}

void LoadedNetwork::SendNetworkStructure()
{
    Graph& order = m_OptimizedNetwork->GetGraph().TopologicalSort();
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
    for (auto&& inputLayer : m_OptimizedNetwork->GetGraph().GetInputLayers())
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
    for (auto&& outputLayer : m_OptimizedNetwork->GetGraph().GetOutputLayers())
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
                                                        m_OptimizedNetwork->GetModelOptions()),
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
                std::make_unique<ConstPassthroughCpuTensorHandle>(inputTensor.GetInfo(),inputTensor.GetMemoryArea());
            LayerBindingId layerId = inputTensorPair.first;

            m_InputTensorPins.emplace_back(std::move(tensorHandle), inputTensor.GetInfo(), layerId);
        }

        for (auto outputTensorPair : outputTensors)
        {
            auto outputTensor = outputTensorPair.second;

            std::unique_ptr<ITensorHandle> tensorHandle =
                std::make_unique<PassthroughCpuTensorHandle>(outputTensor.GetInfo(), outputTensor.GetMemoryArea());
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
    const Graph& graph = m_OptimizedNetwork->GetGraph();

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
    if (m_IsImportEnabled)  // Try import the input tensor
    {
        if(CheckFlag(importFlags, MemorySource::Malloc) )
        {
            needMemCopy = false;
            // This assumes a CPU Tensor handle
            void* mem = tensorHandle->Map(false);
            if (outputTensorHandle->Import(mem, MemorySource::Malloc))
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
    if (m_IsExportEnabled && (layer.GetInputSlots()[0].GetConnectedOutputSlot()->GetNumConnections() == 1))
    {
        if(layer.GetInputSlots()[0].GetConnectedOutputSlot()->GetOwningLayer().GetType() != LayerType::Input)
        {
            MemorySourceFlags importFlags = inputTensorHandle->GetImportFlags();
            if (CheckFlag(importFlags, MemorySource::Malloc))
            {
                needMemCopy = false;
                void *mem = tensorHandle->Map(false);
                bool importOk = inputTensorHandle->Import(mem, MemorySource::Malloc);
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

void LoadedNetwork::RegisterDebugCallback(const DebugCallbackFunction& func)
{
    for (auto&& workloadPtr: m_WorkloadQueue)
    {
        workloadPtr.get()->RegisterDebugCallback(func);
    }
}

}
