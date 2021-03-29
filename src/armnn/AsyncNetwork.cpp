//
// Copyright © 2021 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "AsyncNetwork.hpp"
#include "Graph.hpp"
#include "Layer.hpp"
#include "Profiling.hpp"

#include <armnn/BackendRegistry.hpp>
#include <armnn/Logging.hpp>
#include <armnn/utility/Assert.hpp>

#include <armnn/backends/IMemoryManager.hpp>
#include <backendsCommon/CpuTensorHandle.hpp>
#include <backendsCommon/WorkloadData.hpp>
#include <backendsCommon/MemCopyWorkload.hpp>
#include <LabelsAndEventClasses.hpp>

#include <fmt/format.h>

namespace armnn
{

namespace experimental
{

void AddLayerStructure(std::unique_ptr<profiling::TimelineUtilityMethods>& timelineUtils,
                       const Layer& layer,
                       profiling::ProfilingGuid networkGuid)
{
    // Add layer to the post-optimisation network structure
    std::string layerName = layer.GetNameStr().empty() ? "<Unnamed>" : layer.GetNameStr();
    timelineUtils->CreateNamedTypedChildEntity(layer.GetGuid(),
                                               networkGuid,
                                               layerName,
                                               profiling::LabelsAndEventClasses::LAYER_GUID);
    for (auto&& input : layer.GetInputSlots())
    {
        const IOutputSlot* source = input.GetConnectedOutputSlot();
        ARMNN_ASSERT(source != NULL);
        timelineUtils->CreateConnectionRelationship(profiling::ProfilingRelationshipType::RetentionLink,
                                                    source->GetOwningLayerGuid(),
                                                    layer.GetGuid());
    }
}

void AddWorkloadStructure(std::unique_ptr<profiling::TimelineUtilityMethods>& timelineUtils,
                          std::unique_ptr<IWorkload>& workload,
                          const Layer& layer)
{
    // Add workload to the post-optimisation network structure
    timelineUtils->CreateTypedEntity(workload->GetGuid(), profiling::LabelsAndEventClasses::WORKLOAD_GUID);
    timelineUtils->MarkEntityWithLabel(workload->GetGuid(),
                                       layer.GetBackendId().Get(),
                                       profiling::LabelsAndEventClasses::BACKENDID_GUID);

    // Link the workload to the layer
    timelineUtils->CreateRelationship(profiling::ProfilingRelationshipType::RetentionLink,
                                      layer.GetGuid(),
                                      workload->GetGuid(),
                                      profiling::LabelsAndEventClasses::CHILD_GUID);
}

TensorInfo AsyncNetwork::GetInputTensorInfo(LayerBindingId layerId) const
{
    for (auto&& inputLayer : m_OptimizedNetwork->pOptimizedNetworkImpl->GetGraph().GetInputLayers())
    {
        ARMNN_ASSERT_MSG(inputLayer->GetNumOutputSlots() == 1, "Input layer should have exactly 1 output slot");
        if (inputLayer->GetBindingId() == layerId)
        {
            return inputLayer->GetOutputSlot(0).GetTensorInfo();
        }
    }

    throw InvalidArgumentException(fmt::format("No input layer is associated with id {0}}", layerId));
}

TensorInfo AsyncNetwork::GetOutputTensorInfo(LayerBindingId layerId) const
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

    throw InvalidArgumentException(fmt::format("No output layer is associated with id {0}}", layerId));
}

// Need something like the collectors to get the correct tensors for the inputs
void AsyncNetwork::CollectInputTensorHandles(
        std::unordered_map<LayerGuid, std::vector<ITensorHandle*> >& tensorHandles,
        std::vector<ITensorHandle*>& inputs,
        const armnn::Layer* layer,
        const TensorHandleFactoryRegistry& registry,
        const bool isMemoryManaged)
{
    for (auto&& inputSlot : layer->GetInputSlots())
    {
        // The graph must be well-formed at this point.
        ARMNN_ASSERT(inputSlot.GetConnection());
        auto outputSlot = inputSlot.GetConnectedOutputSlot();
        auto key = outputSlot->GetOwningLayer().GetGuid();
        auto search = tensorHandles.find(key);

        if (search == tensorHandles.end())
        {
            ITensorHandleFactory::FactoryId factoryId = outputSlot->GetTensorHandleFactoryId();
            const TensorInfo& tensorInfo = outputSlot->GetTensorInfo();

            ARMNN_ASSERT(factoryId != ITensorHandleFactory::LegacyFactoryId);
            ITensorHandleFactory* handleFactory = registry.GetFactory(factoryId);
            ARMNN_ASSERT(handleFactory);
            std::unique_ptr<ITensorHandle> tensor = handleFactory->CreateTensorHandle(tensorInfo, isMemoryManaged);
            ITensorHandle* tensorPtr = tensor.release();
            inputs.push_back(tensorPtr);
        }
        else
        {
            unsigned int index = outputSlot->CalculateIndexOnOwner();
            inputs.push_back(search->second[index]);
        }
    }
}

void AsyncNetwork::CreateOutputTensorHandles(
        std::unordered_map<LayerGuid, std::vector<ITensorHandle*> >& tensorHandles,
        std::vector<ITensorHandle*>& outputs,
        const armnn::Layer* layer,
        const TensorHandleFactoryRegistry& registry,
        const bool isMemoryManaged)
{
    auto guid = layer->GetGuid();
    std::vector<ITensorHandle*> tensorHandleVectors;
    tensorHandleVectors.reserve(layer->GetNumOutputSlots());

    for (unsigned int idx=0; idx < layer->GetNumOutputSlots(); idx++)
    {
        const OutputSlot& slot = layer->GetOutputSlot(idx);
        ITensorHandleFactory::FactoryId factoryId = slot.GetTensorHandleFactoryId();
        const TensorInfo& tensorInfo = slot.GetTensorInfo();

        ARMNN_ASSERT(factoryId != ITensorHandleFactory::LegacyFactoryId);
        ITensorHandleFactory* handleFactory = registry.GetFactory(factoryId);
        ARMNN_ASSERT(handleFactory);
        std::unique_ptr<ITensorHandle> tensor = handleFactory->CreateTensorHandle(tensorInfo, isMemoryManaged);
        ITensorHandle* tensorPtr = tensor.release();
        outputs.push_back(tensorPtr);
        tensorHandleVectors.push_back(tensorPtr);
    }
    tensorHandles.insert({guid, tensorHandleVectors});
}

const IWorkloadFactory& AsyncNetwork::GetWorkloadFactory(const Layer& layer) const
{
    const IWorkloadFactory* workloadFactory = nullptr;

    auto it = m_WorkloadFactories.find(layer.GetBackendId());
    if (it == m_WorkloadFactories.end())
    {
        throw RuntimeException(
                        fmt::format("No workload factory for {0} to be used for layer: {1}}",
                                    layer.GetBackendId().Get(),
                                    layer.GetNameStr()),
                                    CHECK_LOCATION());
    }

    workloadFactory = it->second.first.get();

    ARMNN_ASSERT_MSG(workloadFactory, "No workload factory");

    std::string reasonIfUnsupported;
    ARMNN_ASSERT_MSG(IWorkloadFactory::IsLayerSupported(layer, {}, reasonIfUnsupported),
                     "Factory does not support layer");
    IgnoreUnused(reasonIfUnsupported);
    return *workloadFactory;
}

void AsyncNetwork::EnqueueInput(const BindableLayer& layer, const ConstTensor& inputTensor, WorkingMemHandle& context)
{
    if (layer.GetType() != LayerType::Input)
    {
        throw InvalidArgumentException("EnqueueInput: given layer not an InputLayer");
    }
    LayerGuid id = layer.GetOutputSlot(0).GetConnection(0)->GetOwningLayer().GetGuid();
    WorkingMemDescriptor descriptor = context.GetWorkingMemDescriptor(id);
    ARMNN_ASSERT_MSG(descriptor.m_Outputs.size() == 1, "Can only handle Input Layer with one output");

    MemorySourceFlags importFlags = descriptor.m_Outputs[0]->GetImportFlags();
    if (m_NetworkProperties.m_ImportEnabled)  // Try import the input tensor
    {
        if (CheckFlag(importFlags, MemorySource::Malloc) )
        {
            // This assumes a CPU Tensor handle
            std::unique_ptr<ITensorHandle> tensorHandle =
                    std::make_unique<ConstPassthroughCpuTensorHandle>(inputTensor.GetInfo(),
                                                                      inputTensor.GetMemoryArea());

            void* mem = tensorHandle->Map(false);
            if (descriptor.m_Outputs[0]->Import(mem, MemorySource::Malloc))
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
                std::make_unique<ConstPassthroughCpuTensorHandle>(inputTensor.GetInfo(), inputTensor.GetMemoryArea());

        auto copyFunc = [](void* dst, const void* src, size_t size)
        {
            memcpy(dst, src, size);
        };

        for (const auto& input : descriptor.m_Inputs)
        {
            CopyTensorContentsGeneric(tensorHandle.get(), input, copyFunc);
        }
    }
}

void AsyncNetwork::EnqueueOutput(const BindableLayer& layer, const Tensor& outputTensor, WorkingMemHandle& handle)
{
    if (layer.GetType() != LayerType::Output)
    {
        throw InvalidArgumentException("EnqueueOutput: given layer not an OutputLayer");
    }
    ARMNN_ASSERT_MSG(layer.GetNumInputSlots() == 1, "Output Layer should have exactly one input.");

    LayerGuid id = layer.GetInputSlot(0).GetConnectedOutputSlot()->GetOwningLayerGuid();
    WorkingMemDescriptor descriptor = handle.GetWorkingMemDescriptor(id);

    ITensorHandle* inputTensorHandle = descriptor.m_Inputs[0];
    ARMNN_ASSERT_MSG(inputTensorHandle != nullptr, "Data should have been allocated.");

    // Try import the output tensor.
    // Note: We can only import the output pointer if all of the following  hold true:
    // a) The imported pointer is aligned sufficiently
    // b) The tensor has zero padding
    // c) There is only one connection to the OutputSlot and it is to an OutputLayer.
    // d) The output pointer is allocated via malloc. (Other types will be supported in a later release)
    // e) m_IsExportEnabled must be set to true
    if (m_NetworkProperties.m_ExportEnabled &&
        (layer.GetInputSlots()[0].GetConnectedOutputSlot()->GetNumConnections() == 1))
    {
        if (layer.GetInputSlots()[0].GetConnectedOutputSlot()->GetOwningLayer().GetType() != LayerType::Input)
        {
            MemorySourceFlags importFlags = inputTensorHandle->GetImportFlags();
            if (CheckFlag(importFlags, MemorySource::Malloc))
            {
                std::unique_ptr<ITensorHandle> tensorHandle =
                        std::make_unique<PassthroughCpuTensorHandle>(outputTensor.GetInfo(),
                                                                     outputTensor.GetMemoryArea());

                void* mem = tensorHandle->Map(false);
                bool importOk = inputTensorHandle->Import(mem, MemorySource::Malloc);
                tensorHandle->Unmap();

                if (importOk)
                {
                    ARMNN_SCOPED_PROFILING_EVENT(Compute::Undefined, "SyncMemGeneric_Execute");
                    descriptor.m_Inputs[0]->Map(true);
                    descriptor.m_Inputs[0]->Unmap();
                }
                else
                {
                    throw MemoryExportException("EnqueueOutput: Memory Export failed");
                }
            }
            else
            {
                throw MemoryExportException("EnqueueOutput: Memory Export failed, backend does not support Export");
            }
        }
        else
        {
            throw MemoryExportException("EnqueueOutput: Memory Export failed, attempting to export Input Layer");
        }
    }
    else
    {
        auto copyFunc = [](void* dst, const void* src, size_t size)
        {
            memcpy(dst, src, size);
        };

        std::unique_ptr<ITensorHandle> tensorHandle =
                std::make_unique<PassthroughCpuTensorHandle>(outputTensor.GetInfo(), outputTensor.GetMemoryArea());

        CopyTensorContentsGeneric(descriptor.m_Outputs[0], tensorHandle.get(), copyFunc);
    }
}

AsyncNetwork::AsyncNetwork(std::unique_ptr<IOptimizedNetwork> net,
                           const INetworkProperties& networkProperties,
                           profiling::ProfilingService& profilingService) :
    m_OptimizedNetwork(std::move(net)),
    m_NetworkProperties(networkProperties),
    m_ProfilingService(profilingService)
{
    // Create a profiler and register it for the current thread.
    m_Profiler = std::make_shared<IProfiler>();
    ProfilerManager::GetInstance().RegisterProfiler(m_Profiler.get());

    Graph &order = m_OptimizedNetwork->pOptimizedNetworkImpl->GetGraph().TopologicalSort();

    //First create tensor handlers, backends and workload factories.
    //Handlers are created before workloads are.
    //Because workload creation can modify some of the handlers,
    //(for example the splitter and concat layers).
    for (auto &&layer : order)
    {
        auto const &backendId = layer->GetBackendId();
        if (m_Backends.count(backendId) == 0)
        {
            auto createBackend = BackendRegistryInstance().GetFactory(backendId);
            auto it = m_Backends.emplace(std::make_pair(backendId, createBackend()));

            IBackendInternal* backend = it.first->second.get();

            if (backend->SupportsTensorAllocatorAPI())
            {
                backend->RegisterTensorHandleFactories(m_TensorHandleFactoryRegistry);

                auto workloadFactory = backend->CreateWorkloadFactory(m_TensorHandleFactoryRegistry);
                m_WorkloadFactories.emplace(
                        std::make_pair(backendId, std::make_pair(std::move(workloadFactory), nullptr)));
            }
            else
            {
                IBackendInternal::IMemoryManagerSharedPtr memoryManager = backend->CreateMemoryManager();
                auto workloadFactory = backend->CreateWorkloadFactory(memoryManager);

                m_WorkloadFactories.emplace(
                        std::make_pair(backendId, std::make_pair(std::move(workloadFactory), memoryManager)));
            }
        }
    }

    profiling::ProfilingGuid networkGuid = m_OptimizedNetwork->GetGuid();
    std::unique_ptr<profiling::TimelineUtilityMethods> timelineUtils =
            profiling::TimelineUtilityMethods::GetTimelineUtils(m_ProfilingService);
    if (timelineUtils)
    {
        timelineUtils->CreateTypedEntity(networkGuid, profiling::LabelsAndEventClasses::NETWORK_GUID);
    }

    //Then create workloads.
    for (auto &&layer : order)
    {
        if (timelineUtils)
        {
            // Add layer to the post-optimisation network structure
            AddLayerStructure(timelineUtils, *layer, networkGuid);
        }

        const IWorkloadFactory &workloadFactory = GetWorkloadFactory(*layer);

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
                                        layerName,
                                        static_cast<int>(layer->GetType()),
                                        layer->GetBackendId().Get()
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

    if (timelineUtils)
    {
        // Commit to send the post-optimisation network structure
        timelineUtils->Commit();
    }

    // Now that the intermediate tensor memory has been set-up, do any post allocation configuration for each workload.
    // PostAllocationConfiguure will now need to be handled in the ExecuteOn(WorkingMemDescriptor)
    for (auto &workload : m_WorkloadQueue)
    {
        workload->PostAllocationConfigure();
    }
}

Status AsyncNetwork::Execute(const InputTensors& inputTensors,
                             const OutputTensors& outputTensors,
                             IWorkingMemHandle& iWorkingMemHandle)
{
    const Graph& graph = m_OptimizedNetwork->pOptimizedNetworkImpl->GetGraph();

    // Walk graph to determine the order of execution.
    if (graph.GetNumLayers() < 2)
    {
        ARMNN_LOG(warning) << "IRuntime::EnqueueWorkload()::Less than two nodes in graph";
        return Status::Failure;
    }

    if (graph.GetNumInputs() != inputTensors.size())
    {
        throw InvalidArgumentException("Number of inputs provided does not match network.");
    }

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
    WorkingMemHandle& workingMemHandle = dynamic_cast<WorkingMemHandle&>(iWorkingMemHandle);
    std::lock_guard<std::mutex> lockGuard(workingMemHandle.GetMutex());

    if (!workingMemHandle.IsAllocated())
    {
        workingMemHandle.Allocate();
    }

    {
        ARMNN_SCOPED_PROFILING_EVENT(Compute::Undefined, "PrepareInputs");
        unsigned int i = 0;

        for (const BindableLayer* inputLayer : graph.GetInputLayers())
        {
            EnqueueInput(*inputLayer, inputTensors[i].second, workingMemHandle);
            ++i;
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
        Fail(error);
    }
    catch (const std::runtime_error& error)
    {
        Fail(error);
    }
    // For each output to the network, call EnqueueOutput with the data passed by the user.
    {
        ARMNN_SCOPED_PROFILING_EVENT(Compute::Undefined, "PrepareOutputs");
        unsigned int i = static_cast<unsigned int>(m_WorkloadQueue.size() - graph.GetNumOutputs());

        for (const BindableLayer* outputLayer : graph.GetOutputLayers())
        {
            EnqueueOutput(*outputLayer, outputTensors[i].second, workingMemHandle);
            ++i;
        }
    }
    return executionSucceeded ? Status::Success : Status::Failure;
}

/// Get the profiler used for this network
std::shared_ptr<IProfiler> AsyncNetwork::GetProfiler() const
{
    return m_Profiler;
}

void AsyncNetwork::RegisterDebugCallback(const DebugCallbackFunction& func)
{
    for (auto&& workloadPtr: m_WorkloadQueue)
    {
        workloadPtr.get()->RegisterDebugCallback(func);
    }
}

/// Create a new unique WorkingMemHandle object. Create multiple handles if you wish to have
/// overlapped Execution by calling this function from different threads.
std::unique_ptr<IWorkingMemHandle> AsyncNetwork::CreateWorkingMemHandle()
{
    Graph& order = m_OptimizedNetwork->pOptimizedNetworkImpl->GetGraph();
    std::unordered_map<LayerGuid, std::vector<ITensorHandle*> > tensorHandles;
    std::vector<WorkingMemDescriptor> workingMemDescriptors;
    std::unordered_map<LayerGuid, WorkingMemDescriptor> workingMemDescriptorMap;

    for (auto&& layer : order)
    {
        if (layer->GetType() == LayerType::Input || layer->GetType() == LayerType::Output)
        {
            continue;
        }
        WorkingMemDescriptor workingMemDescriptor;
        // Look for the layer with 1 OutputSlot which has 1 connection and that connection is an Output Layer
        // If Export is enabled disable memory management so we can export, otherwise we do a copy
        if((layer->GetNumOutputSlots() == 1) &&
           (layer->GetOutputSlots()[0].GetNumConnections() == 1) &&
           (layer->GetOutputSlots()[0].GetConnection(0)->GetOwningLayer().GetType() == LayerType::Output))
        {
            CollectInputTensorHandles(tensorHandles,
                                      workingMemDescriptor.m_Inputs,
                                      layer,
                                      m_TensorHandleFactoryRegistry,
                                      !m_NetworkProperties.m_ExportEnabled);
            CreateOutputTensorHandles(tensorHandles,
                                      workingMemDescriptor.m_Outputs,
                                      layer,
                                      m_TensorHandleFactoryRegistry,
                                      !m_NetworkProperties.m_ExportEnabled);
        }
        else
        {
            CollectInputTensorHandles(tensorHandles,
                                      workingMemDescriptor.m_Inputs,
                                      layer,
                                      m_TensorHandleFactoryRegistry);
            CreateOutputTensorHandles(tensorHandles,
                                      workingMemDescriptor.m_Outputs,
                                      layer,
                                      m_TensorHandleFactoryRegistry);
        }
        workingMemDescriptorMap.insert({layer->GetGuid(), workingMemDescriptor});
        workingMemDescriptors.push_back(workingMemDescriptor);
    }
    return std::make_unique<WorkingMemHandle>(workingMemDescriptors, workingMemDescriptorMap);
}

void AsyncNetwork::FreeWorkingMemory()
{
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
}

} // end experimental namespace

} // end armnn namespace
