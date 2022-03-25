//
// Copyright © 2017 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//
#pragma once

#include "Network.hpp"
#include "LayerFwd.hpp"
#include "Profiling.hpp"

#include <armnn/Tensor.hpp>

#include <armnn/backends/IBackendInternal.hpp>
#include <armnn/backends/IMemoryOptimizerStrategy.hpp>
#include <armnn/backends/Workload.hpp>
#include <armnn/backends/WorkloadFactory.hpp>

#include <backendsCommon/DefaultAllocator.hpp>
#include <backendsCommon/MemoryManager.hpp>
#include <backendsCommon/TensorHandleFactoryRegistry.hpp>
#include <backendsCommon/memoryOptimizerStrategyLibrary/strategies/SingleAxisPriorityList.hpp>

#include <client/include/IProfilingService.hpp>
#include <client/include/TimelineUtilityMethods.hpp>

#include <common/include/LabelsAndEventClasses.hpp>

#include <mutex>
#include <condition_variable>
#include <unordered_map>

namespace cl
{
class Context;
class CommandQueue;
class Device;
}

namespace armnn
{

class LoadedNetwork
{
public:
    using WorkloadQueue = std::vector<std::unique_ptr<IWorkload>>;

    ~LoadedNetwork()
    {
        FreeWorkingMemory();
    }

    /// Create a new unique WorkingMemHandle object. Create multiple handles if you wish to have
    /// overlapped Execution by calling this function from different threads.
    std::unique_ptr<IWorkingMemHandle> CreateWorkingMemHandle(NetworkId networkId);

    TensorInfo GetInputTensorInfo(LayerBindingId layerId) const;
    TensorInfo GetOutputTensorInfo(LayerBindingId layerId) const;

    std::vector<ImportedInputId> ImportInputs(const InputTensors& inputTensors,
                                              MemorySource forceImportMemorySource = MemorySource::Undefined);
    std::vector<ImportedOutputId> ImportOutputs(const OutputTensors& outputTensors,
                                                MemorySource forceImportMemorySource = MemorySource::Undefined);

    void ClearImportedInputs(const std::vector<ImportedInputId> inputIds);
    void ClearImportedOutputs(const std::vector<ImportedOutputId> outputIds);

    /// Single thread execution of the loaded network
    Status EnqueueWorkload(const InputTensors& inputTensors, const OutputTensors& outputTensors,
                           std::vector<ImportedInputId> preImportedInputIds = {},
                           std::vector<ImportedOutputId> preImportedOutputIds = {});

    /// Thread safe execution of the loaded network
    Status Execute(const InputTensors& inputTensors,
                   const OutputTensors& outputTensors,
                   IWorkingMemHandle& workingMemHandle,
                   std::vector<ImportedInputId> preImportedInputs = {},
                   std::vector<ImportedOutputId> preImportedOutputs = {});

    static std::unique_ptr<LoadedNetwork> MakeLoadedNetwork(std::unique_ptr<IOptimizedNetwork> net,
                                                            std::string& errorMessage,
                                                            const INetworkProperties& networkProperties,
                                                            arm::pipe::IProfilingService* profilingService);

    // NOTE we return by reference as the purpose of this method is only to provide
    // access to the private m_Profiler and in theory we should not need to increment
    // the shared_ptr's reference counter
    const std::shared_ptr<IProfiler>& GetProfiler() const { return m_OptimizedNetwork->GetProfiler(); }

    void FreeWorkingMemory();

    void RegisterDebugCallback(const DebugCallbackFunction& func);

    void SendNetworkStructure(arm::pipe::IProfilingService& profilingService);

    bool IsAsyncEnabled()
    {
        return m_NetworkProperties.m_AsyncEnabled;
    }

    arm::pipe::ProfilingGuid GetNetworkGuid();

private:


    void AllocateWorkingMemory(
#if !defined(ARMNN_DISABLE_THREADS)
        std::lock_guard<std::mutex>& lock
#endif
    );
    void AllocateAndExecuteConstantWorkloads();
    void AllocateAndExecuteConstantWorkloadsAsync();

    std::unordered_map<LayerGuid, std::unique_ptr<IWorkload>> m_ConstantWorkloads;
    std::unordered_map<LayerGuid, ITensorHandle*> m_ConstantTensorHandles;

    std::unique_ptr<IMemoryOptimizerStrategy> m_ConstantStrategy = std::make_unique<SingleAxisPriorityList>();

    LoadedNetwork(std::unique_ptr<IOptimizedNetwork> net,
                  const INetworkProperties& networkProperties,
                  arm::pipe::IProfilingService* profilingService);

    void EnqueueInput(const BindableLayer& layer, ITensorHandle* tensorHandle, const TensorInfo& tensorInfo);

    void EnqueueOutput(const BindableLayer& layer, ITensorHandle* tensorHandle, const TensorInfo& tensorInfo);

    void EnqueueInput(const ConstTensor& inputTensor, ITensorHandle* inputTensorHandle);

    void ImportOutputTensor(const Tensor& outputTensor, ITensorHandle* outputTensorHandle);

    bool Execute(std::unique_ptr<arm::pipe::TimelineUtilityMethods>& timelineUtils,
                 arm::pipe::ProfilingGuid inferenceGuid);

    const IWorkloadFactory& GetWorkloadFactory(const Layer& layer) const;

    inline LayerBindingId ValidateImportedInputID(ImportedInputId id);
    inline LayerBindingId ValidateImportedOutputID(ImportedOutputId id);

    void CreateMemoryProfile();
    void CreateMemoryProfileAsync();

    std::unique_ptr<MemoryManager> CreateExternalMemoryManger(
            std::vector<std::pair<std::shared_ptr<TensorMemory>, MemorySource>>& tensorMemory);

    using BackendPtrMap = std::unordered_map<BackendId, IBackendInternalUniquePtr>;

    BackendPtrMap  m_Backends;
    std::vector<IBackendInternal::IMemoryManagerSharedPtr> m_BackendMemoryMangers;

    using WorkloadFactoryMap = std::unordered_map<BackendId, IBackendInternal::IWorkloadFactoryPtr>;
    WorkloadFactoryMap  m_WorkloadFactories;

    std::unique_ptr<IOptimizedNetwork> m_OptimizedNetwork;

    WorkloadQueue                      m_InputQueue;
    WorkloadQueue                      m_WorkloadQueue;
    WorkloadQueue                      m_OutputQueue;

#if !defined(ARMNN_DISABLE_THREADS)
    mutable std::mutex m_WorkingMemMutex;
#endif

    bool m_IsWorkingMemAllocated = false;

    INetworkProperties m_NetworkProperties;

    TensorHandleFactoryRegistry m_TensorHandleFactoryRegistry;

    // NOTE: raw pointer because the profiling service is controlled by the Runtime
    arm::pipe::IProfilingService* m_ProfilingService;

    struct ImportedTensorHandlePin
    {
        ImportedTensorHandlePin()
        {}

        ImportedTensorHandlePin(LayerBindingId layerBindingId,
                                std::unique_ptr<ITensorHandle> tensorHandle)
        : m_LayerBindingId(layerBindingId)
        , m_TensorHandle(std::move(tensorHandle))
        {}

        ImportedTensorHandlePin(ImportedTensorHandlePin&&) = default;

        ~ImportedTensorHandlePin()
        {
            if (m_TensorHandle)
            {
                m_TensorHandle->Unimport();
            }
        }

        LayerBindingId m_LayerBindingId;
        std::unique_ptr<ITensorHandle> m_TensorHandle;
    };

    std::vector<ImportedTensorHandlePin> m_PreImportedInputHandles;
    std::vector<ImportedTensorHandlePin> m_PreImportedOutputHandles;

    ImportedInputId m_CurImportedInputId = 0;
    ImportedInputId m_CurImportedOutputId = 0;

    std::unordered_map<BackendId, std::vector<MemBlock>> m_MemBlockMap;
    std::unordered_map<BackendId, std::vector<MemBin>> m_MemBinMap;

    std::vector<ITensorHandle*> m_Tensorhandles;

    std::vector<std::pair<std::shared_ptr<TensorMemory>, MemorySource>> m_TensorMemory;

    std::unique_ptr<MemoryManager> m_ExternalMemoryManager;

    std::unordered_map<BackendId, bool> m_SupportsExternallyManagedMemory;

    // A set of vectors to record the workload queue indexes and their corresponding Input/Output Slot indexes
    // which are connected to Inputs and Outputs for the network.
    struct WorkloadIndices
    {
        unsigned int m_WorkloadIndex;
        unsigned int m_SlotIndex;
    };

    struct OutputWorkloadIndices
    {
        WorkloadIndices m_OutputSlotIndices;
        std::vector<WorkloadIndices> m_InputSlotIndices;
    };
    std::unordered_map<LayerBindingId, std::vector<WorkloadIndices>> m_InputWorkloadSlotPairs;
    std::unordered_map<LayerBindingId, OutputWorkloadIndices> m_OutputWorkloadSlotPairs;
    std::vector<bool> m_IsInputImported;
    std::vector<bool> m_IsOutputImported;

};

}
