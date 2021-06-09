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
#include <backendsCommon/TensorHandleFactoryRegistry.hpp>
#include <backendsCommon/Workload.hpp>
#include <backendsCommon/WorkloadFactory.hpp>
#include <ProfilingService.hpp>
#include <TimelineUtilityMethods.hpp>

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

    /// Single thread execution of the loaded network
    Status EnqueueWorkload(const InputTensors& inputTensors, const OutputTensors& outputTensors);

    /// Thread safe execution of the loaded network
    Status Execute(const InputTensors& inputTensors,
                   const OutputTensors& outputTensors,
                   IWorkingMemHandle& workingMemHandle);

    static std::unique_ptr<LoadedNetwork> MakeLoadedNetwork(std::unique_ptr<IOptimizedNetwork> net,
                                                            std::string& errorMessage,
                                                            const INetworkProperties& networkProperties,
                                                            profiling::ProfilingService& profilingService);

    // NOTE we return by reference as the purpose of this method is only to provide
    // access to the private m_Profiler and in theory we should not need to increment
    // the shared_ptr's reference counter
    const std::shared_ptr<IProfiler>& GetProfiler() const { return m_Profiler; }

    void FreeWorkingMemory();

    void RegisterDebugCallback(const DebugCallbackFunction& func);

    void SendNetworkStructure();

    bool IsAsyncEnabled()
    {
        return m_NetworkProperties.m_AsyncEnabled;
    }

    profiling::ProfilingGuid GetNetworkGuid();

private:
    using WorkloadFactoryWithMemoryManager =
    std::pair<IBackendInternal::IWorkloadFactoryPtr, IBackendInternal::IMemoryManagerSharedPtr>;

    using WorkloadFactoryMap = std::unordered_map<BackendId, WorkloadFactoryWithMemoryManager>;

    void AllocateWorkingMemory(std::lock_guard<std::mutex>& lock);
    void AllocateAndExecuteConstantWorkloads();

    std::unordered_map<LayerGuid, ITensorHandle* > m_ConstantTensorHandles;
    std::unordered_map<LayerGuid, std::unique_ptr<IWorkload> > m_ConstantWorkloads;

    LoadedNetwork(std::unique_ptr<IOptimizedNetwork> net,
                  const INetworkProperties& networkProperties,
                  profiling::ProfilingService& profilingService);

    void EnqueueInput(const BindableLayer& layer, ITensorHandle* tensorHandle, const TensorInfo& tensorInfo);

    void EnqueueOutput(const BindableLayer& layer, ITensorHandle* tensorHandle, const TensorInfo& tensorInfo);

    void EnqueueInput(const BindableLayer& layer, const ConstTensor& inputTensor, WorkingMemHandle& handle);

    void EnqueueOutput(const BindableLayer& layer, const Tensor& outputTensor, WorkingMemHandle& handle);

    bool Execute(std::unique_ptr<profiling::TimelineUtilityMethods>& timelineUtils,
                 profiling::ProfilingGuid inferenceGuid);

    const IWorkloadFactory& GetWorkloadFactory(const Layer& layer) const;

    using BackendPtrMap = std::unordered_map<BackendId, IBackendInternalUniquePtr>;

    BackendPtrMap       m_Backends;
    WorkloadFactoryMap  m_WorkloadFactories;

    std::unique_ptr<IOptimizedNetwork> m_OptimizedNetwork;
    std::shared_ptr<IProfiler>         m_Profiler;

    WorkloadQueue                      m_InputQueue;
    WorkloadQueue                      m_WorkloadQueue;
    WorkloadQueue                      m_OutputQueue;

    mutable std::mutex m_WorkingMemMutex;

    bool m_IsWorkingMemAllocated = false;

    INetworkProperties m_NetworkProperties;

    TensorHandleFactoryRegistry m_TensorHandleFactoryRegistry;

    profiling::ProfilingService& m_ProfilingService;
};

}
