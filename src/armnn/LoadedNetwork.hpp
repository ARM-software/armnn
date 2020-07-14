//
// Copyright © 2017 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//
#pragma once

#include <armnn/Tensor.hpp>
#include <armnn/Types.hpp>

#include "Network.hpp"
#include "LayerFwd.hpp"
#include "Profiling.hpp"

#include <armnn/backends/IBackendInternal.hpp>
#include <backendsCommon/TensorHandleFactoryRegistry.hpp>
#include <backendsCommon/Workload.hpp>
#include <backendsCommon/WorkloadFactory.hpp>
#include <ProfilingService.hpp>
#include <TimelineUtilityMethods.hpp>

#include <mutex>
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
    using WorkloadQueue = std::vector< std::unique_ptr<IWorkload> >;
    ~LoadedNetwork(){ FreeWorkingMemory(); }

    TensorInfo GetInputTensorInfo(LayerBindingId layerId) const;
    TensorInfo GetOutputTensorInfo(LayerBindingId layerId) const;

    Status EnqueueWorkload(const InputTensors& inputTensors, const OutputTensors& outputTensors);

    static std::unique_ptr<LoadedNetwork> MakeLoadedNetwork(std::unique_ptr<OptimizedNetwork> net,
                                                            std::string & errorMessage,
                                                            const INetworkProperties& networkProperties,
                                                            profiling::ProfilingService& profilingService);

    // NOTE we return by reference as the purpose of this method is only to provide
    // access to the private m_Profiler and in theory we should not need to increment
    // the shared_ptr's reference counter
    const std::shared_ptr<Profiler>& GetProfiler() const { return m_Profiler; }

    void FreeWorkingMemory();

    void RegisterDebugCallback(const DebugCallbackFunction& func);

    void SendNetworkStructure();

    profiling::ProfilingGuid GetNetworkGuid();

private:
    void AllocateWorkingMemory(std::lock_guard<std::mutex>& lock);

    LoadedNetwork(std::unique_ptr<OptimizedNetwork> net,
                  const INetworkProperties& networkProperties,
                  profiling::ProfilingService& profilingService);

    void EnqueueInput(const BindableLayer& layer, ITensorHandle* tensorHandle, const TensorInfo& tensorInfo);

    void EnqueueOutput(const BindableLayer& layer, ITensorHandle* tensorHandle, const TensorInfo& tensorInfo);

    bool Execute(std::unique_ptr<profiling::TimelineUtilityMethods>& timelineUtils,
                 profiling::ProfilingGuid inferenceGuid);


    const IWorkloadFactory& GetWorkloadFactory(const Layer& layer) const;

    using BackendPtrMap = std::unordered_map<BackendId, IBackendInternalUniquePtr>;

    using WorkloadFactoryWithMemoryManager =
        std::pair<IBackendInternal::IWorkloadFactoryPtr, IBackendInternal::IMemoryManagerSharedPtr>;

    using WorkloadFactoryMap = std::unordered_map<BackendId, WorkloadFactoryWithMemoryManager>;

    BackendPtrMap       m_Backends;
    WorkloadFactoryMap  m_WorkloadFactories;

    std::unique_ptr<OptimizedNetwork> m_OptimizedNetwork;
    WorkloadQueue m_InputQueue;
    WorkloadQueue m_WorkloadQueue;
    WorkloadQueue m_OutputQueue;
    std::shared_ptr<Profiler> m_Profiler;

    mutable std::mutex m_WorkingMemMutex;

    bool m_IsWorkingMemAllocated=false;
    bool m_IsImportEnabled=false;
    bool m_IsExportEnabled=false;

    TensorHandleFactoryRegistry m_TensorHandleFactoryRegistry;

    profiling::ProfilingService&  m_ProfilingService;
};

}
