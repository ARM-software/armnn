//
// Copyright © 2021 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <armnn/IAsyncNetwork.hpp>
#include <armnn/Tensor.hpp>
#include <armnn/Types.hpp>

#include "LayerFwd.hpp"
#include "Network.hpp"
#include "Profiling.hpp"
#include "WorkingMemHandle.hpp"

#include <armnn/backends/IBackendInternal.hpp>
#include <backendsCommon/TensorHandleFactoryRegistry.hpp>
#include <backendsCommon/Workload.hpp>
#include <backendsCommon/WorkloadFactory.hpp>
#include <ProfilingService.hpp>
#include <TimelineUtilityMethods.hpp>

#include <unordered_map>

namespace armnn
{

namespace experimental
{

class AsyncNetwork final : public IAsyncNetwork
{
public:
    using WorkloadQueue = std::vector<std::unique_ptr<IWorkload>>;

    AsyncNetwork(std::unique_ptr<IOptimizedNetwork> net,
                 const INetworkProperties &networkProperties,
                 profiling::ProfilingService &profilingService);

    ~AsyncNetwork() { FreeWorkingMemory(); }

    TensorInfo GetInputTensorInfo(LayerBindingId layerId) const override;
    TensorInfo GetOutputTensorInfo(LayerBindingId layerId) const override;

    /// Thread safe execution of the network. Returns once execution is complete.
    /// Will block until this and any other thread using the same workingMem object completes.
    virtual Status Execute(const InputTensors& inputTensors,
                           const OutputTensors& outputTensors,
                           IWorkingMemHandle& workingMemHandle) override;

    /// Create a new unique WorkingMemHandle object. Create multiple handles if you wish to have
    /// overlapped Execution by calling this function from different threads.
    std::unique_ptr<IWorkingMemHandle> CreateWorkingMemHandle() override;

    /// Get the profiler used for this network
    std::shared_ptr<IProfiler> GetProfiler() const override;

    /// Register a debug callback function to be used with this network
    void RegisterDebugCallback(const DebugCallbackFunction& func) override;

private:
    void FreeWorkingMemory();

    void CollectInputTensorHandles(std::unordered_map<LayerGuid, std::vector<ITensorHandle*> >& tensorHandles,
                                   std::vector<ITensorHandle*>& inputs,
                                   const armnn::Layer* layer,
                                   const TensorHandleFactoryRegistry& registry,
                                   const bool isMemoryManaged = false);

    void CreateOutputTensorHandles(std::unordered_map<LayerGuid, std::vector<ITensorHandle*> >& tensorHandles,
                                   std::vector<ITensorHandle*>& outputs,
                                   const armnn::Layer* layer,
                                   const TensorHandleFactoryRegistry& registry,
                                   const bool isMemoryManaged = false);

    void EnqueueInput(const BindableLayer& layer, const ConstTensor& inputTensor, WorkingMemHandle& handle);

    void EnqueueOutput(const BindableLayer& layer, const Tensor& outputTensor, WorkingMemHandle& handle);

    using BackendPtrMap = std::unordered_map<BackendId, IBackendInternalUniquePtr>;

    using WorkloadFactoryWithMemoryManager =
            std::pair<IBackendInternal::IWorkloadFactoryPtr, IBackendInternal::IMemoryManagerSharedPtr>;

    using WorkloadFactoryMap = std::unordered_map<BackendId, WorkloadFactoryWithMemoryManager>;

    const IWorkloadFactory& GetWorkloadFactory(const Layer& layer) const;

    BackendPtrMap m_Backends;
    WorkloadFactoryMap m_WorkloadFactories;

    std::unique_ptr<IOptimizedNetwork> m_OptimizedNetwork;
    INetworkProperties m_NetworkProperties;
    WorkloadQueue m_WorkloadQueue;
    std::shared_ptr<IProfiler> m_Profiler;

    TensorHandleFactoryRegistry m_TensorHandleFactoryRegistry;

    /// Profiling Service Instance
    profiling::ProfilingService& m_ProfilingService;
};

} // end experimental namespace

} // end armnn namespace
