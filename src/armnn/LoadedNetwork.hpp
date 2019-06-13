//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//
#pragma once

#include <armnn/Tensor.hpp>
#include <armnn/Types.hpp>

#include "Network.hpp"
#include "LayerFwd.hpp"
#include "Profiling.hpp"

#include <backendsCommon/IBackendInternal.hpp>
#include <backendsCommon/TensorHandleFactoryRegistry.hpp>
#include <backendsCommon/Workload.hpp>
#include <backendsCommon/WorkloadFactory.hpp>

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
                                                            std::string & errorMessage);

    // NOTE we return by reference as the purpose of this method is only to provide
    // access to the private m_Profiler and in theory we should not need to increment
    // the shared_ptr's reference counter
    const std::shared_ptr<Profiler>& GetProfiler() const { return m_Profiler; }

    void FreeWorkingMemory();

    void RegisterDebugCallback(const DebugCallbackFunction& func);

private:
    void AllocateWorkingMemory();

    LoadedNetwork(std::unique_ptr<OptimizedNetwork> net);

    void EnqueueInput(const BindableLayer& layer, ITensorHandle* tensorHandle, const TensorInfo& tensorInfo);

    void EnqueueOutput(const BindableLayer& layer, ITensorHandle* tensorHandle, const TensorInfo& tensorInfo);

    bool Execute();

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

    TensorHandleFactoryRegistry m_TensorHandleFactoryRegistry;
};

}
