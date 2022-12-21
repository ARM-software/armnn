//
// Copyright © 2017, 2023 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//
#pragma once

#include "LoadedNetwork.hpp"
#include "DeviceSpec.hpp"

#include <armnn/INetwork.hpp>
#include <armnn/IRuntime.hpp>
#include <armnn/Tensor.hpp>
#include <armnn/BackendId.hpp>

#include <armnn/backends/DynamicBackend.hpp>

#include <client/include/IInitialiseProfilingService.hpp>
#include <client/include/IProfilingService.hpp>
#include <client/include/IReportStructure.hpp>

#include <mutex>
#include <unordered_map>

namespace armnn
{
using LoadedNetworks = std::unordered_map<NetworkId, std::unique_ptr<LoadedNetwork>>;
using IReportStructure = arm::pipe::IReportStructure;
    using IInitialiseProfilingService = arm::pipe::IInitialiseProfilingService;

struct RuntimeImpl final :  public IReportStructure, public IInitialiseProfilingService
{
public:
    /// Loads a complete network into the Runtime.
    /// @param [out] networkIdOut - Unique identifier for the network is returned in this reference.
    /// @param [in] network - Complete network to load into the Runtime.
    /// The runtime takes ownership of the network once passed in.
    /// @return armnn::Status
    Status LoadNetwork(NetworkId& networkIdOut, IOptimizedNetworkPtr network);

    /// Load a complete network into the IRuntime.
    /// @param [out] networkIdOut Unique identifier for the network is returned in this reference.
    /// @param [in] network Complete network to load into the IRuntime.
    /// @param [out] errorMessage Error message if there were any errors.
    /// The runtime takes ownership of the network once passed in.
    /// @return armnn::Status
    Status LoadNetwork(NetworkId& networkIdOut,
                       IOptimizedNetworkPtr network,
                       std::string& errorMessage);

    Status LoadNetwork(NetworkId& networkIdOut,
                       IOptimizedNetworkPtr network,
                       std::string& errorMessage,
                       const INetworkProperties& networkProperties);

    armnn::TensorInfo GetInputTensorInfo(NetworkId networkId, LayerBindingId layerId) const;
    armnn::TensorInfo GetOutputTensorInfo(NetworkId networkId, LayerBindingId layerId) const;

    std::vector<ImportedInputId> ImportInputs(NetworkId networkId, const InputTensors& inputTensors,
                                              MemorySource forceImportMemorySource);
    std::vector<ImportedOutputId> ImportOutputs(NetworkId networkId, const OutputTensors& outputTensors,
                                                MemorySource forceImportMemorySource);

    void ClearImportedInputs(NetworkId networkId, const std::vector<ImportedInputId> inputIds);
    void ClearImportedOutputs(NetworkId networkId, const std::vector<ImportedOutputId> outputIds);

    // Evaluates network using input in inputTensors, outputs filled into outputTensors.
    Status EnqueueWorkload(NetworkId networkId,
                           const InputTensors& inputTensors,
                           const OutputTensors& outputTensors,
                           std::vector<ImportedInputId> preImportedInputIds = {},
                           std::vector<ImportedOutputId> preImportedOutputIds = {});

    /// This is an experimental function.
    /// Evaluates a network using input in inputTensors and outputs filled into outputTensors.
    /// This function performs a thread safe execution of the network. Returns once execution is complete.
    /// Will block until this and any other thread using the same workingMem object completes.
    Status Execute(IWorkingMemHandle& workingMemHandle,
                   const InputTensors& inputTensors,
                   const OutputTensors& outputTensors,
                   std::vector<ImportedInputId> preImportedInputs,
                   std::vector<ImportedOutputId> preImportedOutputs);

    /// Unloads a network from the Runtime.
    /// At the moment this only removes the network from the m_Impl->m_Network.
    /// This might need more work in the future to be AndroidNN compliant.
    /// @param [in] networkId Unique identifier for the network to be unloaded. Generated in LoadNetwork().
    /// @return armnn::Status
    Status UnloadNetwork(NetworkId networkId);

    const IDeviceSpec& GetDeviceSpec() const { return m_DeviceSpec; }

    /// Gets the profiler corresponding to the given network id.
    /// @param networkId The id of the network for which to get the profile.
    /// @return A pointer to the requested profiler, or nullptr if not found.
    const std::shared_ptr<IProfiler> GetProfiler(NetworkId networkId) const;

    /// Create a new unique WorkingMemHandle object. Create multiple handles if you wish to have
    /// overlapped Execution by calling this function from different threads.
    std::unique_ptr<IWorkingMemHandle> CreateWorkingMemHandle(NetworkId networkId);

    /// Registers a callback function to debug layers performing custom computations on intermediate tensors.
    /// @param networkId The id of the network to register the callback.
    /// @param func callback function to pass to the debug layer.
    void RegisterDebugCallback(NetworkId networkId, const DebugCallbackFunction& func);

    /// Creates a runtime for workload execution.
    RuntimeImpl(const IRuntime::CreationOptions& options);

    ~RuntimeImpl();

    //NOTE: we won't need the profiling service reference but it is good to pass the service
    // in this way to facilitate other implementations down the road
    void ReportStructure(arm::pipe::IProfilingService& profilingService) override;

    void InitialiseProfilingService(arm::pipe::IProfilingService& profilingService) override;

private:
    friend void RuntimeLoadedNetworksReserve(RuntimeImpl* runtime); // See RuntimeTests.cpp

    friend arm::pipe::IProfilingService& GetProfilingService(RuntimeImpl* runtime); // See RuntimeTests.cpp

    int GenerateNetworkId();

    LoadedNetwork* GetLoadedNetworkPtr(NetworkId networkId) const;

    template<typename Func>
    void LoadedNetworkFuncSafe(NetworkId networkId, Func f)
    {
#if !defined(ARMNN_DISABLE_THREADS)
        std::lock_guard<std::mutex> lockGuard(m_Mutex);
#endif
        auto iter = m_LoadedNetworks.find(networkId);
        if (iter != m_LoadedNetworks.end())
        {
            f(iter->second.get());
        }
    }

    /// Loads any available/compatible dynamic backend in the runtime.
    void LoadDynamicBackends(const std::string& overrideBackendPath);

#if !defined(ARMNN_DISABLE_THREADS)
    mutable std::mutex m_Mutex;
#endif

    /// Map of Loaded Networks with associated GUID as key
    LoadedNetworks m_LoadedNetworks;

    std::unordered_map<BackendId, IBackendInternal::IBackendContextPtr> m_BackendContexts;

    int m_NetworkIdCounter;

    DeviceSpec m_DeviceSpec;

    /// List of dynamic backends loaded in the runtime
    std::vector<DynamicBackendPtr> m_DynamicBackends;

    /// Profiling Service Instance
    std::unique_ptr<arm::pipe::IProfilingService> m_ProfilingService;

    /// Keep track of backend ids of the custom allocators that this instance of the runtime added. The
    /// destructor can then clean up for this runtime.
    std::set<BackendId> m_AllocatorsAddedByThisRuntime;
};

} // namespace armnn
