//
// Copyright © 2017 Arm Ltd and Contributors. All rights reserved.
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

#include <ProfilingService.hpp>

#include <IProfilingService.hpp>
#include <IReportStructure.hpp>

#include <mutex>
#include <unordered_map>

namespace armnn
{
using LoadedNetworks = std::unordered_map<NetworkId, std::unique_ptr<LoadedNetwork>>;
using IReportStructure = profiling::IReportStructure;

class Runtime final :  public IRuntime,
                       public IReportStructure
{
public:
    /// Loads a complete network into the Runtime.
    /// @param [out] networkIdOut - Unique identifier for the network is returned in this reference.
    /// @param [in] network - Complete network to load into the Runtime.
    /// The runtime takes ownership of the network once passed in.
    /// @return armnn::Status
    virtual Status LoadNetwork(NetworkId& networkIdOut, IOptimizedNetworkPtr network) override;

    /// Load a complete network into the IRuntime.
    /// @param [out] networkIdOut Unique identifier for the network is returned in this reference.
    /// @param [in] network Complete network to load into the IRuntime.
    /// @param [out] errorMessage Error message if there were any errors.
    /// The runtime takes ownership of the network once passed in.
    /// @return armnn::Status
    virtual Status LoadNetwork(NetworkId& networkIdOut,
                               IOptimizedNetworkPtr network,
                               std::string& errorMessage) override;

    virtual Status LoadNetwork(NetworkId& networkIdOut,
                               IOptimizedNetworkPtr network,
                               std::string& errorMessage,
                               const INetworkProperties& networkProperties) override;

    virtual TensorInfo GetInputTensorInfo(NetworkId networkId, LayerBindingId layerId) const override;
    virtual TensorInfo GetOutputTensorInfo(NetworkId networkId, LayerBindingId layerId) const override;

    // Evaluates network using input in inputTensors, outputs filled into outputTensors.
    virtual Status EnqueueWorkload(NetworkId networkId,
        const InputTensors& inputTensors,
        const OutputTensors& outputTensors) override;

    /// Unloads a network from the Runtime.
    /// At the moment this only removes the network from the m_Impl->m_Network.
    /// This might need more work in the future to be AndroidNN compliant.
    /// @param [in] networkId Unique identifier for the network to be unloaded. Generated in LoadNetwork().
    /// @return armnn::Status
    virtual Status UnloadNetwork(NetworkId networkId) override;

    virtual const IDeviceSpec& GetDeviceSpec() const override { return m_DeviceSpec; }

    /// Gets the profiler corresponding to the given network id.
    /// @param networkId The id of the network for which to get the profile.
    /// @return A pointer to the requested profiler, or nullptr if not found.
    virtual const std::shared_ptr<IProfiler> GetProfiler(NetworkId networkId) const override;

    /// Registers a callback function to debug layers performing custom computations on intermediate tensors.
    /// @param networkId The id of the network to register the callback.
    /// @param func callback function to pass to the debug layer.
    virtual void RegisterDebugCallback(NetworkId networkId, const DebugCallbackFunction& func) override;

    /// Creates a runtime for workload execution.
    Runtime(const CreationOptions& options);

    ~Runtime();

    //NOTE: we won't need the profiling service reference but it is good to pass the service
    // in this way to facilitate other implementations down the road
    virtual void ReportStructure() override;

private:
    friend void RuntimeLoadedNetworksReserve(armnn::Runtime* runtime); // See RuntimeTests.cpp

    friend profiling::ProfilingService& GetProfilingService(armnn::Runtime* runtime); // See RuntimeTests.cpp

    int GenerateNetworkId();

    LoadedNetwork* GetLoadedNetworkPtr(NetworkId networkId) const;

    template<typename Func>
    void LoadedNetworkFuncSafe(NetworkId networkId, Func f)
    {
        std::lock_guard<std::mutex> lockGuard(m_Mutex);
        auto iter = m_LoadedNetworks.find(networkId);
        if (iter != m_LoadedNetworks.end())
        {
            f(iter->second.get());
        }
    }

    /// Loads any available/compatible dynamic backend in the runtime.
    void LoadDynamicBackends(const std::string& overrideBackendPath);

    mutable std::mutex m_Mutex;

    /// Map of Loaded Networks with associated GUID as key
    LoadedNetworks m_LoadedNetworks;

    std::unordered_map<BackendId, IBackendInternal::IBackendContextPtr> m_BackendContexts;

    int m_NetworkIdCounter;

    DeviceSpec m_DeviceSpec;

    /// List of dynamic backends loaded in the runtime
    std::vector<DynamicBackendPtr> m_DynamicBackends;

    /// Profiling Service Instance
    profiling::ProfilingService m_ProfilingService;
};

} // namespace armnn
