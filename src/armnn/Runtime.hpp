//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//
#pragma once

#include "LoadedNetwork.hpp"
#include "DeviceSpec.hpp"
#include <armnn/INetwork.hpp>
#include <armnn/IRuntime.hpp>
#include <armnn/Tensor.hpp>
#include <backends/cl/ClContextControl.hpp>

#include <mutex>
#include <unordered_map>

namespace armnn
{

class Runtime final : public IRuntime
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
                               std::string & errorMessage) override;

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

    /// Creates a runtime for workload execution.
    /// May throw a ClRuntimeUnavailableException if @a defaultComputeDevice requires a CL runtime but
    /// it cannot be setup for some reason.
    Runtime(const CreationOptions& options);

    ~Runtime();

private:
    friend void RuntimeLoadedNetworksReserve(armnn::Runtime* runtime); // See RuntimeTests.cpp

    int GenerateNetworkId();

    LoadedNetwork* GetLoadedNetworkPtr(NetworkId networkId) const;

    mutable std::mutex m_Mutex;

    std::unordered_map<NetworkId, std::unique_ptr<LoadedNetwork>> m_LoadedNetworks;

    ClContextControl m_ClContextControl;

    int m_NetworkIdCounter;

    DeviceSpec m_DeviceSpec;
};

}
