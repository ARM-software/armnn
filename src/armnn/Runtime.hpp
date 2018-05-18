//
// Copyright © 2017 Arm Ltd. All rights reserved.
// See LICENSE file in the project root for full license information.
//
#pragma once

#include "LoadedNetwork.hpp"
#include "armnn/INetwork.hpp"
#include "armnn/IRuntime.hpp"
#include "armnn/Tensor.hpp"
#include "backends/ClContextControl.hpp"

#include <mutex>
#include <unordered_map>

namespace armnn
{

class Runtime final : public IRuntime
{
public:
    /// Load a complete network into the Runtime.
    /// @param [out] networkIdOut Unique identifier for the network is returned in this reference.
    /// @param [in] network Complete network to load into the Runtime.
    /// The runtime takes ownership of the network once passed in.
    /// @return armnn::Status
    virtual Status LoadNetwork(NetworkId& networkIdOut, IOptimizedNetworkPtr network) override;

    virtual TensorInfo GetInputTensorInfo(NetworkId networkId, LayerBindingId layerId) const override;
    virtual TensorInfo GetOutputTensorInfo(NetworkId networkId, LayerBindingId layerId) const override;

    // Evaluate network using input in inputTensors, outputs filled into outputTensors
    virtual Status EnqueueWorkload(NetworkId networkId,
        const InputTensors& inputTensors,
        const OutputTensors& outputTensors) override;

    /// Unload a network from the Runtime.
    /// At the moment this only removes the network from the m_Impl->m_Network.
    /// This might need more work in the future to be AndroidNN compliant.
    /// @param [in] networkId Unique identifier for the network to be unloaded. Generated in LoadNetwork().
    /// @return armnn::Status
    virtual Status UnloadNetwork(NetworkId networkId) override;

    virtual const DeviceSpec& GetDeviceSpec() const override { return m_DeviceSpec; }

    /// Creates a runtime for workload execution.
    /// May throw a ClRuntimeUnavailableException if @a defaultComputeDevice requires a CL runtime but
    /// it cannot be setup for some reason.
    Runtime(const CreationOptions& options);

    ~Runtime();

private:
    friend void RuntimeLoadedNetworksReserve(armnn::Runtime* runtime); // see RuntimeTests.cpp

    int GenerateNetworkId();

    LoadedNetwork* GetLoadedNetworkPtr(NetworkId networkId) const;

    mutable std::mutex m_Mutex;

    std::unordered_map<NetworkId, std::unique_ptr<LoadedNetwork>> m_LoadedNetworks;

    ClContextControl m_ClContextControl;

    int m_NetworkIdCounter;

    bool m_UseCpuRefAsFallback;

    DeviceSpec m_DeviceSpec;
};

}
