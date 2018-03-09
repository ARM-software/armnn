//
// Copyright © 2017 Arm Ltd. All rights reserved.
// See LICENSE file in the project root for full license information.
//
#pragma once

#include <memory>

#include "Types.hpp"
#include "Tensor.hpp"
#include "INetwork.hpp"
#include "TypesUtils.hpp"

namespace armnn
{

using NetworkId = int;

class IClTunedParameters;

class IRuntime;
using IRuntimePtr = std::unique_ptr<IRuntime, void(*)(IRuntime* runtime)>;

class IRuntime
{
public:
    struct CreationOptions
    {
        Compute m_DefaultComputeDevice;
        bool m_UseCpuRefAsFallback;
        /// If set, uses the CL tuned parameters from the given object when executing CL workloads.
        /// It will also be updated with new tuned parameters if it is configured to do so.
        IClTunedParameters* m_ClTunedParameters;

        CreationOptions(Compute defaultComputeDevice)
            : m_DefaultComputeDevice(defaultComputeDevice)
            , m_UseCpuRefAsFallback(true)
            , m_ClTunedParameters(nullptr)
        {
        }
    };

    static IRuntime* CreateRaw(const CreationOptions& options);
    static IRuntimePtr Create(const CreationOptions& options);
    static void Destroy(IRuntime* runtime);

    /// Load a complete network into the IRuntime.
    /// @param [out] networkIdOut Unique identifier for the network is returned in this reference.
    /// @param [in] network Complete network to load into the IRuntime.
    /// The runtime takes ownership of the network once passed in.
    /// @return armnn::Status
    virtual Status LoadNetwork(NetworkId& networkIdOut, IOptimizedNetworkPtr network) = 0;

    virtual TensorInfo GetInputTensorInfo(NetworkId networkId, LayerBindingId layerId) const = 0;
    virtual TensorInfo GetOutputTensorInfo(NetworkId networkId, LayerBindingId layerId) const = 0;

    // Evaluate network using input in inputTensors, outputs filled into outputTensors
    virtual Status EnqueueWorkload(NetworkId networkId,
                           const InputTensors& inputTensors,
                           const OutputTensors& outputTensors) = 0;

    /// Unload a network from the IRuntime.
    /// At the moment this only removes the network from the m_Impl->m_Network.
    /// This might need more work in the future to be AndroidNN compliant.
    /// @param [in] networkId Unique identifier for the network to be unloaded. Generated in LoadNetwork().
    /// @return armnn::Status
    virtual Status UnloadNetwork(NetworkId networkId) = 0;

    virtual const DeviceSpec& GetDeviceSpec() const = 0;

protected:
    ~IRuntime() {}
};

using IClTunedParametersPtr = std::unique_ptr<IClTunedParameters, void(*)(IClTunedParameters* params)>;

/// Manages a set of Open CL parameters which have been tuned for maximum performance.
/// Pass an instance of this object to the IRuntime::Create() method (via IRuntime::CreationOptions) to use it
/// for all CL workload execution.
///
/// Can be created in two modes:
///     - In UseTunedParameters mode the parameters stored in this object are used to execute CL workloads.
///     - In UpdateTunedParameters mode, additionally, whenever a CL workload is executed for the first time the
///       optimum parameters will be found and stored in this object. WARNING - This tuning can be slow.
///
/// The parameters can be loaded from and saved to a file so that you first run a slow initial read-write
/// execution, save the parameters for later and then run fast read-only executions using the optimised parameters.
class IClTunedParameters
{
public:
    enum class Mode
    {
        UseTunedParameters,
        UpdateTunedParameters
    };

    /// Creates an IClTunedParameters with the given mode.
    /// @{
    static IClTunedParameters* CreateRaw(Mode mode);
    static IClTunedParametersPtr Create(Mode mode);
    /// @}
    static void Destroy(IClTunedParameters* params);

    /// Loads an existing set of tuned parameters from the given file.
    /// If there is an error loading the file, an armnn::Exception is thrown.
    virtual void Load(const char* filename) = 0;

    /// Saves the current set of tuned parameters to the given file.
    /// If there is an error saving to the file, an armnn::Exception is thrown.
    virtual void Save(const char* filename) const = 0;

protected:
    virtual ~IClTunedParameters() {};
};

}
