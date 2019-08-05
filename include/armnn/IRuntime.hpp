//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//
#pragma once


#include "INetwork.hpp"
#include "IProfiler.hpp"
#include "Tensor.hpp"
#include "Types.hpp"
#include "TypesUtils.hpp"

#include <memory>

namespace armnn
{

using NetworkId = int;

class IGpuAccTunedParameters;

class IRuntime;
using IRuntimePtr = std::unique_ptr<IRuntime, void(*)(IRuntime* runtime)>;

class IRuntime
{
public:
    struct CreationOptions
    {
        CreationOptions()
            : m_GpuAccTunedParameters(nullptr)
            , m_EnableGpuProfiling(false)
            , m_DynamicBackendsPath("")
        {}

        /// If set, uses the GpuAcc tuned parameters from the given object when executing GPU workloads.
        /// It will also be updated with new tuned parameters if it is configured to do so.
        std::shared_ptr<IGpuAccTunedParameters> m_GpuAccTunedParameters;

        // Setting this flag will allow the user to obtain GPU profiling information from the runtime.
        bool m_EnableGpuProfiling;

        // Setting this value will override the paths set by the DYNAMIC_BACKEND_PATHS compiler directive
        // Only a single path is allowed for the override
        std::string m_DynamicBackendsPath;
    };

    static IRuntime* CreateRaw(const CreationOptions& options);
    static IRuntimePtr Create(const CreationOptions& options);
    static void Destroy(IRuntime* runtime);

    /// Loads a complete network into the IRuntime.
    /// @param [out] networkIdOut - Unique identifier for the network is returned in this reference.
    /// @param [in] network - Complete network to load into the IRuntime.
    /// The runtime takes ownership of the network once passed in.
    /// @return armnn::Status
    virtual Status LoadNetwork(NetworkId& networkIdOut, IOptimizedNetworkPtr network) = 0;

    /// Load a complete network into the IRuntime.
    /// @param [out] networkIdOut Unique identifier for the network is returned in this reference.
    /// @param [in] network Complete network to load into the IRuntime.
    /// @param [out] errorMessage Error message if there were any errors.
    /// The runtime takes ownership of the network once passed in.
    /// @return armnn::Status
    virtual Status LoadNetwork(NetworkId& networkIdOut,
                               IOptimizedNetworkPtr network,
                               std::string & errorMessage) = 0;

    virtual TensorInfo GetInputTensorInfo(NetworkId networkId, LayerBindingId layerId) const = 0;
    virtual TensorInfo GetOutputTensorInfo(NetworkId networkId, LayerBindingId layerId) const = 0;

    /// Evaluates a network using input in inputTensors and outputs filled into outputTensors
    virtual Status EnqueueWorkload(NetworkId networkId,
                                   const InputTensors& inputTensors,
                                   const OutputTensors& outputTensors) = 0;

    /// Unloads a network from the IRuntime.
    /// At the moment this only removes the network from the m_Impl->m_Network.
    /// This might need more work in the future to be AndroidNN compliant.
    /// @param [in] networkId - Unique identifier for the network to be unloaded. Generated in LoadNetwork().
    /// @return armnn::Status
    virtual Status UnloadNetwork(NetworkId networkId) = 0;

    virtual const IDeviceSpec& GetDeviceSpec() const = 0;

    /// Gets the profiler corresponding to the given network id.
    /// @param networkId The id of the network for which to get the profile.
    /// @return A pointer to the requested profiler, or nullptr if not found.
    virtual const std::shared_ptr<IProfiler> GetProfiler(NetworkId networkId) const = 0;

    /// Registers a callback function to debug layers performing custom computations on intermediate tensors.
    /// @param networkId The id of the network to register the callback.
    /// @param func callback function to pass to the debug layer.
    virtual void RegisterDebugCallback(NetworkId networkId, const DebugCallbackFunction& func) = 0;

protected:
    ~IRuntime() {}
};

using IGpuAccTunedParametersPtr = std::shared_ptr<IGpuAccTunedParameters>;

/// Manages a set of GpuAcc parameters which have been tuned for maximum performance.
/// Passes an instance of this object to the IRuntime::Create() method (via IRuntime::CreationOptions) to use it
/// for all GPU workload execution.
///
/// Can be created in two modes:
///     - In UseTunedParameters mode, the parameters stored in this object are used to execute GPU workloads.
///     - In UpdateTunedParameters mode, additionally, whenever a GPU workload is executed for the first time, the
///       optimum parameters will be found and stored in this object. WARNING - This tuning can be slow.
///
/// The parameters can be loaded from and saved to a file so that you can first run a slow initial read-write
/// execution, save the parameters for later and then run fast read-only executions using the optimised parameters.
class IGpuAccTunedParameters
{
public:
    enum class Mode
    {
        UseTunedParameters,
        UpdateTunedParameters
    };

    enum class TuningLevel
    {
        Rapid = 0,
        Normal = 1,
        Exhaustive = 2
    };

    /// Creates an IClTunedParameters with the given mode.
    /// @{
    static IGpuAccTunedParameters* CreateRaw(Mode mode, TuningLevel tunerMode);
    static IGpuAccTunedParametersPtr Create(Mode mode, TuningLevel tunerMode);
    /// @}
    static void Destroy(IGpuAccTunedParameters* params);

    /// Loads an existing set of tuned parameters from the given file.
    /// If there is an error loading the file, an armnn::Exception is thrown.
    virtual void Load(const char* filename) = 0;

    /// Saves the current set of tuned parameters to the given file.
    /// If there is an error saving to the file, an armnn::Exception is thrown.
    virtual void Save(const char* filename) const = 0;

protected:
    virtual ~IGpuAccTunedParameters() {};
};

}
