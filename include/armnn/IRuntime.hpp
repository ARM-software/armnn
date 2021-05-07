//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//
#pragma once

#include "BackendOptions.hpp"
#include "INetwork.hpp"
#include "IProfiler.hpp"
#include "IWorkingMemHandle.hpp"
#include "IAsyncExecutionCallback.hpp"
#include "Tensor.hpp"
#include "Types.hpp"
#include "TypesUtils.hpp"
#include "profiling/ILocalPacketHandler.hpp"

#include <memory>

namespace armnn
{

using NetworkId = int;

class IGpuAccTunedParameters;

struct RuntimeImpl;
class IRuntime;
using IRuntimePtr = std::unique_ptr<IRuntime, void(*)(IRuntime* runtime)>;

struct INetworkProperties
{
    ARMNN_DEPRECATED_MSG("Please use INetworkProperties constructor with MemorySource argument")
    INetworkProperties(bool importEnabled = false,
                       bool exportEnabled = false,
                       bool asyncEnabled = false,
                       size_t numThreads = 0)
        : m_ImportEnabled(importEnabled)
        , m_ExportEnabled(exportEnabled)
        , m_AsyncEnabled(asyncEnabled)
        , m_NumThreads(numThreads)
        , m_InputSource(m_ImportEnabled ? MemorySource::Malloc : MemorySource::Undefined)
        , m_OutputSource(m_ExportEnabled ? MemorySource::Malloc : MemorySource::Undefined)
    {}

    INetworkProperties(bool asyncEnabled,
                       MemorySource m_InputSource,
                       MemorySource m_OutputSource,
                       size_t numThreads = 0)
        : m_ImportEnabled(m_InputSource != MemorySource::Undefined)
        , m_ExportEnabled(m_OutputSource != MemorySource::Undefined)
        , m_AsyncEnabled(asyncEnabled)
        , m_NumThreads(numThreads)
        , m_InputSource(m_InputSource)
        , m_OutputSource(m_OutputSource)
        {}

    /// Deprecated and will be removed in future release.
    const bool m_ImportEnabled;
    /// Deprecated and will be removed in future release.
    const bool m_ExportEnabled;

    const bool   m_AsyncEnabled;
    const size_t m_NumThreads;

    const MemorySource m_InputSource;
    const MemorySource m_OutputSource;

    virtual ~INetworkProperties() {}
};

using namespace armnn::experimental;

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

        /// Setting this flag will allow the user to obtain GPU profiling information from the runtime.
        bool m_EnableGpuProfiling;

        /// Setting this value will override the paths set by the DYNAMIC_BACKEND_PATHS compiler directive
        /// Only a single path is allowed for the override
        std::string m_DynamicBackendsPath;

        struct ExternalProfilingOptions
        {
            ExternalProfilingOptions()
                : m_EnableProfiling(false)
                , m_TimelineEnabled(false)
                , m_OutgoingCaptureFile("")
                , m_IncomingCaptureFile("")
                , m_FileOnly(false)
                , m_CapturePeriod(LOWEST_CAPTURE_PERIOD)
                , m_FileFormat("binary")
                , m_LocalPacketHandlers()
            {}

            bool        m_EnableProfiling;
            bool        m_TimelineEnabled;
            std::string m_OutgoingCaptureFile;
            std::string m_IncomingCaptureFile;
            bool        m_FileOnly;
            uint32_t    m_CapturePeriod;
            std::string m_FileFormat;
            std::vector<armnn::profiling::ILocalPacketHandlerSharedPtr> m_LocalPacketHandlers;
        };
        ExternalProfilingOptions m_ProfilingOptions;

        /// Pass backend specific options.
        ///
        /// For example, to enable GpuAcc tuning add the following
        /// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~.cpp
        /// m_BackendOption.emplace_back(
        ///     BackendOptions{"GpuAcc",
        ///       {
        ///         {"TuningLevel", 2},
        ///         {"TuningFile", filename}
        ///       }
        ///     });
        /// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        /// Execute representative workloads through the runtime to generate tuning data.
        /// The tuning file is written once the runtime is destroyed

        /// To execute with the tuning data, start up with just the tuning file specified.
        /// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~.cpp
        /// m_BackendOption.emplace_back(
        ///     BackendOptions{"GpuAcc",
        ///       {
        ///         {"TuningFile", filename}
        ///       }
        ///     });
        /// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        /// The following backend options are available:
        /// GpuAcc:
        ///   "TuningLevel" : int [0..3] (0=UseOnly(default) | 1=RapidTuning | 2=NormalTuning | 3=ExhaustiveTuning)
        ///   "TuningFile" : string [filenameString]
        ///   "KernelProfilingEnabled" : bool [true | false]
        std::vector<BackendOptions> m_BackendOptions;
    };

    static IRuntime* CreateRaw(const CreationOptions& options);
    static IRuntimePtr Create(const CreationOptions& options);
    static void Destroy(IRuntime* runtime);

    /// Loads a complete network into the IRuntime.
    /// @param [out] networkIdOut - Unique identifier for the network is returned in this reference.
    /// @param [in] network - Complete network to load into the IRuntime.
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

    TensorInfo GetInputTensorInfo(NetworkId networkId, LayerBindingId layerId) const;
    TensorInfo GetOutputTensorInfo(NetworkId networkId, LayerBindingId layerId) const;

    /// Evaluates a network using input in inputTensors and outputs filled into outputTensors
    Status EnqueueWorkload(NetworkId networkId,
                           const InputTensors& inputTensors,
                           const OutputTensors& outputTensors);

    /// This is an experimental function.
    /// Evaluates a network using input in inputTensors and outputs filled into outputTensors.
    /// This function performs a thread safe execution of the network. Returns once execution is complete.
    /// Will block until this and any other thread using the same workingMem object completes.
    Status Execute(IWorkingMemHandle& workingMemHandle,
                   const InputTensors& inputTensors,
                   const OutputTensors& outputTensors);

    /// This is an experimental function
    /// Schedule a thread safe execution by taking the input tensors and an execution priority for Quality of Service.
    /// The output tensors will then be filled and the callback object will notify that the execution has either
    /// succeeded or failed.
    void Schedule(NetworkId networkId,
                  const InputTensors& inputTensors,
                  const OutputTensors& outputTensors,
                  const QosExecPriority priority,
                  std::shared_ptr<IAsyncExecutionCallback> callback);

    /// Unloads a network from the IRuntime.
    /// At the moment this only removes the network from the m_Impl->m_Network.
    /// This might need more work in the future to be AndroidNN compliant.
    /// @param [in] networkId - Unique identifier for the network to be unloaded. Generated in LoadNetwork().
    /// @return armnn::Status
    Status UnloadNetwork(NetworkId networkId);

    const IDeviceSpec& GetDeviceSpec() const;

    /// Create a new unique WorkingMemHandle object. Create multiple handles if you wish to have
    /// overlapped Execution by calling this function from different threads.
    std::unique_ptr<IWorkingMemHandle> CreateWorkingMemHandle(NetworkId networkId);

    /// Gets the profiler corresponding to the given network id.
    /// @param networkId The id of the network for which to get the profile.
    /// @return A pointer to the requested profiler, or nullptr if not found.
    const std::shared_ptr<IProfiler> GetProfiler(NetworkId networkId) const;

    /// Registers a callback function to debug layers performing custom computations on intermediate tensors.
    /// @param networkId The id of the network to register the callback.
    /// @param func callback function to pass to the debug layer.
    void RegisterDebugCallback(NetworkId networkId, const DebugCallbackFunction& func);

protected:
    IRuntime();
    IRuntime(const IRuntime::CreationOptions& options);
    ~IRuntime();

    std::unique_ptr<RuntimeImpl> pRuntimeImpl;
};


/// The following API is replaced by the backend options API.
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
        Rapid = 1,
        Normal = 2,
        Exhaustive = 3
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

} // namespace armnn
