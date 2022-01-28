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

#include <armnn/backends/ICustomAllocator.hpp>
#include <armnn/backends/IMemoryOptimizerStrategy.hpp>
#include <memory>
#include <map>

namespace armnn
{

using NetworkId = int;

class IGpuAccTunedParameters;

struct RuntimeImpl;
class IRuntime;
using IRuntimePtr = std::unique_ptr<IRuntime, void(*)(IRuntime* runtime)>;

struct INetworkProperties
{   
    INetworkProperties(bool asyncEnabled,
                       MemorySource inputSource,
                       MemorySource outputSource,
                       bool profilingEnabled = false,
                       ProfilingDetailsMethod detailsMethod = ProfilingDetailsMethod::Undefined,
                       bool externalMemoryManagementEnabled = false)
        : m_ImportEnabled(inputSource != MemorySource::Undefined),
          m_ExportEnabled(outputSource != MemorySource::Undefined),
          m_AsyncEnabled(asyncEnabled),
          m_ProfilingEnabled(profilingEnabled),
          m_OutputNetworkDetailsMethod(detailsMethod),
          m_InputSource(inputSource),
          m_OutputSource(outputSource),
          m_ExternalMemoryManagementEnabled(externalMemoryManagementEnabled)
    {}

    /// Deprecated and will be removed in future release.
    const bool m_ImportEnabled;
    /// Deprecated and will be removed in future release.
    const bool m_ExportEnabled;

    const bool m_AsyncEnabled;

    const bool m_ProfilingEnabled;

    const ProfilingDetailsMethod m_OutputNetworkDetailsMethod;

    const MemorySource m_InputSource;
    const MemorySource m_OutputSource;

    const bool m_ExternalMemoryManagementEnabled;

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
            , m_ProtectedMode(false)
            , m_CustomAllocatorMap()
            , m_MemoryOptimizerStrategyMap()
        {}

        /// If set, uses the GpuAcc tuned parameters from the given object when executing GPU workloads.
        /// It will also be updated with new tuned parameters if it is configured to do so.
        std::shared_ptr<IGpuAccTunedParameters> m_GpuAccTunedParameters;

        /// Setting this flag will allow the user to obtain GPU profiling information from the runtime.
        bool m_EnableGpuProfiling;

        /// Setting this value will override the paths set by the DYNAMIC_BACKEND_PATHS compiler directive
        /// Only a single path is allowed for the override
        /// It defines the path to search for any [dynamic backend libraries](src/dynamic/README.md).
        std::string m_DynamicBackendsPath;

        /// Setting this flag will allow the user to create the Runtime in protected mode.
        /// It will run all the inferences on protected memory and will make sure that
        /// INetworkProperties::m_ImportEnabled set to true with MemorySource::DmaBufProtected option
        /// This requires that the backend supports Protected Memory and has an allocator capable of
        /// allocating Protected Memory associated with it.
        bool m_ProtectedMode;

        /// @brief A map to define a custom memory allocator for specific backend Ids.
        ///
        /// @details  A Custom Allocator is used for allocation of working memory in the backends.
        /// Set this if you need to take control of how memory is allocated on a backend. Required for
        /// Protected Mode in order to correctly allocate Protected Memory
        ///
        /// @note Only supported for GpuAcc
        std::map<BackendId, std::shared_ptr<ICustomAllocator>> m_CustomAllocatorMap;

        /// @brief A map to define a custom memory optimizer strategy for specific backend Ids.
        ///
        /// @details  A Memory Optimizer Strategy provides a solution to an abstract representation of
        /// a network's memory requirements. This can also be used to return a pre-computed solution
        /// for a specific network. Set this if you want to implement a Custom Memory Optimizer Strategy
        /// for a given backend.
        std::map<BackendId, std::shared_ptr<IMemoryOptimizerStrategy>> m_MemoryOptimizerStrategyMap;

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

            /// Indicates whether external profiling is enabled or not.
            bool        m_EnableProfiling;
            /// Indicates whether external timeline profiling is enabled or not.
            bool        m_TimelineEnabled;
            /// Path to a file in which outgoing timeline profiling messages will be stored.
            std::string m_OutgoingCaptureFile;
            /// Path to a file in which incoming timeline profiling messages will be stored.
            std::string m_IncomingCaptureFile;
            /// Enable profiling output to file only.
            bool        m_FileOnly;
            /// The duration at which captured profiling messages will be flushed.
            uint32_t    m_CapturePeriod;
            /// The format of the file used for outputting profiling data.
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
        ///         {"MemoryOptimizerStrategy", strategyname}
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
        /// AllBackends:
        ///   "MemoryOptimizerStrategy" : string [stategynameString]
        ///    (Existing Memory Optimizer Strategies: ConstantMemoryStrategy)
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

    /// ImportInputs separates the importing and mapping of InputTensors from network execution.
    /// Allowing for a set of InputTensors to be imported and mapped once, but used in execution many times.
    /// This function is not thread safe and must not be used while other threads are calling Execute().
    /// Only compatible with AsyncEnabled networks and aligned memory import
    std::vector<ImportedInputId> ImportInputs(NetworkId networkId, const InputTensors& inputTensors,
                                              MemorySource forceImportMemorySource = MemorySource::Undefined);

    /// ImportOutputs separates the importing and mapping of OutputTensors from network execution.
    /// Allowing for a set of OutputTensors to be imported and mapped once, but used in execution many times.
    /// This function is not thread safe and must not be used while other threads are calling Execute().
    /// Only compatible with AsyncEnabled networks and aligned memory import
    std::vector<ImportedOutputId> ImportOutputs(NetworkId networkId, const OutputTensors& outputTensors,
                                                MemorySource forceImportMemorySource = MemorySource::Undefined);

    /// Un-import and delete the imported InputTensor/s
    /// This function is not thread safe and must not be used while other threads are calling Execute().
    /// Only compatible with AsyncEnabled networks
    void ClearImportedInputs(NetworkId networkId, const std::vector<ImportedInputId> inputIds);

    /// Un-import and delete the imported OutputTensor/s
    /// This function is not thread safe and must not be used while other threads are calling Execute().
    /// Only compatible with AsyncEnabled networks
    void ClearImportedOutputs(NetworkId networkId, const std::vector<ImportedOutputId> outputIds);

    /// Evaluates a network using input in inputTensors and outputs filled into outputTensors
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
                   std::vector<ImportedInputId> preImportedInputs = {},
                   std::vector<ImportedOutputId> preImportedOutputs = {});

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
