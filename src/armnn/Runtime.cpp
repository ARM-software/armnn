//
// Copyright © 2017 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//
#include "Runtime.hpp"

#include <armnn/Version.hpp>
#include <armnn/BackendRegistry.hpp>
#include <armnn/BackendHelper.hpp>
#include <armnn/Logging.hpp>
#include <armnn/utility/Timer.hpp>

#include <armnn/backends/IBackendContext.hpp>
#include <backendsCommon/DynamicBackendUtils.hpp>
#include <backendsCommon/memoryOptimizerStrategyLibrary/MemoryOptimizerStrategyLibrary.hpp>
#include <armnn/utility/PolymorphicDowncast.hpp>

#include <common/include/LabelsAndEventClasses.hpp>

#include <iostream>

#include <backends/BackendProfiling.hpp>

using namespace armnn;
using namespace std;

namespace armnn
{
IRuntime::IRuntime() : pRuntimeImpl( new RuntimeImpl(armnn::IRuntime::CreationOptions())) {}

IRuntime::IRuntime(const IRuntime::CreationOptions& options) : pRuntimeImpl(new RuntimeImpl(options)) {}

IRuntime::~IRuntime() = default;

IRuntime* IRuntime::CreateRaw(const CreationOptions& options)
{
    return new IRuntime(options);
}

IRuntimePtr IRuntime::Create(const CreationOptions& options)
{
    return IRuntimePtr(CreateRaw(options), &IRuntime::Destroy);
}

void IRuntime::Destroy(IRuntime* runtime)
{
    delete runtime;
}

Status IRuntime::LoadNetwork(NetworkId& networkIdOut, IOptimizedNetworkPtr network)
{
    return pRuntimeImpl->LoadNetwork(networkIdOut, std::move(network));
}

Status IRuntime::LoadNetwork(NetworkId& networkIdOut,
                             IOptimizedNetworkPtr network,
                             std::string& errorMessage)
{
    return pRuntimeImpl->LoadNetwork(networkIdOut, std::move(network), errorMessage);
}

Status IRuntime::LoadNetwork(NetworkId& networkIdOut,
                             IOptimizedNetworkPtr network,
                             std::string& errorMessage,
                             const INetworkProperties& networkProperties)
{
    return pRuntimeImpl->LoadNetwork(networkIdOut, std::move(network), errorMessage, networkProperties);
}

armnn::TensorInfo IRuntime::GetInputTensorInfo(NetworkId networkId, LayerBindingId layerId) const
{
    return pRuntimeImpl->GetInputTensorInfo(networkId, layerId);
}

armnn::TensorInfo IRuntime::GetOutputTensorInfo(NetworkId networkId, LayerBindingId layerId) const
{
    return pRuntimeImpl->GetOutputTensorInfo(networkId, layerId);
}

std::vector<ImportedInputId> IRuntime::ImportInputs(NetworkId networkId, const InputTensors& inputTensors,
                                                    MemorySource forceImportMemorySource)
{
    return pRuntimeImpl->ImportInputs(networkId, inputTensors, forceImportMemorySource);
}

std::vector<ImportedOutputId> IRuntime::ImportOutputs(NetworkId networkId, const OutputTensors& outputTensors,
                                                      MemorySource forceImportMemorySource)
{
    return pRuntimeImpl->ImportOutputs(networkId, outputTensors, forceImportMemorySource);
}

void IRuntime::ClearImportedInputs(NetworkId networkId, const std::vector<ImportedInputId> inputIds)
{
    return pRuntimeImpl->ClearImportedInputs(networkId, inputIds);
}
void IRuntime::ClearImportedOutputs(NetworkId networkId, const std::vector<ImportedOutputId> outputIds)
{
    return pRuntimeImpl->ClearImportedOutputs(networkId, outputIds);
}

Status IRuntime::EnqueueWorkload(NetworkId networkId,
                                 const InputTensors& inputTensors,
                                 const OutputTensors& outputTensors,
                                 std::vector<ImportedInputId> preImportedInputIds,
                                 std::vector<ImportedOutputId> preImportedOutputIds)
{
    return pRuntimeImpl->EnqueueWorkload(networkId, inputTensors, outputTensors,
                                         preImportedInputIds, preImportedOutputIds);
}

Status IRuntime::Execute(IWorkingMemHandle& workingMemHandle,
                         const InputTensors& inputTensors,
                         const OutputTensors& outputTensors,
                         std::vector<ImportedInputId> preImportedInputs,
                         std::vector<ImportedOutputId> preImportedOutputs)
{
    return pRuntimeImpl->Execute(workingMemHandle, inputTensors, outputTensors, preImportedInputs, preImportedOutputs);
}

Status IRuntime::UnloadNetwork(NetworkId networkId)
{
    return pRuntimeImpl->UnloadNetwork(networkId);
}

const IDeviceSpec& IRuntime::GetDeviceSpec() const
{
    return pRuntimeImpl->GetDeviceSpec();
}

std::unique_ptr<IWorkingMemHandle> IRuntime::CreateWorkingMemHandle(NetworkId networkId)
{
    return pRuntimeImpl->CreateWorkingMemHandle(networkId);
}

const std::shared_ptr<IProfiler> IRuntime::GetProfiler(NetworkId networkId) const
{
    return pRuntimeImpl->GetProfiler(networkId);
}

void IRuntime::RegisterDebugCallback(NetworkId networkId, const DebugCallbackFunction& func)
{
    return pRuntimeImpl->RegisterDebugCallback(networkId, func);
}

int RuntimeImpl::GenerateNetworkId()
{
    return m_NetworkIdCounter++;
}

Status RuntimeImpl::LoadNetwork(NetworkId& networkIdOut, IOptimizedNetworkPtr inNetwork)
{
    std::string ignoredErrorMessage;
    return LoadNetwork(networkIdOut, std::move(inNetwork), ignoredErrorMessage);
}

Status RuntimeImpl::LoadNetwork(NetworkId& networkIdOut,
                                IOptimizedNetworkPtr inNetwork,
                                std::string& errorMessage)
{
    INetworkProperties networkProperties(
            false, MemorySource::Undefined, MemorySource::Undefined);
    return LoadNetwork(networkIdOut, std::move(inNetwork), errorMessage, networkProperties);
}

Status RuntimeImpl::LoadNetwork(NetworkId& networkIdOut,
                                IOptimizedNetworkPtr inNetwork,
                                std::string& errorMessage,
                                const INetworkProperties& networkProperties)
{
    // Register the profiler
    auto profiler = inNetwork->GetProfiler();
    ProfilerManager::GetInstance().RegisterProfiler(profiler.get());

    IOptimizedNetwork* rawNetwork = inNetwork.release();

    networkIdOut = GenerateNetworkId();

    for (auto&& context : m_BackendContexts)
    {
        context.second->BeforeLoadNetwork(networkIdOut);
    }

    unique_ptr<LoadedNetwork> loadedNetwork = LoadedNetwork::MakeLoadedNetwork(
        std::unique_ptr<IOptimizedNetwork>(rawNetwork),
        errorMessage,
        networkProperties,
        m_ProfilingService);

    if (!loadedNetwork)
    {
        return Status::Failure;
    }

    {
        std::lock_guard<std::mutex> lockGuard(m_Mutex);

        // Stores the network
        m_LoadedNetworks[networkIdOut] = std::move(loadedNetwork);
    }

    for (auto&& context : m_BackendContexts)
    {
        context.second->AfterLoadNetwork(networkIdOut);
    }

    if (m_ProfilingService.IsProfilingEnabled())
    {
        m_ProfilingService.IncrementCounterValue(armnn::profiling::NETWORK_LOADS);
    }

    return Status::Success;
}

Status RuntimeImpl::UnloadNetwork(NetworkId networkId)
{
    bool unloadOk = true;
    for (auto&& context : m_BackendContexts)
    {
        unloadOk &= context.second->BeforeUnloadNetwork(networkId);
    }

    if (!unloadOk)
    {
        ARMNN_LOG(warning) << "RuntimeImpl::UnloadNetwork(): failed to unload "
                              "network with ID:" << networkId << " because BeforeUnloadNetwork failed";
        return Status::Failure;
    }

    std::unique_ptr<profiling::TimelineUtilityMethods> timelineUtils =
            profiling::TimelineUtilityMethods::GetTimelineUtils(m_ProfilingService);
    {
        std::lock_guard<std::mutex> lockGuard(m_Mutex);

        // If timeline recording is on mark the Network end of life
        if (timelineUtils)
        {
            auto search = m_LoadedNetworks.find(networkId);
            if (search != m_LoadedNetworks.end())
            {
                profiling::ProfilingGuid networkGuid = search->second->GetNetworkGuid();
                timelineUtils->RecordEvent(networkGuid,
                                           profiling::LabelsAndEventClasses::ARMNN_PROFILING_EOL_EVENT_CLASS);
            }
        }

        if (m_LoadedNetworks.erase(networkId) == 0)
        {
            ARMNN_LOG(warning) << "WARNING: RuntimeImpl::UnloadNetwork(): " << networkId << " not found!";
            return Status::Failure;
        }

        if (m_ProfilingService.IsProfilingEnabled())
        {
            m_ProfilingService.IncrementCounterValue(armnn::profiling::NETWORK_UNLOADS);
        }
    }

    for (auto&& context : m_BackendContexts)
    {
        context.second->AfterUnloadNetwork(networkId);
    }

    // Unregister the profiler
    ProfilerManager::GetInstance().RegisterProfiler(nullptr);

    ARMNN_LOG(debug) << "RuntimeImpl::UnloadNetwork(): Unloaded network with ID: " << networkId;
    return Status::Success;
}

const std::shared_ptr<IProfiler> RuntimeImpl::GetProfiler(NetworkId networkId) const
{
    auto it = m_LoadedNetworks.find(networkId);
    if (it != m_LoadedNetworks.end())
    {
        auto& loadedNetwork = it->second;
        return loadedNetwork->GetProfiler();
    }

    return nullptr;
}

void RuntimeImpl::ReportStructure() // armnn::profiling::IProfilingService& profilingService as param
{
    // No-op for the time being, but this may be useful in future to have the profilingService available
    // if (profilingService.IsProfilingEnabled()){}

    LoadedNetworks::iterator it = m_LoadedNetworks.begin();
    while (it != m_LoadedNetworks.end())
    {
        auto& loadedNetwork = it->second;
        loadedNetwork->SendNetworkStructure();
        // Increment the Iterator to point to next entry
        it++;
    }
}

RuntimeImpl::RuntimeImpl(const IRuntime::CreationOptions& options)
    : m_NetworkIdCounter(0),
      m_ProfilingService(*this)
{
    const auto start_time = armnn::GetTimeNow();
    ARMNN_LOG(info) << "ArmNN v" << ARMNN_VERSION;
    if ( options.m_ProfilingOptions.m_TimelineEnabled && !options.m_ProfilingOptions.m_EnableProfiling )
    {
        throw RuntimeException(
                "It is not possible to enable timeline reporting without profiling being enabled");
    }

    // Load any available/compatible dynamic backend before the runtime
    // goes through the backend registry
    LoadDynamicBackends(options.m_DynamicBackendsPath);

    BackendIdSet supportedBackends;
    for (const auto& id : BackendRegistryInstance().GetBackendIds())
    {
        // Store backend contexts for the supported ones
        try {
            auto factoryFun = BackendRegistryInstance().GetFactory(id);
            ARMNN_ASSERT(factoryFun != nullptr);
            auto backend = factoryFun();
            ARMNN_ASSERT(backend != nullptr);
            ARMNN_ASSERT(backend.get() != nullptr);

            auto customAllocatorMapIterator = options.m_CustomAllocatorMap.find(id);
            if (customAllocatorMapIterator != options.m_CustomAllocatorMap.end() &&
                customAllocatorMapIterator->second == nullptr)
            {
                // We need to manually clean up  the dynamic backends before throwing an exception.
                DynamicBackendUtils::DeregisterDynamicBackends(m_DeviceSpec.GetDynamicBackends());
                m_DeviceSpec.ClearDynamicBackends();
                throw armnn::Exception("Allocator associated with id " + id.Get() + " is null");
            }

            // If the runtime is created in protected mode only add backends that support this mode
            if (options.m_ProtectedMode)
            {
                // check if backend supports ProtectedMode
                using BackendCapability = BackendOptions::BackendOption;
                BackendCapability protectedContentCapability {"ProtectedContentAllocation", true};
                if (!HasCapability(protectedContentCapability, id))
                {
                    // Protected Content Allocation is not supported by the backend
                    // backend should not be registered
                    ARMNN_LOG(warning) << "Backend "
                                       << id
                                       << " is not registered as does not support protected content allocation.";
                    continue;
                }
                // The user is responsible to provide a custom memory allocator which allows to allocate
                // protected memory
                if (customAllocatorMapIterator != options.m_CustomAllocatorMap.end())
                {
                    std::string err;
                    if (customAllocatorMapIterator->second->GetMemorySourceType()
                        == armnn::MemorySource::DmaBufProtected)
                    {
                        if (!backend->UseCustomMemoryAllocator(customAllocatorMapIterator->second, err))
                        {
                            ARMNN_LOG(error) << "The backend "
                                             << id
                                             << " reported an error when entering protected mode. Backend won't be"
                                             << " used. ErrorMsg: " << err;
                            continue;
                        }
                        // No errors so register the Custom Allocator with the BackendRegistry
                        BackendRegistryInstance().RegisterAllocator(id, customAllocatorMapIterator->second);
                    }
                    else
                    {
                        ARMNN_LOG(error) << "The CustomAllocator provided with the runtime options doesn't support "
                                     "protected memory. Protected mode can't be activated. The backend "
                                     << id
                                     << " is not going to be used. MemorySource must be MemorySource::DmaBufProtected";
                        continue;
                    }
                }
                else
                {
                    ARMNN_LOG(error) << "Protected mode can't be activated for backend: "
                                     << id
                                     << " no custom allocator was provided to the runtime options.";
                    continue;
                }
            }
            else
            {
                // If a custom memory allocator is provided make the backend use that instead of the default
                if (customAllocatorMapIterator != options.m_CustomAllocatorMap.end())
                {
                    std::string err;
                    if (!backend->UseCustomMemoryAllocator(customAllocatorMapIterator->second, err))
                    {
                        ARMNN_LOG(error) << "The backend "
                                         << id
                                         << " reported an error when trying to use the provided custom allocator."
                                            " Backend won't be used."
                                         << " ErrorMsg: " << err;
                        continue;
                    }
                    // No errors so register the Custom Allocator with the BackendRegistry
                    BackendRegistryInstance().RegisterAllocator(id, customAllocatorMapIterator->second);
                }
            }

            // check if custom memory optimizer strategy map is set
            if (!options.m_MemoryOptimizerStrategyMap.empty())
            {
                auto customMemoryOptimizerStrategyMapIterator = options.m_MemoryOptimizerStrategyMap.find(id);
                // if a memory optimizer strategy is provided make the backend use that instead of the default
                if (customMemoryOptimizerStrategyMapIterator != options.m_MemoryOptimizerStrategyMap.end())
                {
                    // no errors.. register the memory optimizer strategy with the BackendRegistry
                    BackendRegistryInstance().RegisterMemoryOptimizerStrategy(
                        id, customMemoryOptimizerStrategyMapIterator->second);

                    ARMNN_LOG(info) << "MemoryOptimizerStrategy  "
                                    << customMemoryOptimizerStrategyMapIterator->second->GetName()
                                    << " set for the backend " << id << ".";
                }
            }
            else
            {
                // check if to use one of the existing memory optimizer strategies is set
                std::string memoryOptimizerStrategyName = "";
                ParseOptions(options.m_BackendOptions, id, [&](std::string name, const BackendOptions::Var& value)
                {
                    if (name == "MemoryOptimizerStrategy")
                    {
                        memoryOptimizerStrategyName = ParseStringBackendOption(value, "");
                    }
                });
                if (memoryOptimizerStrategyName != "")
                {
                    std::shared_ptr<IMemoryOptimizerStrategy> strategy =
                            GetMemoryOptimizerStrategy(memoryOptimizerStrategyName);

                    if (!strategy)
                    {
                        ARMNN_LOG(warning) << "MemoryOptimizerStrategy: " << memoryOptimizerStrategyName
                                           << " was not found.";
                    }
                    else
                    {
                        using BackendCapability = BackendOptions::BackendOption;
                        auto strategyType = GetMemBlockStrategyTypeName(strategy->GetMemBlockStrategyType());
                        BackendCapability memOptimizeStrategyCapability {strategyType, true};
                        if (HasCapability(memOptimizeStrategyCapability, id))
                        {
                            BackendRegistryInstance().RegisterMemoryOptimizerStrategy(id, strategy);

                            ARMNN_LOG(info) << "MemoryOptimizerStrategy: "
                                            << memoryOptimizerStrategyName << " set for the backend " << id << ".";
                        }
                        else
                        {
                            ARMNN_LOG(warning) << "Backend "
                                               << id
                                               << " does not have multi-axis packing capability and cannot support"
                                               << "MemoryOptimizerStrategy: " << memoryOptimizerStrategyName << ".";
                        }
                    }
                }
            }

            auto context = backend->CreateBackendContext(options);

            // backends are allowed to return nullptrs if they
            // don't wish to create a backend specific context
            if (context)
            {
                m_BackendContexts.emplace(std::make_pair(id, std::move(context)));
            }
            supportedBackends.emplace(id);

            unique_ptr<armnn::profiling::IBackendProfiling> profilingIface =
                std::make_unique<armnn::profiling::BackendProfiling>(armnn::profiling::BackendProfiling(
                    options, m_ProfilingService, id));

            // Backends may also provide a profiling context. Ask for it now.
            auto profilingContext = backend->CreateBackendProfilingContext(options, profilingIface);
            // Backends that don't support profiling will return a null profiling context.
            if (profilingContext)
            {
                // Pass the context onto the profiling service.
                m_ProfilingService.AddBackendProfilingContext(id, profilingContext);
            }
        }
        catch (const BackendUnavailableException&)
        {
            // Ignore backends which are unavailable
        }
    }

    BackendRegistryInstance().SetProfilingService(m_ProfilingService);
    // pass configuration info to the profiling service
    m_ProfilingService.ConfigureProfilingService(options.m_ProfilingOptions);
    if (options.m_ProfilingOptions.m_EnableProfiling)
    {
        // try to wait for the profiling service to initialise
        m_ProfilingService.WaitForProfilingServiceActivation(3000);
    }

    m_DeviceSpec.AddSupportedBackends(supportedBackends);

    ARMNN_LOG(info) << "Initialization time: " << std::setprecision(2)
                    << std::fixed << armnn::GetTimeDuration(start_time).count() << " ms.";
}

RuntimeImpl::~RuntimeImpl()
{
    const auto startTime = armnn::GetTimeNow();
    std::vector<int> networkIDs;
    try
    {
        // Coverity fix: The following code may throw an exception of type std::length_error.
        std::transform(m_LoadedNetworks.begin(), m_LoadedNetworks.end(),
                       std::back_inserter(networkIDs),
                       [](const auto &pair) { return pair.first; });
    }
    catch (const std::exception& e)
    {
        // Coverity fix: BOOST_LOG_TRIVIAL (typically used to report errors) may throw an
        // exception of type std::length_error.
        // Using stderr instead in this context as there is no point in nesting try-catch blocks here.
        std::cerr << "WARNING: An error has occurred when getting the IDs of the networks to unload: " << e.what()
                  << "\nSome of the loaded networks may not be unloaded" << std::endl;
    }
    // We then proceed to unload all the networks which IDs have been appended to the list
    // up to the point the exception was thrown (if any).

    for (auto networkID : networkIDs)
    {
        try
        {
            // Coverity fix: UnloadNetwork() may throw an exception of type std::length_error,
            // boost::log::v2s_mt_posix::odr_violation or boost::log::v2s_mt_posix::system_error
            UnloadNetwork(networkID);
        }
        catch (const std::exception& e)
        {
            // Coverity fix: BOOST_LOG_TRIVIAL (typically used to report errors) may throw an
            // exception of type std::length_error.
            // Using stderr instead in this context as there is no point in nesting try-catch blocks here.
            std::cerr << "WARNING: An error has occurred when unloading network " << networkID << ": " << e.what()
                      << std::endl;
        }
    }

    // Clear all dynamic backends.
    DynamicBackendUtils::DeregisterDynamicBackends(m_DeviceSpec.GetDynamicBackends());
    m_DeviceSpec.ClearDynamicBackends();
    m_BackendContexts.clear();

    BackendRegistryInstance().SetProfilingService(armnn::EmptyOptional());
    ARMNN_LOG(info) << "Shutdown time: " << std::setprecision(2)
                    << std::fixed << armnn::GetTimeDuration(startTime).count() << " ms.";
}

LoadedNetwork* RuntimeImpl::GetLoadedNetworkPtr(NetworkId networkId) const
{
    std::lock_guard<std::mutex> lockGuard(m_Mutex);
    return m_LoadedNetworks.at(networkId).get();
}

TensorInfo RuntimeImpl::GetInputTensorInfo(NetworkId networkId, LayerBindingId layerId) const
{
    return GetLoadedNetworkPtr(networkId)->GetInputTensorInfo(layerId);
}

TensorInfo RuntimeImpl::GetOutputTensorInfo(NetworkId networkId, LayerBindingId layerId) const
{
    return GetLoadedNetworkPtr(networkId)->GetOutputTensorInfo(layerId);
}

std::vector<ImportedInputId> RuntimeImpl::ImportInputs(NetworkId networkId, const InputTensors& inputTensors,
                                                       MemorySource forceImportMemorySource)
{
    return GetLoadedNetworkPtr(networkId)->ImportInputs(inputTensors, forceImportMemorySource);
}

std::vector<ImportedOutputId> RuntimeImpl::ImportOutputs(NetworkId networkId, const OutputTensors& outputTensors,
                                                         MemorySource forceImportMemorySource)
{
    return GetLoadedNetworkPtr(networkId)->ImportOutputs(outputTensors, forceImportMemorySource);
}

void RuntimeImpl::ClearImportedInputs(NetworkId networkId, const std::vector<ImportedInputId> inputIds)
{
    return GetLoadedNetworkPtr(networkId)->ClearImportedInputs(inputIds);
}
void RuntimeImpl::ClearImportedOutputs(NetworkId networkId, const std::vector<ImportedOutputId> outputIds)
{
    return GetLoadedNetworkPtr(networkId)->ClearImportedOutputs(outputIds);
}

Status RuntimeImpl::EnqueueWorkload(NetworkId networkId,
                                const InputTensors& inputTensors,
                                const OutputTensors& outputTensors,
                                std::vector<ImportedInputId> preImportedInputIds,
                                std::vector<ImportedOutputId> preImportedOutputIds)
{
    const auto startTime = armnn::GetTimeNow();

    LoadedNetwork* loadedNetwork = GetLoadedNetworkPtr(networkId);

    if (!loadedNetwork)
    {
        ARMNN_LOG(error) << "A Network with an id of " << networkId << " does not exist.";
        return Status::Failure;
    }
    if (loadedNetwork->IsAsyncEnabled())
    {
        ARMNN_LOG(error) << "Network " << networkId << " is async enabled.";
        return Status::Failure;
    }
    ProfilerManager::GetInstance().RegisterProfiler(loadedNetwork->GetProfiler().get());

    ARMNN_SCOPED_PROFILING_EVENT(Compute::Undefined, "EnqueueWorkload");

    static thread_local NetworkId lastId = networkId;
    if (lastId != networkId)
    {
        LoadedNetworkFuncSafe(lastId, [](LoadedNetwork* network)
            {
                network->FreeWorkingMemory();
            });
    }
    lastId=networkId;

    auto status = loadedNetwork->EnqueueWorkload(inputTensors, outputTensors,
                                                 preImportedInputIds, preImportedOutputIds);


    // Check if we imported, if not there's no need to call the After EnqueueWorkload events
    if (!preImportedInputIds.empty() || !preImportedOutputIds.empty())
    {
        // Call After EnqueueWorkload events
        for (auto&& context : m_BackendContexts)
        {
            context.second->AfterEnqueueWorkload(networkId);
        }
    }
    ARMNN_LOG(info) << "Execution time: " << std::setprecision(2)
                    << std::fixed << armnn::GetTimeDuration(startTime).count() << " ms.";
    return status;
}

Status RuntimeImpl::Execute(IWorkingMemHandle& iWorkingMemHandle,
                            const InputTensors& inputTensors,
                            const OutputTensors& outputTensors,
                            std::vector<ImportedInputId> preImportedInputs,
                            std::vector<ImportedOutputId> preImportedOutputs)
{
    const auto startTime = armnn::GetTimeNow();

    NetworkId networkId = iWorkingMemHandle.GetNetworkId();
    LoadedNetwork* loadedNetwork = GetLoadedNetworkPtr(networkId);

    if (!loadedNetwork)
    {
        ARMNN_LOG(error) << "A Network with an id of " << networkId << " does not exist.";
        return Status::Failure;
    }
    if (!loadedNetwork->IsAsyncEnabled())
    {
        ARMNN_LOG(error) << "Attempting execute " << networkId << " when it is not async enabled.";
        return Status::Failure;
    }
    ProfilerManager::GetInstance().RegisterProfiler(loadedNetwork->GetProfiler().get());

    ARMNN_SCOPED_PROFILING_EVENT(Compute::Undefined, "Execute");

    auto status = loadedNetwork->Execute(inputTensors,
                                         outputTensors,
                                         iWorkingMemHandle,
                                         preImportedInputs,
                                         preImportedOutputs);

    ARMNN_LOG(info) << "Execution time: " << std::setprecision(2)
                    << std::fixed << armnn::GetTimeDuration(startTime).count() << " ms.";

    return status;
}

/// Create a new unique WorkingMemHandle object. Create multiple handles if you wish to have
/// overlapped Execution by calling this function from different threads.
std::unique_ptr<IWorkingMemHandle> RuntimeImpl::CreateWorkingMemHandle(NetworkId networkId)
{
    LoadedNetwork* loadedNetwork = GetLoadedNetworkPtr(networkId);

    if (!loadedNetwork)
    {
        ARMNN_LOG(error) << "A Network with an id of " << networkId << " does not exist.";
        return nullptr;
    }
    if (!loadedNetwork->IsAsyncEnabled())
    {
        ARMNN_LOG(error) << "Network " << networkId << " is not async enabled.";
        return nullptr;
    }
    ProfilerManager::GetInstance().RegisterProfiler(loadedNetwork->GetProfiler().get());

    ARMNN_SCOPED_PROFILING_EVENT(Compute::Undefined, "CreateWorkingMemHandle");

    static thread_local NetworkId lastId = networkId;
    if (lastId != networkId)
    {
        LoadedNetworkFuncSafe(lastId, [](LoadedNetwork* network)
        {
            network->FreeWorkingMemory();
        });
    }
    lastId=networkId;

    return loadedNetwork->CreateWorkingMemHandle(networkId);
}

void RuntimeImpl::RegisterDebugCallback(NetworkId networkId, const DebugCallbackFunction& func)
{
    LoadedNetwork* loadedNetwork = GetLoadedNetworkPtr(networkId);
    loadedNetwork->RegisterDebugCallback(func);
}

void RuntimeImpl::LoadDynamicBackends(const std::string& overrideBackendPath)
{
    // Get the paths where to load the dynamic backends from
    std::vector<std::string> backendPaths = DynamicBackendUtils::GetBackendPaths(overrideBackendPath);

    // Get the shared objects to try to load as dynamic backends
    std::vector<std::string> sharedObjects = DynamicBackendUtils::GetSharedObjects(backendPaths);

    // Create a list of dynamic backends
    m_DynamicBackends = DynamicBackendUtils::CreateDynamicBackends(sharedObjects);

    // Register the dynamic backends in the backend registry
    BackendIdSet registeredBackendIds = DynamicBackendUtils::RegisterDynamicBackends(m_DynamicBackends);

    // Add the registered dynamic backend ids to the list of supported backends
    m_DeviceSpec.AddSupportedBackends(registeredBackendIds, true);
}

} // namespace armnn
