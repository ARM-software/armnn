//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//
#include "Runtime.hpp"

#include <armnn/Version.hpp>
#include <backendsCommon/BackendRegistry.hpp>
#include <backendsCommon/BackendContextRegistry.hpp>

#include <iostream>

#include <boost/log/trivial.hpp>
#include <boost/polymorphic_cast.hpp>

using namespace armnn;
using namespace std;

namespace armnn
{

IRuntime* IRuntime::CreateRaw(const CreationOptions& options)
{
    return new Runtime(options);
}

IRuntimePtr IRuntime::Create(const CreationOptions& options)
{
    return IRuntimePtr(CreateRaw(options), &IRuntime::Destroy);
}

void IRuntime::Destroy(IRuntime* runtime)
{
    delete boost::polymorphic_downcast<Runtime*>(runtime);
}

int Runtime::GenerateNetworkId()
{
    return m_NetworkIdCounter++;
}

Status Runtime::LoadNetwork(NetworkId& networkIdOut, IOptimizedNetworkPtr inNetwork)
{
    std::string ignoredErrorMessage;
    return LoadNetwork(networkIdOut, std::move(inNetwork), ignoredErrorMessage);
}

Status Runtime::LoadNetwork(NetworkId& networkIdOut,
                            IOptimizedNetworkPtr inNetwork,
                            std::string & errorMessage)
{
    IOptimizedNetwork* rawNetwork = inNetwork.release();
    unique_ptr<LoadedNetwork> loadedNetwork = LoadedNetwork::MakeLoadedNetwork(
        std::unique_ptr<OptimizedNetwork>(boost::polymorphic_downcast<OptimizedNetwork*>(rawNetwork)),
        m_Options,
        errorMessage);

    if (!loadedNetwork)
    {
        return Status::Failure;
    }

    networkIdOut = GenerateNetworkId();

    {
        std::lock_guard<std::mutex> lockGuard(m_Mutex);

        // Stores the network
        m_LoadedNetworks[networkIdOut] = std::move(loadedNetwork);
    }

    return Status::Success;
}

Status Runtime::UnloadNetwork(NetworkId networkId)
{
    {
        std::lock_guard<std::mutex> lockGuard(m_Mutex);

        if (m_LoadedNetworks.erase(networkId) == 0)
        {
            BOOST_LOG_TRIVIAL(warning) << "WARNING: Runtime::UnloadNetwork(): " << networkId << " not found!";
            return Status::Failure;
        }
    }

    BOOST_LOG_TRIVIAL(debug) << "Runtime::UnloadNetwork(): Unloaded network with ID: " << networkId;
    return Status::Success;
}

const std::shared_ptr<IProfiler> Runtime::GetProfiler(NetworkId networkId) const
{
    auto it = m_LoadedNetworks.find(networkId);
    if (it != m_LoadedNetworks.end())
    {
        auto& loadedNetwork = it->second;
        return loadedNetwork->GetProfiler();
    }

    return nullptr;
}

Runtime::Runtime(const CreationOptions& options)
    : m_Options{options}
    , m_NetworkIdCounter(0)
    , m_DeviceSpec{BackendRegistryInstance().GetBackendIds()}
{
    BOOST_LOG_TRIVIAL(info) << "ArmNN v" << ARMNN_VERSION << "\n";

    for (const auto& id : BackendContextRegistryInstance().GetBackendIds())
    {
        // Store backend contexts for the supported ones
        if (m_DeviceSpec.GetSupportedBackends().count(id) > 0)
        {
            // Don't throw an exception, rather return a dummy factory if not
            // found.
            auto factoryFun = BackendContextRegistryInstance().GetFactory(
                id, [](const CreationOptions&) { return IBackendContextUniquePtr(); }
            );

            m_BackendContexts.emplace(std::make_pair(id, factoryFun(options)));
        }
    }
}

Runtime::~Runtime()
{
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
}

LoadedNetwork* Runtime::GetLoadedNetworkPtr(NetworkId networkId) const
{
    std::lock_guard<std::mutex> lockGuard(m_Mutex);
    return m_LoadedNetworks.at(networkId).get();
}

TensorInfo Runtime::GetInputTensorInfo(NetworkId networkId, LayerBindingId layerId) const
{
    return GetLoadedNetworkPtr(networkId)->GetInputTensorInfo(layerId);
}

TensorInfo Runtime::GetOutputTensorInfo(NetworkId networkId, LayerBindingId layerId) const
{
    return GetLoadedNetworkPtr(networkId)->GetOutputTensorInfo(layerId);
}


Status Runtime::EnqueueWorkload(NetworkId networkId,
                                const InputTensors& inputTensors,
                                const OutputTensors& outputTensors)
{
    LoadedNetwork* loadedNetwork = GetLoadedNetworkPtr(networkId);

    static thread_local NetworkId lastId = networkId;
    if (lastId != networkId)
    {
        LoadedNetworkFuncSafe(lastId, [](LoadedNetwork* network)
            {
                network->FreeWorkingMemory();
            });
    }
    lastId=networkId;

    return loadedNetwork->EnqueueWorkload(inputTensors, outputTensors);
}

}
