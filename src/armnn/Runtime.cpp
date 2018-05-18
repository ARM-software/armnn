//
// Copyright © 2017 Arm Ltd. All rights reserved.
// See LICENSE file in the project root for full license information.
//
#include "Runtime.hpp"

#include "armnn/Version.hpp"

#include <iostream>

#ifdef ARMCOMPUTECL_ENABLED
#include <arm_compute/core/CL/OpenCL.h>
#include <arm_compute/core/CL/CLKernelLibrary.h>
#include <arm_compute/runtime/CL/CLScheduler.h>
#endif

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
    IOptimizedNetwork* rawNetwork = inNetwork.release();
    unique_ptr<LoadedNetwork> loadedNetwork = LoadedNetwork::MakeLoadedNetwork(
        std::unique_ptr<OptimizedNetwork>(boost::polymorphic_downcast<OptimizedNetwork*>(rawNetwork)),
        m_UseCpuRefAsFallback);

    if (!loadedNetwork)
    {
        return Status::Failure;
    }

    std::lock_guard<std::mutex> lockGuard(m_Mutex);

    networkIdOut = GenerateNetworkId();

    // store the network
    m_LoadedNetworks[networkIdOut] = std::move(loadedNetwork);

    return Status::Success;
}

Status Runtime::UnloadNetwork(NetworkId networkId)
{
#ifdef ARMCOMPUTECL_ENABLED
    if (arm_compute::CLScheduler::get().context()() != NULL)
    {
        // wait for all queued CL requests to finish before unloading the network they may be using
        try
        {
            // Coverity fix: arm_compute::CLScheduler::sync() may throw an exception of type cl::Error.
            arm_compute::CLScheduler::get().sync();
        }
        catch (const cl::Error&)
        {
            BOOST_LOG_TRIVIAL(warning) << "WARNING: Runtime::UnloadNetwork(): an error occurred while waiting for "
                                          "the queued CL requests to finish";
            return Status::Failure;
        }
    }
#endif
    std::lock_guard<std::mutex> lockGuard(m_Mutex);

    if (m_LoadedNetworks.erase(networkId) == 0)
    {
        BOOST_LOG_TRIVIAL(warning) << "WARNING: Runtime::UnloadNetwork(): " << networkId << " not found!";
        return Status::Failure;
    }
#ifdef ARMCOMPUTECL_ENABLED
    if (arm_compute::CLScheduler::get().context()() != NULL && m_LoadedNetworks.empty())
    {
        // There are no loaded networks left, so clear the CL cache to free up memory
        m_ClContextControl.ClearClCache();
    }
#endif
    BOOST_LOG_TRIVIAL(debug) << "Runtime::UnloadNetwork(): Unloaded network with ID: " << networkId;
    return Status::Success;
}

Runtime::Runtime(const CreationOptions& options)
    : m_ClContextControl(options.m_ClTunedParameters)
    , m_NetworkIdCounter(0)
{
    BOOST_LOG_TRIVIAL(info) << "ArmNN v" << ARMNN_VERSION << "\n";
    BOOST_LOG_TRIVIAL(info) << "Using compute device: " << options.m_DefaultComputeDevice << "\n";
    m_DeviceSpec.DefaultComputeDevice = options.m_DefaultComputeDevice;

    // If useCpuRefAsFallback is false, the reference workload factory will be prevented from creating
    // operation workloads, unless the default compute device is precisely the reference backend.
    // This option is passed to the LoadedNetwork, which owns the workload factories.
    m_UseCpuRefAsFallback = options.m_DefaultComputeDevice == Compute::CpuRef || options.m_UseCpuRefAsFallback;
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
    return loadedNetwork->EnqueueWorkload(inputTensors, outputTensors);
}

}
