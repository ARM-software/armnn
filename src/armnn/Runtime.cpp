//
// Copyright © 2017 Arm Ltd. All rights reserved.
// See LICENSE file in the project root for full license information.
//
#include "Runtime.hpp"

#include "armnn/Version.hpp"

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
        m_WorkloadFactories);

    if (!loadedNetwork)
    {
        return Status::Failure;
    }

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
        arm_compute::CLScheduler::get().sync();
    }
#endif
    if (m_LoadedNetworks.erase(networkId) == 0)
    {
        BOOST_LOG_TRIVIAL(warning) << "WARNING: Runtime::UnloadNetwork(): " << networkId << " not found!";
        return Status::Failure;
    }
#ifdef ARMCOMPUTECL_ENABLED
    if (arm_compute::CLScheduler::get().context()() != NULL && m_LoadedNetworks.empty())
    {
        m_WorkloadFactories.m_GpuAcc.get()->LoadOpenClRuntime();
    }
#endif
    BOOST_LOG_TRIVIAL(debug) << "Runtime::UnloadNetwork(): Unloaded network with ID: " << networkId;
    return Status::Success;
}

Runtime::Runtime(const CreationOptions& options)
: m_NetworkIdCounter(0)
{
    BOOST_LOG_TRIVIAL(info) << "ArmNN v" << ARMNN_VERSION << "\n";
    BOOST_LOG_TRIVIAL(info) << "Using compute device: " << options.m_DefaultComputeDevice << "\n";
    m_DeviceSpec.DefaultComputeDevice = options.m_DefaultComputeDevice;

   // If useCpuRefAsFallback is false, the reference workload factory will be prevented from creating
   // operation workloads, unless the default compute device is precisely the reference backend.
    m_WorkloadFactories.m_CpuRef = make_shared<RefWorkloadFactory>(
        options.m_DefaultComputeDevice == Compute::CpuRef ? true : options.m_UseCpuRefAsFallback);
    m_WorkloadFactories.m_CpuAcc = make_shared<NeonWorkloadFactory>();
    m_WorkloadFactories.m_GpuAcc = make_shared<ClWorkloadFactory>(options.m_ClTunedParameters);

    if (options.m_DefaultComputeDevice == Compute::GpuAcc)
    {
        m_WorkloadFactories.m_GpuAcc.get()->LoadOpenClRuntime();
    }
}

Runtime::~Runtime()
{
    std::vector<int> networkIDs;
    std::transform(m_LoadedNetworks.begin(), m_LoadedNetworks.end(),
                   std::back_inserter(networkIDs),
                   [](const auto &pair) { return pair.first; });

    for (auto networkID : networkIDs)
    {
        UnloadNetwork(networkID);
    }
}

TensorInfo Runtime::GetInputTensorInfo(NetworkId networkId, LayerBindingId layerId) const
{
    LoadedNetwork* net = m_LoadedNetworks.at(networkId).get();
    return net->GetInputTensorInfo(layerId);
}

TensorInfo Runtime::GetOutputTensorInfo(NetworkId networkId, LayerBindingId layerId) const
{
    const LoadedNetwork* net = m_LoadedNetworks.at(networkId).get();
    return net->GetOutputTensorInfo(layerId);
}

Status Runtime::EnqueueWorkload(NetworkId networkId,
                                     const InputTensors& inputTensors,
                                     const OutputTensors& outputTensors)
{
    LoadedNetwork* loadedNetwork = m_LoadedNetworks.at(networkId).get();
    return loadedNetwork->EnqueueWorkload(inputTensors, outputTensors, m_WorkloadFactories);
}

}
