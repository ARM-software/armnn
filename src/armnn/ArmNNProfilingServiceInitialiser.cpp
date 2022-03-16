//
// Copyright © 2022 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "ArmNNProfilingServiceInitialiser.hpp"

#include <armnn/BackendRegistry.hpp>
#include <armnn/profiling/ArmNNProfiling.hpp>
#include <armnn/utility/Assert.hpp>

#include <common/include/Counter.hpp>

namespace armnn
{

void ArmNNProfilingServiceInitialiser::InitialiseProfilingService(arm::pipe::IProfilingService& profilingService)
{
    uint16_t ZERO = 0;
    double ONE = 1.0;
    std::string ArmNN_Runtime("ArmNN_Runtime");
    // Register a category for the basic runtime counters
    if (!profilingService.IsCategoryRegistered(ArmNN_Runtime))
    {
        profilingService.GetCounterRegistry().RegisterCategory(ArmNN_Runtime);
    }

    std::string networks("networks");
    std::string networkLoads("Network loads");
    // Register a counter for the number of Network loads
    if (!profilingService.IsCounterRegistered(networkLoads))
    {
        const arm::pipe::Counter* loadedNetworksCounter =
            profilingService.GetCounterRegistry().RegisterCounter(armnn::profiling::BACKEND_ID.Get(),
                                                                  arm::pipe::NETWORK_LOADS,
                                                                  ArmNN_Runtime,
                                                                  ZERO,
                                                                  ZERO,
                                                                  ONE,
                                                                  networkLoads,
                                                                  "The number of networks loaded at runtime",
                                                                  networks);
        ARMNN_ASSERT(loadedNetworksCounter);
        profilingService.InitializeCounterValue(loadedNetworksCounter->m_Uid);
    }
    // Register a counter for the number of unloaded networks
    std::string networkUnloads("Network unloads");
    if (!profilingService.IsCounterRegistered(networkUnloads))
    {
        const arm::pipe::Counter* unloadedNetworksCounter =
            profilingService.GetCounterRegistry().RegisterCounter(armnn::profiling::BACKEND_ID.Get(),
                                                                  arm::pipe::NETWORK_UNLOADS,
                                                                  ArmNN_Runtime,
                                                                  ZERO,
                                                                  ZERO,
                                                                  ONE,
                                                                  networkUnloads,
                                                                  "The number of networks unloaded at runtime",
                                                                  networks);
        ARMNN_ASSERT(unloadedNetworksCounter);
        profilingService.InitializeCounterValue(unloadedNetworksCounter->m_Uid);
    }
    std::string backends("backends");
    // Register a counter for the number of registered backends
    std::string backendsRegistered("Backends registered");
    if (!profilingService.IsCounterRegistered(backendsRegistered))
    {
        const arm::pipe::Counter* registeredBackendsCounter =
            profilingService.GetCounterRegistry().RegisterCounter(armnn::profiling::BACKEND_ID.Get(),
                                                                  arm::pipe::REGISTERED_BACKENDS,
                                                                  ArmNN_Runtime,
                                                                  ZERO,
                                                                  ZERO,
                                                                  ONE,
                                                                  backendsRegistered,
                                                                  "The number of registered backends",
                                                                  backends);
        ARMNN_ASSERT(registeredBackendsCounter);
        profilingService.InitializeCounterValue(registeredBackendsCounter->m_Uid);

        // Due to backends being registered before the profiling service becomes active,
        // we need to set the counter to the correct value here
        profilingService.SetCounterValue(arm::pipe::REGISTERED_BACKENDS, static_cast<uint32_t>(
            armnn::BackendRegistryInstance().Size()));
    }
    // Register a counter for the number of registered backends
    std::string backendsUnregistered("Backends unregistered");
    if (!profilingService.IsCounterRegistered(backendsUnregistered))
    {
        const arm::pipe::Counter* unregisteredBackendsCounter =
            profilingService.GetCounterRegistry().RegisterCounter(armnn::profiling::BACKEND_ID.Get(),
                                                                  arm::pipe::UNREGISTERED_BACKENDS,
                                                                  ArmNN_Runtime,
                                                                  ZERO,
                                                                  ZERO,
                                                                  ONE,
                                                                  backendsUnregistered,
                                                                  "The number of unregistered backends",
                                                                  backends);
        ARMNN_ASSERT(unregisteredBackendsCounter);
        profilingService.InitializeCounterValue(unregisteredBackendsCounter->m_Uid);
    }
    // Register a counter for the number of inferences run
    std::string inferences("inferences");
    std::string inferencesRun("Inferences run");
    if (!profilingService.IsCounterRegistered(inferencesRun))
    {
        const arm::pipe::Counter* inferencesRunCounter =
            profilingService.GetCounterRegistry().RegisterCounter(armnn::profiling::BACKEND_ID.Get(),
                                                                 arm::pipe::INFERENCES_RUN,
                                                                 ArmNN_Runtime,
                                                                 ZERO,
                                                                 ZERO,
                                                                 ONE,
                                                                 inferencesRun,
                                                                 "The number of inferences run",
                                                                 inferences);
        ARMNN_ASSERT(inferencesRunCounter);
        profilingService.InitializeCounterValue(inferencesRunCounter->m_Uid);
    }
}

} // namespace armnn
