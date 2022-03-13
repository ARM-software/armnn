//
// Copyright © 2020 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "RegisterBackendCounters.hpp"

namespace arm
{

namespace pipe
{

void RegisterBackendCounters::RegisterCategory(const std::string& categoryName)
{
     m_CounterDirectory.RegisterCategory(categoryName);
}

uint16_t RegisterBackendCounters::RegisterDevice(const std::string& deviceName,
                                                 uint16_t cores,
                                                 const arm::pipe::Optional<std::string>& parentCategoryName)
{
    const Device* devicePtr = m_CounterDirectory.RegisterDevice(deviceName, cores, parentCategoryName);
    return devicePtr->m_Uid;
}

uint16_t RegisterBackendCounters::RegisterCounterSet(const std::string& counterSetName,
                                                     uint16_t count,
                                                     const arm::pipe::Optional<std::string>& parentCategoryName)
{
    const CounterSet* counterSetPtr = m_CounterDirectory.RegisterCounterSet(counterSetName, count, parentCategoryName);
    return counterSetPtr->m_Uid;
}

uint16_t RegisterBackendCounters::RegisterCounter(const uint16_t uid,
                                                  const std::string& parentCategoryName,
                                                  uint16_t counterClass,
                                                  uint16_t interpolation,
                                                  double multiplier,
                                                  const std::string& name,
                                                  const std::string& description,
                                                  const arm::pipe::Optional<std::string>& units,
                                                  const arm::pipe::Optional<uint16_t>& numberOfCores,
                                                  const arm::pipe::Optional<uint16_t>& deviceUid,
                                                  const arm::pipe::Optional<uint16_t>& counterSetUid)
{
    ++m_CurrentMaxGlobalCounterID;
    const Counter* counterPtr = m_CounterDirectory.RegisterCounter(m_BackendId,
                                                                   m_CurrentMaxGlobalCounterID,
                                                                   parentCategoryName,
                                                                   counterClass,
                                                                   interpolation,
                                                                   multiplier,
                                                                   name,
                                                                   description,
                                                                   units,
                                                                   numberOfCores,
                                                                   deviceUid,
                                                                   counterSetUid);
    m_CurrentMaxGlobalCounterID = counterPtr->m_MaxCounterUid;
    // register mappings
    IRegisterCounterMapping& counterIdMap = m_ProfilingService.GetCounterMappingRegistry();
    uint16_t globalCounterId = counterPtr->m_Uid;
    if (globalCounterId == counterPtr->m_MaxCounterUid)
    {
        counterIdMap.RegisterMapping(globalCounterId, uid, m_BackendId);
    }
    else
    {
        uint16_t backendCounterId = uid;
        while (globalCounterId <= counterPtr->m_MaxCounterUid)
        {
            // register mapping
            // globalCounterId -> backendCounterId, m_BackendId
            counterIdMap.RegisterMapping(globalCounterId, backendCounterId, m_BackendId);
            ++globalCounterId;
            ++backendCounterId;
        }
    }
    return m_CurrentMaxGlobalCounterID;
}

} // namespace pipe

} // namespace arm
