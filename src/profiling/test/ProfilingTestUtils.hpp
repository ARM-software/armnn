//
// Copyright © 2019 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include "ProfilingUtils.hpp"
#include "Runtime.hpp"

#include <armnn/BackendId.hpp>
#include <armnn/Optional.hpp>
#include <BufferManager.hpp>
#include <ProfilingService.hpp>

#include <common/include/ProfilingGuid.hpp>

using namespace armnn;
using namespace armnn::profiling;

const static uint32_t bodyHeaderSize = 6;

uint32_t GetStreamMetaDataPacketSize();

/// Returns a vector of CpuRef, CpuAcc or GpuAcc backends if they where registered
std::vector<BackendId> GetSuitableBackendRegistered();

inline unsigned int OffsetToNextWord(unsigned int numberOfBytes);

void VerifyTimelineHeaderBinary(const unsigned char* readableData,
                                unsigned int& offset,
                                uint32_t packetDataLength);

ProfilingGuid VerifyTimelineLabelBinaryPacketData(Optional<ProfilingGuid> guid,
                                                  const std::string& label,
                                                  const unsigned char* readableData,
                                                  unsigned int& offset);

void VerifyTimelineEventClassBinaryPacketData(ProfilingGuid guid,
                                              ProfilingGuid nameGuid,
                                              const unsigned char* readableData,
                                              unsigned int& offset);

void VerifyTimelineRelationshipBinaryPacketData(ProfilingRelationshipType relationshipType,
                                                Optional<ProfilingGuid> relationshipGuid,
                                                Optional<ProfilingGuid> headGuid,
                                                Optional<ProfilingGuid> tailGuid,
                                                Optional<ProfilingGuid> attributeGuid,
                                                const unsigned char* readableData,
                                                unsigned int& offset);

ProfilingGuid VerifyTimelineEntityBinaryPacketData(Optional<ProfilingGuid> guid,
                                                   const unsigned char* readableData,
                                                   unsigned int& offset);

ProfilingGuid VerifyTimelineEventBinaryPacket(Optional<uint64_t> timestamp,
                                              Optional<int> threadId,
                                              Optional<ProfilingGuid> eventGuid,
                                              const unsigned char* readableData,
                                              unsigned int& offset);

void VerifyPostOptimisationStructureTestImpl(armnn::BackendId backendId);

bool CompareOutput(std::vector<std::string> output, std::vector<std::string> expectedOutput);

namespace armnn
{

namespace profiling
{

class ProfilingServiceRuntimeHelper : public ProfilingService
{
public:
    ProfilingServiceRuntimeHelper(ProfilingService& profilingService)
    : m_ProfilingService(profilingService) {}
    ~ProfilingServiceRuntimeHelper() = default;

    BufferManager& GetProfilingBufferManager()
    {
        return GetBufferManager(m_ProfilingService);
    }
    armnn::profiling::ProfilingService& m_ProfilingService;

    void ForceTransitionToState(ProfilingState newState)
    {
        TransitionToState(m_ProfilingService, newState);
    }
};

} // namespace profiling

} // namespace armnn

