//
// Copyright © 2019 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include "ProfilingUtils.hpp"

#include <armnn/BackendId.hpp>
#include <armnn/Optional.hpp>
#include <armnn/Types.hpp>
#include <BufferManager.hpp>
#include <ProfilingService.hpp>

using namespace armnn;
using namespace armnn::profiling;

inline unsigned int OffsetToNextWord(unsigned int numberOfBytes);

void VerifyTimelineHeaderBinary(const unsigned char* readableData,
                                unsigned int& offset,
                                uint32_t packetDataLength);

void VerifyTimelineLabelBinaryPacketData(Optional<ProfilingGuid> guid,
                                         const std::string& label,
                                         const unsigned char* readableData,
                                         unsigned int& offset);

void VerifyTimelineEventClassBinaryPacketData(ProfilingGuid guid,
                                              const unsigned char* readableData,
                                              unsigned int& offset);

void VerifyTimelineRelationshipBinaryPacketData(ProfilingRelationshipType relationshipType,
                                                Optional<ProfilingGuid> relationshipGuid,
                                                Optional<ProfilingGuid> headGuid,
                                                Optional<ProfilingGuid> tailGuid,
                                                const unsigned char* readableData,
                                                unsigned int& offset);

void VerifyTimelineEntityBinaryPacketData(Optional<ProfilingGuid> guid,
                                          const unsigned char* readableData,
                                          unsigned int& offset);

void VerifyTimelineEventBinaryPacket(Optional<uint64_t> timestamp,
                                     Optional<std::thread::id> threadId,
                                     Optional<ProfilingGuid> eventGuid,
                                     const unsigned char* readableData,
                                     unsigned int& offset);

void VerifyPostOptimisationStructureTestImpl(armnn::BackendId backendId);

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
};

} // namespace profiling

} // namespace armnn

