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

void VerifyTimelineLabelBinaryPacket(Optional<ProfilingGuid> guid,
                                     const std::string& label,
                                     const unsigned char* readableData,
                                     unsigned int& offset);

void VerifyTimelineEventClassBinaryPacket(ProfilingGuid guid,
                                          const unsigned char* readableData,
                                          unsigned int& offset);

void VerifyTimelineRelationshipBinaryPacket(ProfilingRelationshipType relationshipType,
                                            Optional<ProfilingGuid> relationshipGuid,
                                            Optional<ProfilingGuid> headGuid,
                                            Optional<ProfilingGuid> tailGuid,
                                            const unsigned char* readableData,
                                            unsigned int& offset);

void VerifyTimelineEntityBinaryPacket(Optional<ProfilingGuid> guid,
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
    ProfilingServiceRuntimeHelper() = default;
    ~ProfilingServiceRuntimeHelper() = default;

    BufferManager& GetProfilingBufferManager()
    {
        return GetBufferManager(ProfilingService::Instance());
    }
};

} // namespace profiling

} // namespace armnn

