//
// Copyright © 2019 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include "Runtime.hpp"

#include <client/src/BufferManager.hpp>
#include <client/src/ProfilingService.hpp>
#include <client/src/ProfilingUtils.hpp>


#include <armnn/profiling/ArmNNProfiling.hpp>

#include <common/include/Optional.hpp>
#include <common/include/ProfilingGuid.hpp>

using namespace armnn;
using namespace arm::pipe;

const static uint32_t bodyHeaderSize = 6;

uint32_t GetStreamMetaDataPacketSize();

/// Returns a vector of CpuRef, CpuAcc or GpuAcc backends if they where registered
std::vector<BackendId> GetSuitableBackendRegistered();

inline unsigned int OffsetToNextWord(unsigned int numberOfBytes);

void VerifyTimelineHeaderBinary(const unsigned char* readableData,
                                unsigned int& offset,
                                uint32_t packetDataLength);

ProfilingGuid VerifyTimelineLabelBinaryPacketData(arm::pipe::Optional<ProfilingGuid> guid,
                                                  const std::string& label,
                                                  const unsigned char* readableData,
                                                  unsigned int& offset);

void VerifyTimelineEventClassBinaryPacketData(ProfilingGuid guid,
                                              ProfilingGuid nameGuid,
                                              const unsigned char* readableData,
                                              unsigned int& offset);

void VerifyTimelineRelationshipBinaryPacketData(ProfilingRelationshipType relationshipType,
                                                arm::pipe::Optional<ProfilingGuid> relationshipGuid,
                                                arm::pipe::Optional<ProfilingGuid> headGuid,
                                                arm::pipe::Optional<ProfilingGuid> tailGuid,
                                                arm::pipe::Optional<ProfilingGuid> attributeGuid,
                                                const unsigned char* readableData,
                                                unsigned int& offset);

ProfilingGuid VerifyTimelineEntityBinaryPacketData(arm::pipe::Optional<ProfilingGuid> guid,
                                                   const unsigned char* readableData,
                                                   unsigned int& offset);

ProfilingGuid VerifyTimelineEventBinaryPacket(arm::pipe::Optional<uint64_t> timestamp,
                                              arm::pipe::Optional<int> threadId,
                                              arm::pipe::Optional<ProfilingGuid> eventGuid,
                                              const unsigned char* readableData,
                                              unsigned int& offset);

void VerifyPostOptimisationStructureTestImpl(armnn::BackendId backendId);

bool CompareOutput(std::vector<std::string> output, std::vector<std::string> expectedOutput);

namespace arm
{

namespace pipe
{

class ProfilingServiceRuntimeHelper : public ProfilingService
{
public:
    ProfilingServiceRuntimeHelper(uint16_t maxGlobalCounterId,
                                  IInitialiseProfilingService& initialiser,
                                  arm::pipe::IProfilingService& profilingService)
        : ProfilingService(maxGlobalCounterId,
                           initialiser,
                           arm::pipe::ARMNN_SOFTWARE_INFO,
                           arm::pipe::ARMNN_SOFTWARE_VERSION,
                           arm::pipe::ARMNN_HARDWARE_VERSION),
          m_ProfilingService(profilingService) {}
    ~ProfilingServiceRuntimeHelper() = default;

    BufferManager& GetProfilingBufferManager()
    {
        return GetBufferManager(static_cast<arm::pipe::ProfilingService&>(m_ProfilingService));
    }
    arm::pipe::IProfilingService& m_ProfilingService;

    void ForceTransitionToState(ProfilingState newState)
    {
        TransitionToState(static_cast<arm::pipe::ProfilingService&>(m_ProfilingService), newState);
    }
};

struct LogLevelSwapper
{
public:
    LogLevelSwapper(arm::pipe::LogSeverity severity)
    {
        // Set the new log level
        arm::pipe::ConfigureLogging(true, true, severity);
    }
    ~LogLevelSwapper()
    {
        // The default log level for unit tests is "Fatal"
        arm::pipe::ConfigureLogging(true, true, arm::pipe::LogSeverity::Fatal);
    }
};

struct StreamRedirector
{
public:
    StreamRedirector(std::ostream& stream, std::streambuf* newStreamBuffer)
        : m_Stream(stream)
        , m_BackupBuffer(m_Stream.rdbuf(newStreamBuffer))
    {}

    ~StreamRedirector() { CancelRedirect(); }

    void CancelRedirect()
    {
        // Only cancel the redirect once.
        if (m_BackupBuffer != nullptr )
        {
            m_Stream.rdbuf(m_BackupBuffer);
            m_BackupBuffer = nullptr;
        }
    }

private:
    std::ostream& m_Stream;
    std::streambuf* m_BackupBuffer;
};

} // namespace pipe

} // namespace arm
