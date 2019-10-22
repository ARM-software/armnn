//
// Copyright © 2019 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include "ITimelineDecoder.h"

#include <CommandHandlerFunctor.hpp>
#include <Packet.hpp>
#include <ProfilingUtils.hpp>

namespace armnn
{

namespace gatordmock
{

class TimelineCaptureCommandHandler : public profiling::CommandHandlerFunctor
{
    // Utils
    uint32_t uint32_t_size = sizeof(uint32_t);
    uint32_t uint64_t_size = sizeof(uint64_t);
    uint32_t threadId_size = sizeof(std::thread::id);

    using ReadFunction = void (TimelineCaptureCommandHandler::*)(const unsigned char*, uint32_t);

public:
    TimelineCaptureCommandHandler(uint32_t familyId,
                                  uint32_t packetId,
                                  uint32_t version,
                                  Model* model,
                                  bool quietOperation = false)
            : CommandHandlerFunctor(familyId, packetId, version)
            , m_Model(model)
            , m_QuietOperation(quietOperation)
    {}

    void operator()(const armnn::profiling::Packet& packet) override;

    void ReadLabel(const unsigned char* data, uint32_t offset);
    void ReadEntity(const unsigned char* data, uint32_t offset);
    void ReadEventClass(const unsigned char* data, uint32_t offset);
    void ReadRelationship(const unsigned char* data, uint32_t offset);
    void ReadEvent(const unsigned char* data, uint32_t offset);

    void print();

private:
    void ParseData(const armnn::profiling::Packet& packet);

    Model* m_Model;
    bool m_QuietOperation;
    static const ReadFunction m_ReadFunctions[];

    void printLabels();
    void printEntities();
    void printEventClasses();
    void printRelationships();
    void printEvents();
};

} //namespace gatordmock

} //namespace armnn
