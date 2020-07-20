//
// Copyright © 2020 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//
#pragma once

#include <common/include/CommandHandlerFunctor.hpp>
#include <common/include/IgnoreUnused.hpp>

#include <vector>

namespace armnn
{

namespace gatordmock
{

class StubCommandHandler : public arm::pipe::CommandHandlerFunctor
{

public:
    /**
     *
     * @param packetId The id of packets this handler will process.
     * @param version The version of that id.
     */
    StubCommandHandler(uint32_t familyId,
                       uint32_t packetId,
                       uint32_t version)
            : CommandHandlerFunctor(familyId, packetId, version)
    {}

    void operator()(const arm::pipe::Packet& packet) override
    {
        //No op
        arm::pipe::IgnoreUnused(packet);
    }

};

}    // namespace gatordmock
}    // namespace armnn
