//
// Copyright © 2020 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//
#pragma once

#include <CommandHandlerFunctor.hpp>
#include <armnn/utility/IgnoreUnused.hpp>

#include <vector>

namespace armnn
{

namespace gatordmock
{

class StubCommandHandler : public profiling::CommandHandlerFunctor
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

    void operator()(const armnn::profiling::Packet& packet) override
    {
        //No op
        IgnoreUnused(packet);
    }

};

}    // namespace gatordmock
}    // namespace armnn
