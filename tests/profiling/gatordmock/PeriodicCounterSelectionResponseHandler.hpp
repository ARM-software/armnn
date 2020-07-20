//
// Copyright © 2019 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//
#pragma once

#include <common/include/CommandHandlerFunctor.hpp>
#include <common/include/Packet.hpp>

#include <vector>

namespace armnn
{

namespace gatordmock
{

class PeriodicCounterSelectionResponseHandler : public arm::pipe::CommandHandlerFunctor
{

public:
    /**
     *
     * @param packetId The id of packets this handler will process.
     * @param version The version of that id.
     * @param quietOperation Optional parameter to turn off printouts. This is useful for unittests.
     */
    PeriodicCounterSelectionResponseHandler(uint32_t familyId,
                                            uint32_t packetId,
                                            uint32_t version,
                                            bool quietOperation = true)
        : CommandHandlerFunctor(familyId, packetId, version)
        , m_QuietOperation(quietOperation)
    {}

    void operator()(const arm::pipe::Packet& packet) override;

private:
    bool m_QuietOperation;
};

}    // namespace gatordmock

}    // namespace armnn