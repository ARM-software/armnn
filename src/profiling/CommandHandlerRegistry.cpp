//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "CommandHandlerRegistry.hpp"

#include <boost/assert.hpp>
#include <boost/log/trivial.hpp>

namespace armnn
{

namespace profiling
{

void CommandHandlerRegistry::RegisterFunctor(CommandHandlerFunctor* functor, uint32_t packetId, uint32_t version)
{
    BOOST_ASSERT_MSG(functor, "Provided functor should not be a nullptr.");
    CommandHandlerKey key(packetId, version);
    registry[key] = functor;
}

CommandHandlerFunctor* CommandHandlerRegistry::GetFunctor(uint32_t packetId, uint32_t version) const
{
    CommandHandlerKey key(packetId, version);

    // Check that the requested key exists
    if (registry.find(key) == registry.end())
    {
        throw armnn::Exception("Functor with requested PacketId or Version does not exist.");
    }

    return registry.at(key);
}

} // namespace profiling

} // namespace armnn
