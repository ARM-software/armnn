//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "CommandHandlerRegistry.hpp"

#include <boost/assert.hpp>
#include <boost/format.hpp>

namespace armnn
{

namespace profiling
{

void CommandHandlerRegistry::RegisterFunctor(CommandHandlerFunctor* functor,
                                             uint32_t familyId,
                                             uint32_t packetId,
                                             uint32_t version)
{
    BOOST_ASSERT_MSG(functor, "Provided functor should not be a nullptr");

    CommandHandlerKey key(familyId, packetId, version);
    registry[key] = functor;
}

void CommandHandlerRegistry::RegisterFunctor(CommandHandlerFunctor* functor)
{
    BOOST_ASSERT_MSG(functor, "Provided functor should not be a nullptr");

    RegisterFunctor(functor, functor->GetFamilyId(), functor->GetPacketId(), functor->GetVersion());
}

CommandHandlerFunctor* CommandHandlerRegistry::GetFunctor(uint32_t familyId,uint32_t packetId, uint32_t version) const
{
    CommandHandlerKey key(familyId, packetId, version);

    // Check that the requested key exists
    if (registry.find(key) == registry.end())
    {
        throw armnn::InvalidArgumentException(
                    boost::str(boost::format("Functor with requested PacketId=%1% and Version=%2% does not exist")
                               % packetId
                               % version));
    }

    CommandHandlerFunctor* commandHandlerFunctor = registry.at(key);
    if (commandHandlerFunctor == nullptr)
    {
        throw RuntimeException(
                    boost::str(boost::format("Invalid functor registered for PacketId=%1% and Version=%2%")
                               % packetId
                               % version));
    }

    return commandHandlerFunctor;
}

} // namespace profiling

} // namespace armnn
