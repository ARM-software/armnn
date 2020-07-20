//
// Copyright © 2017 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//


#include <common/include/Assert.hpp>
#include <common/include/CommandHandlerRegistry.hpp>

#include <sstream>

namespace arm
{

namespace pipe
{

void CommandHandlerRegistry::RegisterFunctor(CommandHandlerFunctor* functor,
                                             uint32_t familyId,
                                             uint32_t packetId,
                                             uint32_t version)
{
    ARM_PIPE_ASSERT_MSG(functor, "Provided functor should not be a nullptr");

    CommandHandlerKey key(familyId, packetId, version);
    registry[key] = functor;
}

void CommandHandlerRegistry::RegisterFunctor(CommandHandlerFunctor* functor)
{
    ARM_PIPE_ASSERT_MSG(functor, "Provided functor should not be a nullptr");

    RegisterFunctor(functor, functor->GetFamilyId(), functor->GetPacketId(), functor->GetVersion());
}

CommandHandlerFunctor* CommandHandlerRegistry::GetFunctor(uint32_t familyId,uint32_t packetId, uint32_t version) const
{
    CommandHandlerKey key(familyId, packetId, version);

    // Check that the requested key exists
    if (registry.find(key) == registry.end())
    {
        std::stringstream ss;
        ss << "Functor with requested PacketId=" << packetId << " and Version=" << version << " does not exist";
        throw ProfilingException(ss.str());
    }

    CommandHandlerFunctor* commandHandlerFunctor = registry.at(key);
    if (commandHandlerFunctor == nullptr)
    {
        std::stringstream ss;
        ss << "Invalid functor registered for PacketId=" << packetId << " and Version=" << version;
        throw ProfilingException(ss.str());
    }

    return commandHandlerFunctor;
}

} // namespace pipe

} // namespace arm
