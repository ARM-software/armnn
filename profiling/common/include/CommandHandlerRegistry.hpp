//
// Copyright © 2017 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include "CommandHandlerFunctor.hpp"
#include "CommandHandlerKey.hpp"

#include <functional>
#include <unordered_map>

namespace arm
{

namespace pipe
{

struct CommandHandlerHash
{
    std::size_t operator() (const CommandHandlerKey& commandHandlerKey) const
    {
        std::size_t seed = 0;
        std::hash<uint32_t> hasher;
        seed ^= hasher(commandHandlerKey.GetPacketId()) + 0x9e3779b9 + (seed<<6) + (seed>>2);
        seed ^= hasher(commandHandlerKey.GetVersion()) + 0x9e3779b9 + (seed<<6) + (seed>>2);
        return seed;
    }
};

class CommandHandlerRegistry
{
public:
    CommandHandlerRegistry() = default;

    void RegisterFunctor(CommandHandlerFunctor* functor, uint32_t familyId, uint32_t packetId, uint32_t version);

    void RegisterFunctor(CommandHandlerFunctor* functor);

    CommandHandlerFunctor* GetFunctor(uint32_t familyId, uint32_t packetId, uint32_t version) const;

private:
    std::unordered_map<CommandHandlerKey, CommandHandlerFunctor*, CommandHandlerHash> registry;
};

} // namespace pipe

} // namespace arm
