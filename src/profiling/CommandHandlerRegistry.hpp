//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include "CommandHandlerFunctor.hpp"
#include "CommandHandlerKey.hpp"

#include <boost/functional/hash.hpp>

#include <unordered_map>

namespace armnn
{

namespace profiling
{

struct CommandHandlerHash
{
    std::size_t operator() (const CommandHandlerKey& commandHandlerKey) const
    {
        std::size_t seed = 0;
        boost::hash_combine(seed, commandHandlerKey.GetPacketId());
        boost::hash_combine(seed, commandHandlerKey.GetVersion());
        return seed;
    }
};

class CommandHandlerRegistry
{
public:
    CommandHandlerRegistry() = default;

    void RegisterFunctor(CommandHandlerFunctor* functor, uint32_t packetId, uint32_t version);

    CommandHandlerFunctor* GetFunctor(uint32_t packetId, uint32_t version) const;

private:
    std::unordered_map<CommandHandlerKey, CommandHandlerFunctor*, CommandHandlerHash> registry;
};

} // namespace profiling

} // namespace armnn
