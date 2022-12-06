//
// Copyright © 2020,2022 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <common/include/Optional.hpp>

namespace arm
{

namespace pipe
{

class Category;
class Device;
class CounterSet;
class Counter;

class ICounterRegistry
{
public:
    virtual ~ICounterRegistry() {}

    // Register profiling objects
    virtual const Category*   RegisterCategory  (const std::string& categoryName) = 0;

    virtual const Device*     RegisterDevice    (const std::string& deviceName,
                                                 uint16_t cores,
                                                 const arm::pipe::Optional<std::string>& parentCategoryName) = 0;

    virtual const CounterSet* RegisterCounterSet(const std::string& counterSetName,
                                                 uint16_t count,
                                                 const arm::pipe::Optional<std::string>& parentCategoryName) = 0;

    virtual const Counter* RegisterCounter(const std::string& backendId,
        const uint16_t uid,
        const std::string& parentCategoryName,
        uint16_t counterClass,
        uint16_t interpolation,
        double multiplier,
        const std::string& name,
        const std::string& description,
        const arm::pipe::Optional<std::string>& units = arm::pipe::EmptyOptional(),
        const arm::pipe::Optional<uint16_t>& numberOfCores = arm::pipe::EmptyOptional(),
        const arm::pipe::Optional<uint16_t>& deviceUid = arm::pipe::EmptyOptional(),
        const arm::pipe::Optional<uint16_t>& counterSetUid = arm::pipe::EmptyOptional()) = 0;

};

} // namespace pipe

} // namespace arm
