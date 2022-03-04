//
// Copyright © 2020 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <armnn/Optional.hpp>

namespace arm
{

namespace pipe
{

class ICounterRegistry
{
public:
    virtual ~ICounterRegistry() {}

    // Register profiling objects
    virtual const Category*   RegisterCategory  (const std::string& categoryName) = 0;

    virtual const Device*     RegisterDevice    (const std::string& deviceName,
                                                 uint16_t cores,
                                                 const armnn::Optional<std::string>& parentCategoryName) = 0;

    virtual const CounterSet* RegisterCounterSet(const std::string& counterSetName,
                                                 uint16_t count,
                                                 const armnn::Optional<std::string>& parentCategoryName) = 0;

    virtual const Counter* RegisterCounter(const std::string& backendId,
                                           const uint16_t uid,
                                           const std::string& parentCategoryName,
                                           uint16_t counterClass,
                                           uint16_t interpolation,
                                           double multiplier,
                                           const std::string& name,
                                           const std::string& description,
                                           const armnn::Optional<std::string>& units = armnn::EmptyOptional(),
                                           const armnn::Optional<uint16_t>& numberOfCores = armnn::EmptyOptional(),
                                           const armnn::Optional<uint16_t>& deviceUid = armnn::EmptyOptional(),
                                           const armnn::Optional<uint16_t>& counterSetUid = armnn::EmptyOptional()) = 0;

};

} // namespace pipe

} // namespace arm
