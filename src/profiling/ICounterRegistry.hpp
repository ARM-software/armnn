//
// Copyright © 2020 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <armnn/Optional.hpp>
#include <armnn/BackendId.hpp>

namespace armnn
{

namespace profiling
{

class ICounterRegistry
{
public:
    virtual ~ICounterRegistry() {}

    // Register profiling objects
    virtual const Category*   RegisterCategory  (const std::string& categoryName) = 0;

    virtual const Device*     RegisterDevice    (const std::string& deviceName,
                                                 uint16_t cores,
                                                 const Optional<std::string>& parentCategoryName) = 0;

    virtual const CounterSet* RegisterCounterSet(const std::string& counterSetName,
                                                 uint16_t count,
                                                 const Optional<std::string>& parentCategoryName) = 0;

    virtual const Counter* RegisterCounter(const BackendId& backendId,
                                           const uint16_t uid,
                                           const std::string& parentCategoryName,
                                           uint16_t counterClass,
                                           uint16_t interpolation,
                                           double multiplier,
                                           const std::string& name,
                                           const std::string& description,
                                           const Optional<std::string>& units,
                                           const Optional<uint16_t>& numberOfCores,
                                           const Optional<uint16_t>& deviceUid,
                                           const Optional<uint16_t>& counterSetUid) = 0;

};

} // namespace profiling

} // namespace armnn
