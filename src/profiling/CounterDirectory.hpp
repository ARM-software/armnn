//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include "ICounterDirectory.hpp"

#include <armnn/Optional.hpp>

#include <string>
#include <unordered_set>
#include <unordered_map>

#include <boost/numeric/conversion/cast.hpp>

namespace armnn
{

namespace profiling
{

class CounterDirectory final : public ICounterDirectory
{
public:
    CounterDirectory() = default;
    ~CounterDirectory() = default;

    // Register profiling objects
    const Category*   RegisterCategory  (const std::string& categoryName,
                                         const Optional<uint16_t>& deviceUid = EmptyOptional(),
                                         const Optional<uint16_t>& counterSetUid = EmptyOptional());
    const Device*     RegisterDevice    (const std::string& deviceName,
                                         uint16_t cores = 0,
                                         const Optional<std::string>& parentCategoryName = EmptyOptional());
    const CounterSet* RegisterCounterSet(const std::string& counterSetName,
                                         uint16_t count = 0,
                                         const Optional<std::string>& parentCategoryName = EmptyOptional());
    const Counter*    RegisterCounter   (const std::string& parentCategoryName,
                                         uint16_t counterClass,
                                         uint16_t interpolation,
                                         double multiplier,
                                         const std::string& name,
                                         const std::string& description,
                                         const Optional<std::string>& units = EmptyOptional(),
                                         const Optional<uint16_t>& numberOfCores = EmptyOptional(),
                                         const Optional<uint16_t>& deviceUid = EmptyOptional(),
                                         const Optional<uint16_t>& counterSetUid = EmptyOptional());

    // Getters for counts
    uint16_t GetCategoryCount()   const override { return boost::numeric_cast<uint16_t>(m_Categories.size());  }
    uint16_t GetDeviceCount()     const override { return boost::numeric_cast<uint16_t>(m_Devices.size());     }
    uint16_t GetCounterSetCount() const override { return boost::numeric_cast<uint16_t>(m_CounterSets.size()); }
    uint16_t GetCounterCount()    const override { return boost::numeric_cast<uint16_t>(m_Counters.size());    }

    // Getters for collections
    const Categories&  GetCategories()  const override { return m_Categories;  }
    const Devices&     GetDevices()     const override { return m_Devices;     }
    const CounterSets& GetCounterSets() const override { return m_CounterSets; }
    const Counters&    GetCounters()    const override { return m_Counters;    }

    // Getters for profiling objects
    const Category*   GetCategory(const std::string& name) const override;
    const Device*     GetDevice(uint16_t uid) const override;
    const CounterSet* GetCounterSet(uint16_t uid) const override;
    const Counter*    GetCounter(uint16_t uid) const override;

private:
    // The profiling collections owned by the counter directory
    Categories  m_Categories;
    Devices     m_Devices;
    CounterSets m_CounterSets;
    Counters    m_Counters;

    // Helper functions
    CategoriesIt  FindCategory(const std::string& categoryName) const;
    DevicesIt     FindDevice(uint16_t deviceUid) const;
    DevicesIt     FindDevice(const std::string& deviceName) const;
    CounterSetsIt FindCounterSet(uint16_t counterSetUid) const;
    CounterSetsIt FindCounterSet(const std::string& counterSetName) const;
    CountersIt    FindCounter(uint16_t counterUid) const;
    bool CheckIfCategoryIsRegistered(const std::string& categoryName) const;
    bool CheckIfDeviceIsRegistered(uint16_t deviceUid) const;
    bool CheckIfDeviceIsRegistered(const std::string& deviceName) const;
    bool CheckIfCounterSetIsRegistered(uint16_t counterSetUid) const;
    bool CheckIfCounterSetIsRegistered(const std::string& counterSetName) const;
    uint16_t GetNumberOfCores(const Optional<uint16_t>& numberOfCores,
                              uint16_t deviceUid,
                              const CategoryPtr& parentCategory);
};

} // namespace profiling

} // namespace armnn
