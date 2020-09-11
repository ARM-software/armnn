//
// Copyright © 2019 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <armnn/BackendId.hpp>

#include <string>
#include <vector>
#include <memory>
#include <unordered_set>
#include <unordered_map>

namespace armnn
{

namespace profiling
{

// Forward declarations
class Category;
class Device;
class CounterSet;
class Counter;

// Profiling objects smart pointer types
using CategoryPtr   = std::unique_ptr<Category>;
using DevicePtr     = std::unique_ptr<Device>;
using CounterSetPtr = std::unique_ptr<CounterSet>;
using CounterPtr    = std::shared_ptr<Counter>;

// Profiling objects collection types
using Categories  = std::unordered_set<CategoryPtr>;
using Devices     = std::unordered_map<uint16_t, DevicePtr>;
using CounterSets = std::unordered_map<uint16_t, CounterSetPtr>;
using Counters    = std::unordered_map<uint16_t, CounterPtr>;

// Profiling objects collection iterator types
using CategoriesIt  = Categories::const_iterator;
using DevicesIt     = Devices::const_iterator;
using CounterSetsIt = CounterSets::const_iterator;
using CountersIt    = Counters::const_iterator;

class Category final
{
public:
    // Constructors
    Category(const std::string& name)
        : m_Name(name)
    {}

    // Fields
    std::string m_Name;

    // Connections
    std::vector<uint16_t> m_Counters;      // The UIDs of the counters associated with this category
};

class Device final
{
public:
    // Constructors
    Device(uint16_t deviceUid, const std::string& name, uint16_t cores)
        : m_Uid(deviceUid)
        , m_Name(name)
        , m_Cores(cores)
    {}

    // Fields
    uint16_t    m_Uid;
    std::string m_Name;
    uint16_t    m_Cores;
};

class CounterSet final
{
public:
    // Constructors
    CounterSet(uint16_t counterSetUid, const std::string& name, uint16_t count)
        : m_Uid(counterSetUid)
        , m_Name(name)
        , m_Count(count)
    {}

    // Fields
    uint16_t    m_Uid;
    std::string m_Name;
    uint16_t    m_Count;
};

class Counter final
{
public:
    // Constructors
    Counter(BackendId          backendId,
            uint16_t           counterUid,
            uint16_t           maxCounterUid,
            uint16_t           counterClass,
            uint16_t           interpolation,
            double             multiplier,
            const std::string& name,
            const std::string& description,
            const std::string& units,
            uint16_t           deviceUid,
            uint16_t           counterSetUid)
        : m_BackendId(backendId)
        , m_Uid(counterUid)
        , m_MaxCounterUid(maxCounterUid)
        , m_Class(counterClass)
        , m_Interpolation(interpolation)
        , m_Multiplier(multiplier)
        , m_Name(name)
        , m_Description(description)
        , m_Units(units)
        , m_DeviceUid(deviceUid)
        , m_CounterSetUid(counterSetUid)
    {}

    // Fields
    BackendId   m_BackendId;
    uint16_t    m_Uid;
    uint16_t    m_MaxCounterUid;
    uint16_t    m_Class;
    uint16_t    m_Interpolation;
    double      m_Multiplier;
    std::string m_Name;
    std::string m_Description;
    std::string m_Units;      // Optional, leave empty if the counter does not need units

    // Connections
    uint16_t m_DeviceUid;     // Optional, set to zero if the counter is not associated with a device
    uint16_t m_CounterSetUid; // Optional, set to zero if the counter is not associated with a counter set
};

class ICounterDirectory
{
public:
    virtual ~ICounterDirectory() {}

    // Getters for counts
    virtual uint16_t GetCategoryCount()   const = 0;
    virtual uint16_t GetDeviceCount()     const = 0;
    virtual uint16_t GetCounterSetCount() const = 0;
    virtual uint16_t GetCounterCount()    const = 0;

    // Getters for collections
    virtual const Categories&  GetCategories()  const = 0;
    virtual const Devices&     GetDevices()     const = 0;
    virtual const CounterSets& GetCounterSets() const = 0;
    virtual const Counters&    GetCounters()    const = 0;

    // Getters for profiling objects
    virtual const Category*   GetCategory(const std::string& name) const = 0;
    virtual const Device*     GetDevice(uint16_t uid)              const = 0;
    virtual const CounterSet* GetCounterSet(uint16_t uid)          const = 0;
    virtual const Counter*    GetCounter(uint16_t uid)             const = 0;
};

} // namespace profiling

} // namespace armnn
