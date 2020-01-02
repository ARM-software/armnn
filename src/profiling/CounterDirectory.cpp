//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "CounterDirectory.hpp"
#include "ProfilingUtils.hpp"

#include <armnn/Exceptions.hpp>
#include <armnn/Conversion.hpp>

#include <boost/core/ignore_unused.hpp>
#include <boost/format.hpp>

namespace armnn
{

namespace profiling
{

const Category* CounterDirectory::RegisterCategory(const std::string& categoryName,
                                                   const Optional<uint16_t>& deviceUid,
                                                   const Optional<uint16_t>& counterSetUid)
{
    // Check that the given category name is valid
    if (categoryName.empty() ||
            !IsValidSwTraceString<SwTraceNameCharPolicy>(categoryName))
    {
        throw InvalidArgumentException("Trying to register a category with an invalid name");
    }

    // Check that the given category is not already registered
    if (IsCategoryRegistered(categoryName))
    {
        throw InvalidArgumentException(
                    boost::str(boost::format("Trying to register a category already registered (\"%1%\")")
                               % categoryName));
    }

    // Check that a device with the given (optional) UID is already registered
    uint16_t deviceUidValue = deviceUid.has_value() ? deviceUid.value() : 0;
    if (deviceUidValue > 0)
    {
        // Check that the (optional) device is already registered
        if (!IsDeviceRegistered(deviceUidValue))
        {
            throw InvalidArgumentException(
                        boost::str(boost::format("Trying to connect a category (\"%1%\") to a device that is "
                                                 "not registered (UID %2%)")
                                   % categoryName
                                   % deviceUidValue));
        }
    }

    // Check that a counter set with the given (optional) UID is already registered
    uint16_t counterSetUidValue = counterSetUid.has_value() ? counterSetUid.value() : 0;
    if (counterSetUidValue > 0)
    {
        // Check that the (optional) counter set is already registered
        if (!IsCounterSetRegistered(counterSetUidValue))
        {
            throw InvalidArgumentException(
                        boost::str(boost::format("Trying to connect a category (name: \"%1%\") to a counter set "
                                                 "that is not registered (UID: %2%)")
                                   % categoryName
                                   % counterSetUidValue));
        }
    }

    // Create the category
    CategoryPtr category = std::make_unique<Category>(categoryName, deviceUidValue, counterSetUidValue);
    BOOST_ASSERT(category);

    // Get the raw category pointer
    const Category* categoryPtr = category.get();
    BOOST_ASSERT(categoryPtr);

    // Register the category
    m_Categories.insert(std::move(category));

    return categoryPtr;
}

const Device* CounterDirectory::RegisterDevice(const std::string& deviceName,
                                               uint16_t cores,
                                               const Optional<std::string>& parentCategoryName)
{
    // Check that the given device name is valid
    if (deviceName.empty() ||
            !IsValidSwTraceString<SwTraceCharPolicy>(deviceName))
    {
        throw InvalidArgumentException("Trying to register a device with an invalid name");
    }

    // Check that a device with the given name is not already registered
    if (IsDeviceRegistered(deviceName))
    {
        throw InvalidArgumentException(
                    boost::str(boost::format("Trying to register a device already registered (\"%1%\")")
                               % deviceName));
    }

    // Peek the next UID, do not get an actual valid UID just now as we don't want to waste a good UID in case
    // the registration fails. We'll get a proper one once we're sure that the device can be registered
    uint16_t deviceUidPeek = GetNextUid(true);

    // Check that a category with the given (optional) parent category name is already registered
    Category* parentCategoryPtr = nullptr;
    if (parentCategoryName.has_value())
    {
        // Get the (optional) parent category name
        const std::string& parentCategoryNameValue = parentCategoryName.value();
        if (parentCategoryNameValue.empty())
        {
            throw InvalidArgumentException(
                        boost::str(boost::format("Trying to connect a device (name: \"%1%\") to an invalid "
                                                 "parent category (name: \"%2%\")")
                                   % deviceName
                                   % parentCategoryNameValue));
        }

        // Check that the given parent category is already registered
        auto categoryIt = FindCategory(parentCategoryNameValue);
        if (categoryIt == m_Categories.end())
        {
            throw InvalidArgumentException(
                        boost::str(boost::format("Trying to connect a device (name: \"%1%\") to a parent category that "
                                                 "is not registered (name: \"%2%\")")
                                   % deviceName
                                   % parentCategoryNameValue));
        }

        // Get the parent category
        const CategoryPtr& parentCategory = *categoryIt;
        BOOST_ASSERT(parentCategory);

        // Check that the given parent category is not already connected to another device
        if (parentCategory->m_DeviceUid != 0 && parentCategory->m_DeviceUid != deviceUidPeek)
        {
            throw InvalidArgumentException(
                        boost::str(boost::format("Trying to connect a device (UID: %1%) to a parent category that is "
                                                 "already connected to a different device "
                                                 "(category \"%2%\" connected to device %3%)")
                                   % deviceUidPeek
                                   % parentCategoryNameValue
                                   % parentCategory->m_DeviceUid));
        }

        // The parent category can be associated to the device that is about to be registered.
        // Get the raw pointer to the parent category (to be used later when the device is actually been
        // registered, to make sure that the category is associated to an existing device)
        parentCategoryPtr = parentCategory.get();
    }

    // Get the device UID
    uint16_t deviceUid = GetNextUid();
    BOOST_ASSERT(deviceUid == deviceUidPeek);

    // Create the device
    DevicePtr device = std::make_unique<Device>(deviceUid, deviceName, cores);
    BOOST_ASSERT(device);

    // Get the raw device pointer
    const Device* devicePtr = device.get();
    BOOST_ASSERT(devicePtr);

    // Register the device
    m_Devices.insert(std::make_pair(deviceUid, std::move(device)));

    // Connect the device to the parent category, if required
    if (parentCategoryPtr)
    {
        // Set the device UID in the parent category
        parentCategoryPtr->m_DeviceUid = deviceUid;
    }

    return devicePtr;
}

const CounterSet* CounterDirectory::RegisterCounterSet(const std::string& counterSetName,
                                                       uint16_t count,
                                                       const Optional<std::string>& parentCategoryName)
{
    // Check that the given counter set name is valid
    if (counterSetName.empty() ||
            !IsValidSwTraceString<SwTraceNameCharPolicy>(counterSetName))
    {
        throw InvalidArgumentException("Trying to register a counter set with an invalid name");
    }

    // Check that a counter set with the given name is not already registered
    if (IsCounterSetRegistered(counterSetName))
    {
        throw InvalidArgumentException(
                    boost::str(boost::format("Trying to register a counter set already registered (\"%1%\")")
                               % counterSetName));
    }

    // Peek the next UID, do not get an actual valid UID just now as we don't want to waste a good UID in case
    // the registration fails. We'll get a proper one once we're sure that the counter set can be registered
    uint16_t counterSetUidPeek = GetNextUid(true);

    // Check that a category with the given (optional) parent category name is already registered
    Category* parentCategoryPtr = nullptr;
    if (parentCategoryName.has_value())
    {
        // Get the (optional) parent category name
        const std::string& parentCategoryNameValue = parentCategoryName.value();
        if (parentCategoryNameValue.empty())
        {
            throw InvalidArgumentException(
                        boost::str(boost::format("Trying to connect a counter set (UID: %1%) to an invalid "
                                                 "parent category (name: \"%2%\")")
                                   % counterSetUidPeek
                                   % parentCategoryNameValue));
        }

        // Check that the given parent category is already registered
        auto it = FindCategory(parentCategoryNameValue);
        if (it == m_Categories.end())
        {
            throw InvalidArgumentException(
                        boost::str(boost::format("Trying to connect a counter set (UID: %1%) to a parent category "
                                                 "that is not registered (name: \"%2%\")")
                                   % counterSetUidPeek
                                   % parentCategoryNameValue));
        }

        // Get the parent category
        const CategoryPtr& parentCategory = *it;
        BOOST_ASSERT(parentCategory);

        // Check that the given parent category is not already connected to another counter set
        if (parentCategory->m_CounterSetUid != 0 && parentCategory->m_CounterSetUid != counterSetUidPeek)
        {
            throw InvalidArgumentException(
                        boost::str(boost::format("Trying to connect a counter set (UID: %1%) to a parent category "
                                                 "that is already connected to a different counter set "
                                                 "(category \"%2%\" connected to counter set %3%)")
                                   % counterSetUidPeek
                                   % parentCategoryNameValue
                                   % parentCategory->m_CounterSetUid));
        }

        // The parent category can be associated to the counter set that is about to be registered.
        // Get the raw pointer to the parent category (to be used later when the counter set is actually been
        // registered, to make sure that the category is associated to an existing counter set)
        parentCategoryPtr = parentCategory.get();
    }

    // Get the counter set UID
    uint16_t counterSetUid = GetNextUid();
    BOOST_ASSERT(counterSetUid == counterSetUidPeek);

    // Create the counter set
    CounterSetPtr counterSet = std::make_unique<CounterSet>(counterSetUid, counterSetName, count);
    BOOST_ASSERT(counterSet);

    // Get the raw counter set pointer
    const CounterSet* counterSetPtr = counterSet.get();
    BOOST_ASSERT(counterSetPtr);

    // Register the counter set
    m_CounterSets.insert(std::make_pair(counterSetUid, std::move(counterSet)));

    // Connect the counter set to the parent category, if required
    if (parentCategoryPtr)
    {
        // Set the counter set UID in the parent category
        parentCategoryPtr->m_CounterSetUid = counterSetUid;
    }

    return counterSetPtr;
}

const Counter* CounterDirectory::RegisterCounter(const BackendId& backendId,
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
                                                 const Optional<uint16_t>& counterSetUid)
{
    boost::ignore_unused(backendId);

    // Check that the given parent category name is valid
    if (parentCategoryName.empty() ||
            !IsValidSwTraceString<SwTraceNameCharPolicy>(parentCategoryName))
    {
        throw InvalidArgumentException("Trying to register a counter with an invalid parent category name");
    }

    // Check that the given class is valid
    if (counterClass != 0 && counterClass != 1)
    {
        throw InvalidArgumentException("Trying to register a counter with an invalid class");
    }

    // Check that the given interpolation is valid
    if (interpolation != 0 && interpolation != 1)
    {
        throw InvalidArgumentException("Trying to register a counter with an invalid interpolation");
    }

    // Check that the given multiplier is valid
    if (multiplier == .0f)
    {
        throw InvalidArgumentException("Trying to register a counter with an invalid multiplier");
    }

    // Check that the given name is valid
    if (name.empty() ||
            !IsValidSwTraceString<SwTraceCharPolicy>(name))
    {
        throw InvalidArgumentException("Trying to register a counter with an invalid name");
    }

    // Check that the given description is valid
    if (description.empty() ||
            !IsValidSwTraceString<SwTraceCharPolicy>(description))
    {
        throw InvalidArgumentException("Trying to register a counter with an invalid description");
    }

    // Check that the given units are valid
    if (units.has_value()
            && !IsValidSwTraceString<SwTraceNameCharPolicy>(units.value()))
    {
        throw InvalidArgumentException("Trying to register a counter with a invalid units");
    }

    // Check that the given parent category is registered
    auto categoryIt = FindCategory(parentCategoryName);
    if (categoryIt == m_Categories.end())
    {
        throw InvalidArgumentException(
                    boost::str(boost::format("Trying to connect a counter to a category "
                                             "that is not registered (name: \"%1%\")")
                               % parentCategoryName));
    }

    // Get the parent category
    const CategoryPtr& parentCategory = *categoryIt;
    BOOST_ASSERT(parentCategory);

    // Check that a counter with the given name is not already registered within the parent category
    const std::vector<uint16_t>& parentCategoryCounters = parentCategory->m_Counters;
    for (uint16_t parentCategoryCounterUid : parentCategoryCounters)
    {
        const Counter* parentCategoryCounter = GetCounter(parentCategoryCounterUid);
        BOOST_ASSERT(parentCategoryCounter);

        if (parentCategoryCounter->m_Name == name)
        {
            throw InvalidArgumentException(
                        boost::str(boost::format("Trying to register a counter to category \"%1%\" with a name that "
                                                 "is already used within that category (name: \"%2%\")")
                                   % parentCategoryName
                                   % name));
        }
    }

    // Check that a counter set with the given (optional) UID is already registered
    uint16_t counterSetUidValue = counterSetUid.has_value() ? counterSetUid.value() : 0;
    if (counterSetUidValue > 0)
    {
        // Check that the (optional) counter set is already registered
        if (!IsCounterSetRegistered(counterSetUidValue))
        {
            throw InvalidArgumentException(
                        boost::str(boost::format("Trying to connect a counter to a counter set that is "
                                                 "not registered (counter set UID: %1%)")
                                   % counterSetUidValue));
        }
    }

    // Get the number of cores (this call may throw)
    uint16_t deviceUidValue = deviceUid.has_value() ? deviceUid.value() : 0;
    uint16_t deviceCores = GetNumberOfCores(numberOfCores, deviceUidValue, parentCategory);

    // Get the counter UIDs and calculate the max counter UID
    std::vector<uint16_t> counterUids = GetNextCounterUids(uid, deviceCores);
    BOOST_ASSERT(!counterUids.empty());
    uint16_t maxCounterUid = deviceCores <= 1 ? counterUids.front() : counterUids.back();

    // Get the counter units
    const std::string unitsValue = units.has_value() ? units.value() : "";

    // Create the counter
    CounterPtr counter = std::make_shared<Counter>(armnn::profiling::BACKEND_ID,
                                                   counterUids.front(),
                                                   maxCounterUid,
                                                   counterClass,
                                                   interpolation,
                                                   multiplier,
                                                   name,
                                                   description,
                                                   unitsValue,
                                                   deviceUidValue,
                                                   counterSetUidValue);
    BOOST_ASSERT(counter);

    // Get the raw counter pointer
    const Counter* counterPtr = counter.get();
    BOOST_ASSERT(counterPtr);

    // Process multiple counters if necessary
    for (uint16_t counterUid : counterUids)
    {
        // Connect the counter to the parent category
        parentCategory->m_Counters.push_back(counterUid);

        // Register the counter
        m_Counters.insert(std::make_pair(counterUid, counter));
    }

    return counterPtr;
}

const Category* CounterDirectory::GetCategory(const std::string& categoryName) const
{
    auto it = FindCategory(categoryName);
    if (it == m_Categories.end())
    {
        return nullptr;
    }

    const Category* category = it->get();
    BOOST_ASSERT(category);

    return category;
}

const Device* CounterDirectory::GetDevice(uint16_t deviceUid) const
{
    auto it = FindDevice(deviceUid);
    if (it == m_Devices.end())
    {
        return nullptr;
    }

    const Device* device = it->second.get();
    BOOST_ASSERT(device);
    BOOST_ASSERT(device->m_Uid == deviceUid);

    return device;
}

const CounterSet* CounterDirectory::GetCounterSet(uint16_t counterSetUid) const
{
    auto it = FindCounterSet(counterSetUid);
    if (it == m_CounterSets.end())
    {
        return nullptr;
    }

    const CounterSet* counterSet = it->second.get();
    BOOST_ASSERT(counterSet);
    BOOST_ASSERT(counterSet->m_Uid == counterSetUid);

    return counterSet;
}

const Counter* CounterDirectory::GetCounter(uint16_t counterUid) const
{
    auto it = FindCounter(counterUid);
    if (it == m_Counters.end())
    {
        return nullptr;
    }

    const Counter* counter = it->second.get();
    BOOST_ASSERT(counter);
    BOOST_ASSERT(counter->m_Uid <= counterUid);
    BOOST_ASSERT(counter->m_Uid <= counter->m_MaxCounterUid);

    return counter;
}

bool CounterDirectory::IsCategoryRegistered(const std::string& categoryName) const
{
    auto it = FindCategory(categoryName);

    return it != m_Categories.end();
}

bool CounterDirectory::IsDeviceRegistered(uint16_t deviceUid) const
{
    auto it = FindDevice(deviceUid);

    return it != m_Devices.end();
}

bool CounterDirectory::IsDeviceRegistered(const std::string& deviceName) const
{
    auto it = FindDevice(deviceName);

    return it != m_Devices.end();
}

bool CounterDirectory::IsCounterSetRegistered(uint16_t counterSetUid) const
{
    auto it = FindCounterSet(counterSetUid);

    return it != m_CounterSets.end();
}

bool CounterDirectory::IsCounterSetRegistered(const std::string& counterSetName) const
{
    auto it = FindCounterSet(counterSetName);

    return it != m_CounterSets.end();
}

bool CounterDirectory::IsCounterRegistered(uint16_t counterUid) const
{
    auto it = FindCounter(counterUid);

    return it != m_Counters.end();
}

bool CounterDirectory::IsCounterRegistered(const std::string& counterName) const
{
    auto it = FindCounter(counterName);

    return it != m_Counters.end();
}

void CounterDirectory::Clear()
{
    // Clear all the counter directory contents
    m_Categories.clear();
    m_Devices.clear();
    m_CounterSets.clear();
    m_Counters.clear();
}

CategoriesIt CounterDirectory::FindCategory(const std::string& categoryName) const
{
    return std::find_if(m_Categories.begin(), m_Categories.end(), [&categoryName](const CategoryPtr& category)
    {
        BOOST_ASSERT(category);

        return category->m_Name == categoryName;
    });
}

DevicesIt CounterDirectory::FindDevice(uint16_t deviceUid) const
{
    return m_Devices.find(deviceUid);
}

DevicesIt CounterDirectory::FindDevice(const std::string& deviceName) const
{
    return std::find_if(m_Devices.begin(), m_Devices.end(), [&deviceName](const auto& pair)
    {
        BOOST_ASSERT(pair.second);
        BOOST_ASSERT(pair.second->m_Uid == pair.first);

        return pair.second->m_Name == deviceName;
    });
}

CounterSetsIt CounterDirectory::FindCounterSet(uint16_t counterSetUid) const
{
    return m_CounterSets.find(counterSetUid);
}

CounterSetsIt CounterDirectory::FindCounterSet(const std::string& counterSetName) const
{
    return std::find_if(m_CounterSets.begin(), m_CounterSets.end(), [&counterSetName](const auto& pair)
    {
        BOOST_ASSERT(pair.second);
        BOOST_ASSERT(pair.second->m_Uid == pair.first);

        return pair.second->m_Name == counterSetName;
    });
}

CountersIt CounterDirectory::FindCounter(uint16_t counterUid) const
{
    return m_Counters.find(counterUid);
}

CountersIt CounterDirectory::FindCounter(const std::string& counterName) const
{
    return std::find_if(m_Counters.begin(), m_Counters.end(), [&counterName](const auto& pair)
    {
        BOOST_ASSERT(pair.second);
        BOOST_ASSERT(pair.second->m_Uid == pair.first);

        return pair.second->m_Name == counterName;
    });
}

uint16_t CounterDirectory::GetNumberOfCores(const Optional<uint16_t>& numberOfCores,
                                            uint16_t deviceUid,
                                            const CategoryPtr& parentCategory)
{
    BOOST_ASSERT(parentCategory);

    // To get the number of cores, apply the following rules:
    //
    // 1. If numberOfCores is set then take it as the deviceCores value
    // 2. If numberOfCores is not set then check to see if this counter is directly associated with a device,
    //    if so then that devices number of cores is taken as the deviceCores value
    // 3. If neither of the above is set then look at the category to see if it has a device associated with it,
    //    if it does then take that device's numberOfCores as the deviceCores value
    // 4. If none of the above holds then set deviceCores to zero

    // 1. If numberOfCores is set then take it as the deviceCores value
    if (numberOfCores.has_value())
    {
        // Get the number of cores
        return numberOfCores.value();
    }

    // 2. If numberOfCores is not set then check to see if this counter is directly associated with a device,
    //    if so then that devices number of cores is taken as the deviceCores value
    if (deviceUid > 0)
    {
        // Check that the (optional) device is already registered
        auto deviceIt = FindDevice(deviceUid);
        if (deviceIt == m_Devices.end())
        {
            throw InvalidArgumentException(
                        boost::str(boost::format("Trying to connect a counter to a device that is "
                                                 "not registered (device UID %1%)")
                                   % deviceUid));
        }

        // Get the associated device
        const DevicePtr& device = deviceIt->second;
        BOOST_ASSERT(device);

        // Get the number of cores of the associated device
        return device->m_Cores;
    }

    // 3. If neither of the above is set then look at the category to see if it has a device associated with it,
    //    if it does then take that device's numberOfCores as the deviceCores value
    uint16_t parentCategoryDeviceUid = parentCategory->m_DeviceUid;
    if (parentCategoryDeviceUid > 0)
    {
        // Check that the device associated to the parent category is already registered
        auto deviceIt = FindDevice(parentCategoryDeviceUid);
        if (deviceIt == m_Devices.end())
        {
            throw InvalidArgumentException(
                        boost::str(boost::format("Trying to get the number of cores from a device that is "
                                                 "not registered (device UID %1%)")
                                   % parentCategoryDeviceUid));
        }

        // Get the associated device
        const DevicePtr& device = deviceIt->second;
        BOOST_ASSERT(device);

        // Get the number of cores of the device associated to the parent category
        return device->m_Cores;
    }

    // 4. If none of the above holds then set deviceCores to zero
    return 0;
}

} // namespace profiling

} // namespace armnn
