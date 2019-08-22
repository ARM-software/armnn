//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "CounterDirectory.hpp"

#include <armnn/Exceptions.hpp>

namespace armnn
{

namespace profiling
{

CounterDirectory::CounterDirectory(uint16_t uid,
                                   const std::string& name,
                                   uint16_t deviceCount,
                                   uint16_t counterCount,
                                   uint16_t categoryCount)
    : m_Uid(uid)
    , m_Name(name)
    , m_DeviceCount(deviceCount)
    , m_CounterCount(counterCount)
    , m_CategoryCount(categoryCount)
    , m_DeviceIds(deviceCount)
    , m_CounterIds(counterCount)
    , m_CategoryIds(categoryCount)
    , m_DeviceObjects(deviceCount)
    , m_CounterObjects(counterCount)
    , m_CategoryObjects(categoryCount)
{}

// Helper methods
void CounterDirectory::CheckDeviceIndex(uint16_t index) const
{
    if (index >= m_DeviceCount)
    {
        throw InvalidArgumentException("Invalid device index");
    }
}

void CounterDirectory::CheckCounterIndex(uint16_t index) const
{
    if (index >= m_CounterCount)
    {
        throw InvalidArgumentException("Invalid counter index");
    }
}

void CounterDirectory::CheckCategoryIndex(uint16_t index) const
{
    if (index >= m_CategoryCount)
    {
        throw InvalidArgumentException("Invalid category index");
    }
}

// Getters for basic attributes
uint16_t CounterDirectory::GetUid() const
{
    return m_Uid;
}

const std::string& CounterDirectory::GetName() const
{
    return m_Name;
}

// Getters for counts
uint16_t CounterDirectory::GetDeviceCount() const
{
    return m_DeviceCount;
}

uint16_t CounterDirectory::GetCounterCount() const
{
    return m_CounterCount;
}

uint16_t CounterDirectory::GetCategoryCount() const
{
    return m_CategoryCount;
}

// Getters and setters for devices
void CounterDirectory::GetDeviceValue(uint16_t index, uint32_t& value) const
{
    CheckDeviceIndex(index);
    value = m_DeviceIds[index].load();
}

void CounterDirectory::SetDeviceValue(uint16_t index, uint32_t value)
{
    CheckDeviceIndex(index);
    m_DeviceIds[index].store(value);
}

void CounterDirectory::GetDeviceObject(uint16_t index, Device* device) const
{
    CheckDeviceIndex(index);
    device = m_DeviceObjects[index].load();
}

void CounterDirectory::SetDeviceObject(uint16_t index, Device* device)
{
    CheckDeviceIndex(index);
    m_DeviceObjects[index].store(device);
}

// Getters and setters for counters
void CounterDirectory::GetCounterValue(uint16_t index, uint32_t& value) const
{
    CheckCounterIndex(index);
    value = m_CounterIds[index].load();
}

void CounterDirectory::SetCounterValue(uint16_t index, uint32_t value)
{
    CheckCounterIndex(index);
    m_CounterIds[index].store(value);
}

void CounterDirectory::GetCounterObject(uint16_t index, Counter* counter) const
{
    CheckCounterIndex(index);
    counter = m_CounterObjects[index].load();
}

void CounterDirectory::SetCounterObject(uint16_t index, Counter* counter)
{
    CheckCounterIndex(index);
    m_CounterObjects[index].store(counter);
}

// Getters and setters for categories
void CounterDirectory::GetCategoryValue(uint16_t index, uint32_t& value) const
{
    CheckCategoryIndex(index);
    value = m_CategoryIds[index].load();
}

void CounterDirectory::SetCategoryValue(uint16_t index, uint32_t value)
{
    CheckCategoryIndex(index);
    m_CategoryIds[index].store(value);
}

void CounterDirectory::GetCategoryObject(uint16_t index, Category* category) const
{
    CheckCategoryIndex(index);
    category = m_CategoryObjects[index].load();
}

void CounterDirectory::SetCategoryObject(uint16_t index, Category* category)
{
    CheckCategoryIndex(index);
    m_CategoryObjects[index].store(category);
}

} // namespace profiling

} // namespace armnn
