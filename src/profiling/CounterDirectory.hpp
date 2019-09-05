//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <atomic>
#include <string>
#include <vector>

namespace armnn
{

namespace profiling
{

class Category
{
public:
    std::string m_Name;
};

class Device
{
public:
    uint16_t    m_Uid;
    std::string m_Name;
    uint16_t    m_Cores;
};

class Counter
{
public:
    uint16_t    m_Uid;
    uint16_t    m_MaxCounterUid;
    uint16_t    m_Class;
    uint16_t    m_Interpolation;
    float       m_Multiplier;
    std::string m_Name;
    std::string m_Description;
    std::string m_Units;
};

class CounterSet
{
public:
    uint16_t    m_Uid;
    std::string m_Name;
    uint16_t    m_Count;
};

class CounterDirectory final
{
public:
    CounterDirectory(uint16_t uid,
                     const std::string& name,
                     uint16_t deviceCount,
                     uint16_t counterCount,
                     uint16_t categoryCount);

    ~CounterDirectory() = default;

    uint16_t GetUid() const;
    const std::string& GetName() const;

    uint16_t GetDeviceCount() const;
    uint16_t GetCounterCount() const;
    uint16_t GetCategoryCount() const;

    void GetDeviceValue(uint16_t index, uint32_t& value) const;
    void SetDeviceValue(uint16_t index, uint32_t value);

    void GetDeviceObject(uint16_t index, Device* counter) const;
    void SetDeviceObject(uint16_t index, Device* counter);

    void GetCounterValue(uint16_t index, uint32_t& value) const;
    void SetCounterValue(uint16_t index, uint32_t value);

    void GetCounterObject(uint16_t index, Counter* counter) const;
    void SetCounterObject(uint16_t index, Counter* counter);

    void GetCategoryValue(uint16_t index, uint32_t& value) const;
    void SetCategoryValue(uint16_t index, uint32_t value);

    void GetCategoryObject(uint16_t index, Category* counter) const;
    void SetCategoryObject(uint16_t index, Category* counter);

private:
    uint16_t    m_Uid;
    std::string m_Name;

    uint16_t m_DeviceCount;
    uint16_t m_CounterCount;
    uint16_t m_CategoryCount;

    std::vector<std::atomic<uint32_t>> m_DeviceIds;
    std::vector<std::atomic<uint32_t>> m_CounterIds;
    std::vector<std::atomic<uint32_t>> m_CategoryIds;

    std::vector<std::atomic<Device*>>   m_DeviceObjects;
    std::vector<std::atomic<Counter*>>  m_CounterObjects;
    std::vector<std::atomic<Category*>> m_CategoryObjects;

    void CheckDeviceIndex(uint16_t index) const;
    void CheckCounterIndex(uint16_t index) const;
    void CheckCategoryIndex(uint16_t index) const;
};

} // namespace profiling

} // namespace armnn
