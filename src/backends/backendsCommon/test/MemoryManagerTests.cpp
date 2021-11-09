//
// Copyright © 2021 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include <backendsCommon/MemoryManager.hpp>
#include <armnn/utility/IgnoreUnused.hpp>

#include <doctest/doctest.h>
#include <numeric>

namespace armnn
{
/// @brief Class that implements a sample custom allocator.
class SampleCustomAllocator : public armnn::ICustomAllocator
{
public:
    SampleCustomAllocator() = default;

    void* allocate(size_t size, size_t alignment) override
    {
        IgnoreUnused(alignment);
        CHECK(size == m_Values.size());
        m_CounterAllocate+=1;
        return m_Values.data();
    }

    void free(void* ptr) override
    {
        CHECK(ptr == m_Values.data());
        m_CounterFree+=1;
    }

    armnn::MemorySource GetMemorySourceType() override
    {
        return armnn::MemorySource::Malloc;
    }

    virtual void* GetMemoryRegionAtOffset(void* buffer, size_t offset, size_t alignment = 0 ) override
    {
        IgnoreUnused(alignment);
        return (static_cast<char*>(buffer) + offset);
    }

    /// Holds the data in the tensors. Create for testing purposes.
    std::vector<uint8_t> m_Values;
    /// Counts the number of times the function allocate is called.
    unsigned long m_CounterAllocate= 0;
    /// Counts the number of times the function free is called.
    unsigned long m_CounterFree = 0;
};

TEST_SUITE("MemoryManagerTests")
{
/// Unit test Storing, Allocating and Deallocating with a custom allocator.
TEST_CASE("MemoryManagerTest")
{
    using namespace armnn;

    // Create mock up bufferStorageVector with 2 BufferStorage with the same TensorMemory
    size_t numTensors = 5;
    std::vector<std::shared_ptr<TensorMemory>> tensorMemoryPointerVector(numTensors);
    std::vector<std::shared_ptr<TensorMemory>> tensorMemoryVector;
    tensorMemoryVector.reserve(numTensors);

    std::vector<size_t> offsets(numTensors);
    std::iota(std::begin(offsets), std::end(offsets), 0);

    for (uint32_t idx = 0; idx < tensorMemoryPointerVector.size(); ++idx)
    {
        tensorMemoryVector.emplace_back(std::make_shared<TensorMemory>(TensorMemory{offsets[idx], 0, nullptr}));

        tensorMemoryPointerVector[idx] = tensorMemoryVector[idx];
    }

    std::vector<BufferStorage> bufferStorageVector;
    bufferStorageVector.emplace_back(BufferStorage{tensorMemoryPointerVector, numTensors});
    bufferStorageVector.emplace_back(BufferStorage{tensorMemoryPointerVector, numTensors});

    // Create an instance of the SampleCustomAllocator
    std::shared_ptr<SampleCustomAllocator> customAllocator =
            std::make_unique<SampleCustomAllocator>(SampleCustomAllocator());

    customAllocator->m_Values = {10, 11, 12, 13, 14};
    // Check that the test was set up correctly
    CHECK(customAllocator->m_Values.size() == numTensors);

    size_t bufferVecSize =  bufferStorageVector.size();
    // Utilise 3 functions in the MemoryManager. Check the counters and the pointer to the values are correct.
    MemoryManager memoryManager;
    memoryManager.StoreMemToAllocate(bufferStorageVector, customAllocator);

    memoryManager.Allocate();
    CHECK(customAllocator->m_CounterAllocate == bufferVecSize);

    uint32_t idx = 0;
    for (auto tensorMemory : tensorMemoryVector)
    {
        auto value = reinterpret_cast<uint8_t *>(tensorMemory->m_Data);
        CHECK(customAllocator->m_Values[idx] == *value);
        idx += 1;
    }

    memoryManager.Deallocate();
    CHECK(customAllocator->m_CounterFree == bufferStorageVector.size());
}
}

} // namespace armnn