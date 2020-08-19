//
// Copyright © 2020 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//
#pragma once

#include <armnn/backends/IMemoryManager.hpp>

#include <forward_list>
#include <vector>

namespace sdb // sample dynamic backend
{

// An implementation of IMemoryManager to be used with SampleTensorHandle
class SampleMemoryManager : public armnn::IMemoryManager
{
public:
    SampleMemoryManager();
    virtual ~SampleMemoryManager();

    class Pool;

    Pool* Manage(unsigned int numBytes);

    void Allocate(Pool *pool);

    void* GetPointer(Pool *pool);

    void Acquire() override;
    void Release() override;

    class Pool
    {
    public:
        Pool(unsigned int numBytes);
        ~Pool();

        void Acquire();
        void Release();

        void* GetPointer();

        void Reserve(unsigned int numBytes);

    private:
        unsigned int m_Size;
        void* m_Pointer;
    };

private:
    SampleMemoryManager(const SampleMemoryManager&) = delete; // Noncopyable
    SampleMemoryManager& operator=(const SampleMemoryManager&) = delete; // Noncopyable

    std::forward_list<Pool> m_Pools;
    std::vector<Pool*> m_FreePools;
};

} // namespace sdb
