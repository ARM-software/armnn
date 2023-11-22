//
// Copyright © 2022-2023 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//
#pragma once

#include <armnn/backends/IMemoryManager.hpp>

#include <forward_list>
#include <vector>

namespace armnn
{

// A dummy MemoryManager which will be deleted once the GpuFsa Backend is integrated with ClMemoryManager
class GpuFsaMemoryManager : public IMemoryManager
{
public:
    GpuFsaMemoryManager();
    virtual ~GpuFsaMemoryManager();

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
    GpuFsaMemoryManager(const GpuFsaMemoryManager&) = delete; // Noncopyable
    GpuFsaMemoryManager& operator=(const GpuFsaMemoryManager&) = delete; // Noncopyable

    std::forward_list<Pool> m_Pools;
    std::vector<Pool*> m_FreePools;
};

}
