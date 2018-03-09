//
// Copyright © 2017 Arm Ltd. All rights reserved.
// See LICENSE file in the project root for full license information.
//
#pragma once
#include "CpuTensorHandleFwd.hpp"

#include "armnn/TypesUtils.hpp"

#include "OutputHandler.hpp"

namespace armnn
{

// Abstract tensor handle wrapping a CPU-readable region of memory, interpreting it as tensor data.
class ConstCpuTensorHandle : public ITensorHandle
{
public:
    template <typename T>
    const T* GetConstTensor() const
    {
        BOOST_ASSERT(GetTensorInfo().GetDataType() == GetDataType<T>());
        return reinterpret_cast<const T*>(m_Memory);
    }

    const TensorInfo& GetTensorInfo() const
    {
        return m_TensorInfo;
    }

    virtual ITensorHandle::Type GetType() const override
    {
        return ITensorHandle::Cpu;
    }

protected:
    ConstCpuTensorHandle(const TensorInfo& tensorInfo);

    void SetConstMemory(const void* mem) { m_Memory = mem; }

private:
    ConstCpuTensorHandle(const ConstCpuTensorHandle& other) = delete;
    ConstCpuTensorHandle& operator=(const ConstCpuTensorHandle& other) = delete;

    TensorInfo m_TensorInfo;
    const void* m_Memory;
};

// Abstract specialization of ConstCpuTensorHandle that allows write access to the same data
class CpuTensorHandle : public ConstCpuTensorHandle
{
public:
    template <typename T>
    T* GetTensor() const
    {
        BOOST_ASSERT(GetTensorInfo().GetDataType() == GetDataType<T>());
        return reinterpret_cast<T*>(m_MutableMemory);
    }

protected:
    CpuTensorHandle(const TensorInfo& tensorInfo);

    void SetMemory(void* mem)
    {
        m_MutableMemory = mem;
        SetConstMemory(m_MutableMemory);
    }

private:

    CpuTensorHandle(const CpuTensorHandle& other) = delete;
    CpuTensorHandle& operator=(const CpuTensorHandle& other) = delete;
    void* m_MutableMemory;
};

// A CpuTensorHandle that owns the wrapped memory region.
class ScopedCpuTensorHandle : public CpuTensorHandle
{
public:
    explicit ScopedCpuTensorHandle(const TensorInfo& tensorInfo);

    // Copies contents from Tensor
    explicit ScopedCpuTensorHandle(const ConstTensor& tensor);

    ScopedCpuTensorHandle(const ScopedCpuTensorHandle& other);
    ScopedCpuTensorHandle& operator=(const ScopedCpuTensorHandle& other);
    ~ScopedCpuTensorHandle();

    virtual void Allocate() override;

private:
    void CopyFrom(const ScopedCpuTensorHandle& other);
    void CopyFrom(const void* srcMemory, unsigned int numBytes);
};

// A CpuTensorHandle that wraps an already allocated memory region.
//
// Clients must make sure the passed in memory region stays alive for the lifetime of
// the PassthroughCpuTensorHandle instance.
//
// Note there is no polymorphism to/from ConstPassthroughCpuTensorHandle
class PassthroughCpuTensorHandle : public CpuTensorHandle
{
public:
    PassthroughCpuTensorHandle(const TensorInfo& tensorInfo, void* mem)
    :   CpuTensorHandle(tensorInfo)
    {
        SetMemory(mem);
    }

    virtual void Allocate() override;
};

// A ConstCpuTensorHandle that wraps an already allocated memory region.
//
// This allows users to pass in const memory to a network.
// Clients must make sure the passed in memory region stays alive for the lifetime of
// the PassthroughCpuTensorHandle instance.
//
// Note there is no polymorphism to/from PassthroughCpuTensorHandle
class ConstPassthroughCpuTensorHandle : public ConstCpuTensorHandle
{
public:
    ConstPassthroughCpuTensorHandle(const TensorInfo& tensorInfo, const void* mem)
    :   ConstCpuTensorHandle(tensorInfo)
    {
        SetConstMemory(mem);
    }

    virtual void Allocate() override;
};


// template specializations

template <>
const void* ConstCpuTensorHandle::GetConstTensor() const;

template <>
void* CpuTensorHandle::GetTensor() const;

} // namespace armnn
