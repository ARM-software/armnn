//
// Copyright © 2021 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//
#include <armnn/Exceptions.hpp>
#include <armnn/utility/IgnoreUnused.hpp>

#include <armnn/backends/TensorHandle.hpp>

#include <cstring>

namespace armnn
{

TensorShape GetUnpaddedTensorStrides(const TensorInfo& tensorInfo)
{
    TensorShape shape(tensorInfo.GetShape());
    auto size = GetDataTypeSize(tensorInfo.GetDataType());
    auto runningSize = size;
    std::vector<unsigned int> strides(shape.GetNumDimensions());
    auto lastIdx = shape.GetNumDimensions()-1;
    for (unsigned int i=0; i < lastIdx ; i++)
    {
        strides[lastIdx-i] = runningSize;
        runningSize *= shape[lastIdx-i];
    }
    strides[0] = runningSize;
    return TensorShape(shape.GetNumDimensions(), strides.data());
}

ConstTensorHandle::ConstTensorHandle(const TensorInfo& tensorInfo)
: m_TensorInfo(tensorInfo)
, m_Memory(nullptr)
{
}

template <>
const void* ConstTensorHandle::GetConstTensor<void>() const
{
    return m_Memory;
}

TensorHandle::TensorHandle(const TensorInfo& tensorInfo)
: ConstTensorHandle(tensorInfo)
, m_MutableMemory(nullptr)
{
}

template <>
void* TensorHandle::GetTensor<void>() const
{
    return m_MutableMemory;
}

ScopedTensorHandle::ScopedTensorHandle(const TensorInfo& tensorInfo)
: TensorHandle(tensorInfo)
{
}

ScopedTensorHandle::ScopedTensorHandle(const ConstTensor& tensor)
: ScopedTensorHandle(tensor.GetInfo())
{
    CopyFrom(tensor.GetMemoryArea(), tensor.GetNumBytes());
}

ScopedTensorHandle::ScopedTensorHandle(const ConstTensorHandle& tensorHandle)
: ScopedTensorHandle(tensorHandle.GetTensorInfo())
{
    CopyFrom(tensorHandle.GetConstTensor<void>(), tensorHandle.GetTensorInfo().GetNumBytes());
}

ScopedTensorHandle::ScopedTensorHandle(const ScopedTensorHandle& other)
: TensorHandle(other.GetTensorInfo())
{
    CopyFrom(other);
}

ScopedTensorHandle& ScopedTensorHandle::operator=(const ScopedTensorHandle& other)
{
    ::operator delete(GetTensor<void>());
    SetMemory(nullptr);
    CopyFrom(other);
    return *this;
}

ScopedTensorHandle::~ScopedTensorHandle()
{
    ::operator delete(GetTensor<void>());
}

void ScopedTensorHandle::Allocate()
{
    if (GetTensor<void>() == nullptr)
    {
        SetMemory(::operator new(GetTensorInfo().GetNumBytes()));
    }
    else
    {
        throw InvalidArgumentException("TensorHandle::Allocate Trying to allocate a TensorHandle"
            "that already has allocated memory.");
    }
}

void ScopedTensorHandle::CopyOutTo(void* memory) const
{
    const void* src = GetTensor<void>();
    if (src == nullptr)
    {
        throw NullPointerException("TensorHandle::CopyOutTo called with a null src pointer");
    }
    if (memory == nullptr)
    {
        throw NullPointerException("TensorHandle::CopyOutTo called with a null dest pointer");
    }
    memcpy(memory, src, GetTensorInfo().GetNumBytes());
}

void ScopedTensorHandle::CopyInFrom(const void* memory)
{
    void* dest = GetTensor<void>();
    if (dest == nullptr)
    {
        throw NullPointerException("TensorHandle::CopyInFrom called with a null dest pointer");
    }
    if (memory == nullptr)
    {
        throw NullPointerException("TensorHandle::CopyInFrom called with a null src pointer");
    }
    memcpy(dest, memory, GetTensorInfo().GetNumBytes());
}

void ScopedTensorHandle::CopyFrom(const ScopedTensorHandle& other)
{
    CopyFrom(other.GetTensor<void>(), other.GetTensorInfo().GetNumBytes());
}

void ScopedTensorHandle::CopyFrom(const void* srcMemory, unsigned int numBytes)
{
    if (GetTensor<void>() != nullptr)
    {
        throw NullPointerException("TensorHandle::CopyFrom called on an already allocated TensorHandle");
    }
    if (GetTensorInfo().GetNumBytes() != numBytes)
    {
        std::stringstream msg;
        msg << "TensorHandle:CopyFrom: Number of bytes in the tensor info (" << GetTensorInfo().GetNumBytes() <<
            ") does not match the number of bytes being copied (" << numBytes << ")";
        throw armnn::Exception(msg.str());
    }

    if (srcMemory)
    {
        Allocate();
        memcpy(GetTensor<void>(), srcMemory, numBytes);
    }
}

void PassthroughTensorHandle::Allocate()
{
    throw InvalidArgumentException("PassthroughTensorHandle::Allocate() should never be called");
}

void ConstPassthroughTensorHandle::Allocate()
{
    throw InvalidArgumentException("ConstPassthroughTensorHandle::Allocate() should never be called");
}

} // namespace armnn
