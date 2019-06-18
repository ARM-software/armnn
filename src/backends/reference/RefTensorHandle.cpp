//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//
#include "RefTensorHandle.hpp"

namespace armnn
{

RefTensorHandle::RefTensorHandle(const TensorInfo &tensorInfo):
    m_TensorInfo(tensorInfo),
    m_Memory(nullptr)
{

}

RefTensorHandle::~RefTensorHandle()
{
    ::operator delete(m_Memory);
}

void RefTensorHandle::Allocate()
{
    if (m_Memory == nullptr)
    {
        m_Memory = ::operator new(m_TensorInfo.GetNumBytes());
    }
    else
    {
        throw InvalidArgumentException("RefTensorHandle::Allocate Trying to allocate a RefTensorHandle"
                                           "that already has allocated memory.");
    }
}

void RefTensorHandle::CopyOutTo(void* memory) const
{
    memcpy(memory, m_Memory, m_TensorInfo.GetNumBytes());
}

void RefTensorHandle::CopyInFrom(const void* memory)
{
    memcpy(m_Memory, memory, m_TensorInfo.GetNumBytes());
}

}