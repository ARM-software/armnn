//
// Copyright © 2022 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//
#pragma once

#include <arm_compute/core/CL/ICLTensor.h>
#include <arm_compute/core/ITensorInfo.h>

namespace armnn
{

class ICLTensorProxy : public arm_compute::ICLTensor
{
public:
    ICLTensorProxy(arm_compute::ICLTensor* iclTensor) : m_DelegateTensor(iclTensor) {}
    ICLTensorProxy(const ICLTensorProxy&) = delete;
    ICLTensorProxy& operator=(const ICLTensorProxy&) = delete;
    ICLTensorProxy(ICLTensorProxy&&) = default;
    ICLTensorProxy& operator=(ICLTensorProxy&&) = default;

    void set(arm_compute::ICLTensor* iclTensor)
    {
        if(iclTensor != nullptr)
        {
            m_DelegateTensor = iclTensor;
        }
    }

    // Inherited methods overridden:
    arm_compute::ITensorInfo* info() const
    {
        ARM_COMPUTE_ERROR_ON(m_DelegateTensor == nullptr);
        return m_DelegateTensor->info();
    }

    arm_compute::ITensorInfo* info()
    {
        ARM_COMPUTE_ERROR_ON(m_DelegateTensor == nullptr);
        return m_DelegateTensor->info();
    }

    uint8_t* buffer() const
    {
        ARM_COMPUTE_ERROR_ON(m_DelegateTensor == nullptr);
        return m_DelegateTensor->buffer();
    }

    arm_compute::CLQuantization quantization() const
    {
        ARM_COMPUTE_ERROR_ON(m_DelegateTensor == nullptr);
        return m_DelegateTensor->quantization();
    }

    const cl::Buffer& cl_buffer() const
    {
        ARM_COMPUTE_ERROR_ON(m_DelegateTensor == nullptr);
        return m_DelegateTensor->cl_buffer();
    }

protected:
    uint8_t* do_map(cl::CommandQueue& q, bool blocking)
    {
        ARM_COMPUTE_ERROR_ON(m_DelegateTensor == nullptr);
        m_DelegateTensor->map(q, blocking);
        return m_DelegateTensor->buffer();
    }
    void do_unmap(cl::CommandQueue& q)
    {
        ARM_COMPUTE_ERROR_ON(m_DelegateTensor == nullptr);
        return m_DelegateTensor->unmap(q);
    }

private:
    arm_compute::ICLTensor* m_DelegateTensor{ nullptr };
};

} //namespace armnn