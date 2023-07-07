//
// Copyright © 2023 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "ClTensorHandle.hpp"

#include <aclCommon/ArmComputeTensorUtils.hpp>

namespace armnn
{
    std::shared_ptr<ITensorHandle> ClTensorHandle::DecorateTensorHandle(const TensorInfo& tensorInfo)
    {
        auto* parent = const_cast<ClTensorHandle*>(this);
        auto decorated = std::make_shared<ClTensorHandleDecorator>(parent, tensorInfo);
        m_Decorated.emplace_back(decorated);
        return decorated;
    }

    ClTensorDecorator::ClTensorDecorator()
    : m_Original(nullptr), m_TensorInfo()
    {
    }

    ClTensorDecorator::ClTensorDecorator(arm_compute::ICLTensor* original, const TensorInfo& tensorInfo)
    : m_Original(nullptr), m_TensorInfo()
    {
        m_TensorInfo = armcomputetensorutils::BuildArmComputeTensorInfo(tensorInfo);
        m_Original = original;
    }

    arm_compute::ITensorInfo* ClTensorDecorator::info() const
    {
        return &m_TensorInfo;
    }

    arm_compute::ITensorInfo* ClTensorDecorator::info()
    {
        return &m_TensorInfo;
    }

    const cl::Buffer& ClTensorDecorator::cl_buffer() const
    {
        ARM_COMPUTE_ERROR_ON(m_Original == nullptr);
        return m_Original->cl_buffer();
    }

    arm_compute::ICLTensor* ClTensorDecorator::parent()
    {
        return nullptr;
    }

    arm_compute::CLQuantization ClTensorDecorator::quantization() const
    {
        return m_Original->quantization();
    }

    void ClTensorDecorator::map(bool blocking)
    {
        arm_compute::ICLTensor::map(arm_compute::CLScheduler::get().queue(), blocking);
    }

    void ClTensorDecorator::unmap()
    {
        arm_compute::ICLTensor::unmap(arm_compute::CLScheduler::get().queue());
    }

    uint8_t* ClTensorDecorator::do_map(cl::CommandQueue& q, bool blocking)
    {
        if(m_Original->buffer() == nullptr)
        {
            m_Original->map(q, blocking);
        }
        return m_Original->buffer();
    }

    void ClTensorDecorator::do_unmap(cl::CommandQueue& q)
    {
        m_Original->unmap(q);
    }

}