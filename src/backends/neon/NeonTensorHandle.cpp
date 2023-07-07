//
// Copyright © 2023 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "NeonTensorHandle.hpp"

#include <aclCommon/ArmComputeTensorUtils.hpp>

namespace armnn
{
std::shared_ptr<ITensorHandle> NeonTensorHandle::DecorateTensorHandle(const TensorInfo &tensorInfo)
{
    auto* parent = const_cast<NeonTensorHandle*>(this);
    auto decorated = std::make_shared<NeonTensorHandleDecorator>(parent, tensorInfo);
    m_Decorated.emplace_back(decorated);
    return decorated;
}

NeonTensorDecorator::NeonTensorDecorator()
        : m_Original(nullptr), m_TensorInfo()
{
}

NeonTensorDecorator::NeonTensorDecorator(arm_compute::ITensor *parent, const TensorInfo& tensorInfo)
        : m_Original(nullptr), m_TensorInfo()
{
    m_TensorInfo = armcomputetensorutils::BuildArmComputeTensorInfo(tensorInfo);
    m_Original = parent;
}

arm_compute::ITensorInfo *NeonTensorDecorator::info() const
{
    return &m_TensorInfo;
}

arm_compute::ITensorInfo *NeonTensorDecorator::info()
{
    return &m_TensorInfo;
}

uint8_t *NeonTensorDecorator::buffer() const
{
    return m_Original->buffer();
}

}