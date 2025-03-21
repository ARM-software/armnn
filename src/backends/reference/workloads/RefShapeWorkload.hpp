//
// Copyright © 2021-2024 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include "RefBaseWorkload.hpp"
#include <armnn/backends/WorkloadData.hpp>

#include "RefWorkloadUtils.hpp"

namespace armnn
{

struct RefShapeWorkload : public RefBaseWorkload<ShapeQueueDescriptor>
{
public:
    using RefBaseWorkload<ShapeQueueDescriptor>::RefBaseWorkload;
    virtual void Execute() const override
    {
        Execute(m_Data.m_Inputs, m_Data.m_Outputs);
    }

private:
    void Execute(std::vector<ITensorHandle*> inputs, std::vector<ITensorHandle*> outputs) const
    {
        ARMNN_SCOPED_PROFILING_EVENT_REF_NAME_GUID("RefShapeWorkload_Execute");

        const TensorShape Shape = GetTensorInfo(inputs[0]).GetShape();

        const TensorInfo& outputInfo = GetTensorInfo(outputs[0]);

        unsigned int numBytes =
            GetTensorInfo(inputs[0]).GetNumDimensions() * GetDataTypeSize(outputInfo.GetDataType());

        std::memcpy(outputs[0]->Map(), &Shape, numBytes);
        outputs[0]->Unmap();
    }
};

} //namespace armnn




