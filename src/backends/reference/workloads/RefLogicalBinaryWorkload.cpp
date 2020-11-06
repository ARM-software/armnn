//
// Copyright © 2020 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "RefLogicalBinaryWorkload.hpp"

#include "Decoders.hpp"
#include "ElementwiseFunction.hpp"
#include "Encoders.hpp"
#include "RefWorkloadUtils.hpp"

#include <Profiling.hpp>

#include <armnn/TypesUtils.hpp>

namespace armnn
{

RefLogicalBinaryWorkload::RefLogicalBinaryWorkload(const LogicalBinaryQueueDescriptor& desc,
                                                   const WorkloadInfo& info)
    : BaseWorkload<LogicalBinaryQueueDescriptor>(desc, info)
{}

void RefLogicalBinaryWorkload::PostAllocationConfigure()
{
    const TensorInfo& inputInfo0 = GetTensorInfo(m_Data.m_Inputs[0]);
    const TensorInfo& inputInfo1 = GetTensorInfo(m_Data.m_Inputs[1]);
    const TensorInfo& outputInfo = GetTensorInfo(m_Data.m_Outputs[0]);

    m_Input0 = MakeDecoder<InType>(inputInfo0);
    m_Input1 = MakeDecoder<InType>(inputInfo1);
    m_Output = MakeEncoder<OutType>(outputInfo);
}

void RefLogicalBinaryWorkload::Execute() const
{
    ARMNN_SCOPED_PROFILING_EVENT(Compute::CpuRef, "RefLogicalBinaryWorkload_Execute");

    const TensorInfo& inputInfo0 = GetTensorInfo(m_Data.m_Inputs[0]);
    const TensorInfo& inputInfo1 = GetTensorInfo(m_Data.m_Inputs[1]);
    const TensorInfo& outputInfo = GetTensorInfo(m_Data.m_Outputs[0]);

    const TensorShape& inShape0 = inputInfo0.GetShape();
    const TensorShape& inShape1 = inputInfo1.GetShape();
    const TensorShape& outShape = outputInfo.GetShape();

    m_Input0->Reset(m_Data.m_Inputs[0]->Map());
    m_Input1->Reset(m_Data.m_Inputs[1]->Map());
    m_Output->Reset(m_Data.m_Outputs[0]->Map());

    using AndFunction = LogicalBinaryFunction<std::logical_and<bool>>;
    using OrFunction  = LogicalBinaryFunction<std::logical_or<bool>>;

    switch (m_Data.m_Parameters.m_Operation)
    {
        case LogicalBinaryOperation::LogicalAnd:
        {
            AndFunction(inShape0, inShape1, outShape, *m_Input0, *m_Input1, *m_Output);
            break;
        }
        case LogicalBinaryOperation::LogicalOr:
        {
            OrFunction(inShape0, inShape1, outShape, *m_Input0, *m_Input1, *m_Output);
            break;
        }
        default:
        {
            throw InvalidArgumentException(std::string("Unsupported Logical Binary operation") +
                GetLogicalBinaryOperationAsCString(m_Data.m_Parameters.m_Operation), CHECK_LOCATION());
        }
    }
}

} // namespace armnn
