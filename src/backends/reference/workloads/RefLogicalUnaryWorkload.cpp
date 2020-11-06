//
// Copyright © 2020 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "RefLogicalUnaryWorkload.hpp"

#include "Decoders.hpp"
#include "ElementwiseFunction.hpp"
#include "Encoders.hpp"
#include "RefWorkloadUtils.hpp"

#include <Profiling.hpp>

#include <armnn/TypesUtils.hpp>

namespace armnn
{

RefLogicalUnaryWorkload::RefLogicalUnaryWorkload(const ElementwiseUnaryQueueDescriptor& desc,
                                                 const WorkloadInfo& info)
    : BaseWorkload<ElementwiseUnaryQueueDescriptor>(desc, info)
{}

void RefLogicalUnaryWorkload::PostAllocationConfigure()
{
    const TensorInfo& inputInfo = GetTensorInfo(m_Data.m_Inputs[0]);
    const TensorInfo& outputInfo = GetTensorInfo(m_Data.m_Outputs[0]);

    m_Input = MakeDecoder<InType>(inputInfo);
    m_Output = MakeEncoder<OutType>(outputInfo);
}

void RefLogicalUnaryWorkload::Execute() const
{
    ARMNN_SCOPED_PROFILING_EVENT(Compute::CpuRef, "RefLogicalUnaryWorkload_Execute");

    const TensorInfo& inputInfo = GetTensorInfo(m_Data.m_Inputs[0]);
    const TensorInfo& outputInfo = GetTensorInfo(m_Data.m_Outputs[0]);

    const TensorShape& inShape = inputInfo.GetShape();
    const TensorShape& outShape = outputInfo.GetShape();

    m_Input->Reset(m_Data.m_Inputs[0]->Map());
    m_Output->Reset(m_Data.m_Outputs[0]->Map());

    using NotFunction = LogicalUnaryFunction<std::logical_not<bool>>;

    switch (m_Data.m_Parameters.m_Operation)
    {
        case UnaryOperation::LogicalNot:
        {
            NotFunction(inShape, outShape, *m_Input, *m_Output);
            break;
        }
        default:
        {
            throw InvalidArgumentException(std::string("Unsupported Logical Unary operation") +
                GetUnaryOperationAsCString(m_Data.m_Parameters.m_Operation), CHECK_LOCATION());
        }
    }
}

} // namespace armnn
