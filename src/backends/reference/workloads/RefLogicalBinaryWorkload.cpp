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
    : RefBaseWorkload<LogicalBinaryQueueDescriptor>(desc, info)
{}

void RefLogicalBinaryWorkload::Execute() const
{
    Execute(m_Data.m_Inputs, m_Data.m_Outputs);
}

void RefLogicalBinaryWorkload::ExecuteAsync(WorkingMemDescriptor &workingMemDescriptor)
{
    Execute(workingMemDescriptor.m_Inputs, workingMemDescriptor.m_Outputs);
}

void RefLogicalBinaryWorkload::Execute(std::vector<ITensorHandle*> inputs, std::vector<ITensorHandle*> outputs) const
{
    ARMNN_SCOPED_PROFILING_EVENT(Compute::CpuRef, "RefLogicalBinaryWorkload_Execute");

    const TensorInfo& inputInfo0 = GetTensorInfo(inputs[0]);
    const TensorInfo& inputInfo1 = GetTensorInfo(inputs[1]);
    const TensorInfo& outputInfo = GetTensorInfo(outputs[0]);

    const TensorShape& inShape0 = inputInfo0.GetShape();
    const TensorShape& inShape1 = inputInfo1.GetShape();
    const TensorShape& outShape = outputInfo.GetShape();

    std::unique_ptr<Decoder<InType>>  input0 = MakeDecoder<InType>(inputInfo0, inputs[0]->Map());
    std::unique_ptr<Decoder<InType>>  input1 = MakeDecoder<InType>(inputInfo1, inputs[1]->Map());
    std::unique_ptr<Encoder<OutType>> output = MakeEncoder<OutType>(outputInfo, outputs[0]->Map());

    using AndFunction = LogicalBinaryFunction<std::logical_and<bool>>;
    using OrFunction  = LogicalBinaryFunction<std::logical_or<bool>>;

    switch (m_Data.m_Parameters.m_Operation)
    {
        case LogicalBinaryOperation::LogicalAnd:
        {
            AndFunction(inShape0, inShape1, outShape, *input0, *input1, *output);
            break;
        }
        case LogicalBinaryOperation::LogicalOr:
        {
            OrFunction(inShape0, inShape1, outShape, *input0, *input1, *output);
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
