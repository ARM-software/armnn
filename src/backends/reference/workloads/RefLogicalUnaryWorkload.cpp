//
// Copyright © 2022 Arm Ltd and Contributors. All rights reserved.
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
    : RefBaseWorkload<ElementwiseUnaryQueueDescriptor>(desc, info)
{}

void RefLogicalUnaryWorkload::Execute() const
{
    Execute(m_Data.m_Inputs, m_Data.m_Outputs);
}

void RefLogicalUnaryWorkload::ExecuteAsync(ExecutionData& executionData)
{
    WorkingMemDescriptor* workingMemDescriptor = static_cast<WorkingMemDescriptor*>(executionData.m_Data);
    Execute(workingMemDescriptor->m_Inputs, workingMemDescriptor->m_Outputs);
}

void RefLogicalUnaryWorkload::Execute(std::vector<ITensorHandle*> inputs, std::vector<ITensorHandle*> outputs) const
{
    ARMNN_SCOPED_PROFILING_EVENT(Compute::CpuRef, "RefLogicalUnaryWorkload_Execute");

    const TensorInfo& inputInfo = GetTensorInfo(inputs[0]);
    const TensorInfo& outputInfo = GetTensorInfo(outputs[0]);

    const TensorShape& inShape = inputInfo.GetShape();
    const TensorShape& outShape = outputInfo.GetShape();

    std::unique_ptr<Decoder<InType>>  input = MakeDecoder<InType>(inputInfo, inputs[0]->Map());
    std::unique_ptr<Encoder<OutType>> output = MakeEncoder<OutType>(outputInfo, outputs[0]->Map());

    using NotFunction = LogicalUnaryFunction<std::logical_not<bool>>;

    switch (m_Data.m_Parameters.m_Operation)
    {
        case UnaryOperation::LogicalNot:
        {
            NotFunction(inShape, outShape, *input, *output);
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
