//
// Copyright © 2019 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "RefComparisonWorkload.hpp"

#include "Decoders.hpp"
#include "ElementwiseFunction.hpp"
#include "Encoders.hpp"
#include "RefWorkloadUtils.hpp"

#include <Profiling.hpp>

#include <armnn/TypesUtils.hpp>

#include <functional>

namespace armnn
{

RefComparisonWorkload::RefComparisonWorkload(const ComparisonQueueDescriptor& desc,
                                             const WorkloadInfo& info)
    : RefBaseWorkload<ComparisonQueueDescriptor>(desc, info)
{}

void RefComparisonWorkload::PostAllocationConfigure()
{
    PostAllocationConfigure(m_Data.m_Inputs, m_Data.m_Outputs);
}

void RefComparisonWorkload::PostAllocationConfigure(std::vector<ITensorHandle*> inputs,
                                                    std::vector<ITensorHandle*> outputs)
{
    const TensorInfo& inputInfo0 = GetTensorInfo(inputs[0]);
    const TensorInfo& inputInfo1 = GetTensorInfo(inputs[1]);
    const TensorInfo& outputInfo = GetTensorInfo(outputs[0]);

    m_Input0 = MakeDecoder<InType>(inputInfo0);
    m_Input1 = MakeDecoder<InType>(inputInfo1);

    m_Output = MakeEncoder<OutType>(outputInfo);
}

void RefComparisonWorkload::Execute() const
{
    Execute(m_Data.m_Inputs, m_Data.m_Outputs);
}

void RefComparisonWorkload::ExecuteAsync(WorkingMemDescriptor &workingMemDescriptor)
{
    PostAllocationConfigure(workingMemDescriptor.m_Inputs, workingMemDescriptor.m_Outputs);

    Execute(workingMemDescriptor.m_Inputs, workingMemDescriptor.m_Outputs);
}

void RefComparisonWorkload::Execute(std::vector<ITensorHandle*> inputs, std::vector<ITensorHandle*> outputs) const
{
    ARMNN_SCOPED_PROFILING_EVENT(Compute::CpuRef, "RefComparisonWorkload_Execute");

    const TensorInfo& inputInfo0 = GetTensorInfo(inputs[0]);
    const TensorInfo& inputInfo1 = GetTensorInfo(inputs[1]);
    const TensorInfo& outputInfo = GetTensorInfo(outputs[0]);

    const TensorShape& inShape0 = inputInfo0.GetShape();
    const TensorShape& inShape1 = inputInfo1.GetShape();
    const TensorShape& outShape = outputInfo.GetShape();

    m_Input0->Reset(inputs[0]->Map());
    m_Input1->Reset(inputs[1]->Map());
    m_Output->Reset(outputs[0]->Map());

    using EqualFunction          = ElementwiseBinaryFunction<std::equal_to<InType>>;
    using GreaterFunction        = ElementwiseBinaryFunction<std::greater<InType>>;
    using GreaterOrEqualFunction = ElementwiseBinaryFunction<std::greater_equal<InType>>;
    using LessFunction           = ElementwiseBinaryFunction<std::less<InType>>;
    using LessOrEqualFunction    = ElementwiseBinaryFunction<std::less_equal<InType>>;
    using NotEqualFunction       = ElementwiseBinaryFunction<std::not_equal_to<InType>>;

    switch (m_Data.m_Parameters.m_Operation)
    {
        case ComparisonOperation::Equal:
        {
            EqualFunction(inShape0, inShape1, outShape, *m_Input0, *m_Input1, *m_Output);
            break;
        }
        case ComparisonOperation::Greater:
        {
            GreaterFunction(inShape0, inShape1, outShape, *m_Input0, *m_Input1, *m_Output);
            break;
        }
        case ComparisonOperation::GreaterOrEqual:
        {
            GreaterOrEqualFunction(inShape0, inShape1, outShape, *m_Input0, *m_Input1, *m_Output);
            break;
        }
        case ComparisonOperation::Less:
        {
            LessFunction(inShape0, inShape1, outShape, *m_Input0, *m_Input1, *m_Output);
            break;
        }
        case ComparisonOperation::LessOrEqual:
        {
            LessOrEqualFunction(inShape0, inShape1, outShape, *m_Input0, *m_Input1, *m_Output);
            break;
        }
        case ComparisonOperation::NotEqual:
        {
            NotEqualFunction(inShape0, inShape1, outShape, *m_Input0, *m_Input1, *m_Output);
            break;
        }
        default:
        {
            throw InvalidArgumentException(std::string("Unsupported comparison operation ") +
                GetComparisonOperationAsCString(m_Data.m_Parameters.m_Operation), CHECK_LOCATION());
        }
    }
}

} // namespace armnn
