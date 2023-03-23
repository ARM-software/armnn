//
// Copyright © 2020-2023 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "RefElementwiseUnaryWorkload.hpp"

#include "Decoders.hpp"
#include "ElementwiseFunction.hpp"
#include "Encoders.hpp"
#include "RefWorkloadUtils.hpp"
#include "Abs.hpp"
#include "Ceil.hpp"
#include "Exp.hpp"
#include "Log.hpp"
#include "Rsqrt.hpp"
#include "Sin.hpp"
#include "Sqrt.hpp"

#include <Profiling.hpp>

#include <armnn/TypesUtils.hpp>

#include <functional>

namespace armnn
{

RefElementwiseUnaryWorkload::RefElementwiseUnaryWorkload(const ElementwiseUnaryQueueDescriptor& desc,
                                                         const WorkloadInfo& info)
    : RefBaseWorkload<ElementwiseUnaryQueueDescriptor>(desc, info)
{}

void RefElementwiseUnaryWorkload::Execute() const
{
    Execute(m_Data.m_Inputs, m_Data.m_Outputs);
}

void RefElementwiseUnaryWorkload::ExecuteAsync(ExecutionData& executionData)
{

    WorkingMemDescriptor* workingMemDescriptor = static_cast<WorkingMemDescriptor*>(executionData.m_Data);
    Execute(workingMemDescriptor->m_Inputs, workingMemDescriptor->m_Outputs);
}

void RefElementwiseUnaryWorkload::Execute(std::vector<ITensorHandle*> inputs, std::vector<ITensorHandle*> outputs) const
{
    ARMNN_SCOPED_PROFILING_EVENT(Compute::CpuRef, "RefElementwiseUnaryWorkload_Execute");

    const TensorInfo& inputInfo = GetTensorInfo(inputs[0]);
    const TensorInfo& outputInfo = GetTensorInfo(outputs[0]);

    const TensorShape& inShape = inputInfo.GetShape();
    const TensorShape& outShape = outputInfo.GetShape();

    std::unique_ptr<Decoder<InType>>  input = MakeDecoder<InType>(inputInfo, inputs[0]->Map());
    std::unique_ptr<Encoder<OutType>> output= MakeEncoder<OutType>(outputInfo, outputs[0]->Map());

    using AbsFunction   = ElementwiseUnaryFunction<abs<InType>>;
    using CeilFunction  = ElementwiseUnaryFunction<ceil<InType>>;
    using ExpFunction   = ElementwiseUnaryFunction<exp<InType>>;
    using LogFunction   = ElementwiseUnaryFunction<log<InType>>;
    using NegFunction   = ElementwiseUnaryFunction<std::negate<InType>>;
    using RsqrtFunction = ElementwiseUnaryFunction<rsqrt<InType>>;
    using SinFunction   = ElementwiseUnaryFunction<sin<InType>>;
    using SqrtFunction  = ElementwiseUnaryFunction<sqrt<InType>>;

    switch (m_Data.m_Parameters.m_Operation)
    {
        case UnaryOperation::Abs:
        {
            AbsFunction(inShape, outShape, *input, *output);
            break;
        }
        case UnaryOperation::Ceil:
        {
            CeilFunction(inShape, outShape, *input, *output);
            break;
        }
        case UnaryOperation::Exp:
        {
            ExpFunction(inShape, outShape, *input, *output);
            break;
        }
        case UnaryOperation::Log:
        {
            LogFunction(inShape, outShape, *input, *output);
            break;
        }
        case UnaryOperation::Neg:
        {
            NegFunction(inShape, outShape, *input, *output);
            break;
        }
        case UnaryOperation::Rsqrt:
        {
            RsqrtFunction(inShape, outShape, *input, *output);
            break;
        }
        case UnaryOperation::Sin:
        {
            SinFunction(inShape, outShape, *input, *output);
            break;
        }
        case UnaryOperation::Sqrt:
        {
            SqrtFunction(inShape, outShape, *input, *output);
            break;
        }
        default:
        {
            throw InvalidArgumentException(std::string("Unsupported unary operation ") +
                GetUnaryOperationAsCString(m_Data.m_Parameters.m_Operation), CHECK_LOCATION());
        }
    }
}

} // namespace armnn
