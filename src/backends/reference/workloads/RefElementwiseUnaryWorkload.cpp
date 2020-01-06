//
// Copyright © 2019 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "RefElementwiseUnaryWorkload.hpp"

#include "Decoders.hpp"
#include "ElementwiseFunction.hpp"
#include "Encoders.hpp"
#include "RefWorkloadUtils.hpp"
#include "Abs.hpp"
#include "Exp.hpp"
#include "Rsqrt.hpp"
#include "Sqrt.hpp"

#include <Profiling.hpp>

#include <armnn/TypesUtils.hpp>

#include <functional>

namespace armnn
{

RefElementwiseUnaryWorkload::RefElementwiseUnaryWorkload(const ElementwiseUnaryQueueDescriptor& desc,
                                                         const WorkloadInfo& info)
    : BaseWorkload<ElementwiseUnaryQueueDescriptor>(desc, info)
{}

void RefElementwiseUnaryWorkload::PostAllocationConfigure()
{
    const TensorInfo& inputInfo = GetTensorInfo(m_Data.m_Inputs[0]);
    const TensorInfo& outputInfo = GetTensorInfo(m_Data.m_Outputs[0]);

    m_Input = MakeDecoder<InType>(inputInfo);

    m_Output = MakeEncoder<OutType>(outputInfo);
}

void RefElementwiseUnaryWorkload::Execute() const
{
    ARMNN_SCOPED_PROFILING_EVENT(Compute::CpuRef, "RefElementwiseUnaryWorkload_Execute");

    const TensorInfo& inputInfo = GetTensorInfo(m_Data.m_Inputs[0]);
    const TensorInfo& outputInfo = GetTensorInfo(m_Data.m_Outputs[0]);

    const TensorShape& inShape = inputInfo.GetShape();
    const TensorShape& outShape = outputInfo.GetShape();

    m_Input->Reset(m_Data.m_Inputs[0]->Map());
    m_Output->Reset(m_Data.m_Outputs[0]->Map());

    using AbsFunction   = ElementwiseUnaryFunction<abs<InType>>;
    using ExpFunction   = ElementwiseUnaryFunction<exp<InType>>;
    using NegFunction   = ElementwiseUnaryFunction<std::negate<InType>>;
    using RsqrtFunction = ElementwiseUnaryFunction<rsqrt<InType>>;
    using SqrtFunction  = ElementwiseUnaryFunction<sqrt<InType>>;

    switch (m_Data.m_Parameters.m_Operation)
    {
        case UnaryOperation::Abs:
        {
            AbsFunction(inShape, outShape, *m_Input, *m_Output);
            break;
        }
        case UnaryOperation::Exp:
        {
            ExpFunction(inShape, outShape, *m_Input, *m_Output);
            break;
        }
        case UnaryOperation::Neg:
        {
            NegFunction(inShape, outShape, *m_Input, *m_Output);
            break;
        }
        case UnaryOperation::Rsqrt:
        {
            RsqrtFunction(inShape, outShape, *m_Input, *m_Output);
            break;
        }
        case UnaryOperation::Sqrt:
        {
            SqrtFunction(inShape, outShape, *m_Input, *m_Output);
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
