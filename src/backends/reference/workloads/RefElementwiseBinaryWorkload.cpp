//
// Copyright © 2023-2024 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "RefElementwiseBinaryWorkload.hpp"

#include "Decoders.hpp"
#include "ElementwiseFunction.hpp"
#include "Encoders.hpp"
#include "RefWorkloadUtils.hpp"
#include "Maximum.hpp"
#include "Minimum.hpp"
#include "SquaredDifference.hpp"
#include "Power.hpp"

#include <Profiling.hpp>

#include <armnn/TypesUtils.hpp>

#include <functional>

namespace armnn
{

template<typename DataType>
void ExecuteFunction(std::vector<ITensorHandle*> inputs,
                     std::vector<ITensorHandle*> outputs,
                     BinaryOperation operation,
                     const std::string& layerName = "")
{
    const TensorInfo& inputInfo0 = GetTensorInfo(inputs[0]);
    const TensorInfo& inputInfo1 = GetTensorInfo(inputs[1]);
    const TensorInfo& outputInfo = GetTensorInfo(outputs[0]);

    const TensorShape& inShape0 = inputInfo0.GetShape();
    const TensorShape& inShape1 = inputInfo1.GetShape();
    const TensorShape& outShape = outputInfo.GetShape();

    std::unique_ptr<Decoder<DataType>> input0 = MakeDecoder<DataType>(inputInfo0, inputs[0]->Map());
    std::unique_ptr<Decoder<DataType>> input1 = MakeDecoder<DataType>(inputInfo1, inputs[1]->Map());
    std::unique_ptr<Encoder<DataType>> output = MakeEncoder<DataType>(outputInfo, outputs[0]->Map());

    using AddFunction     = ElementwiseBinaryFunction<std::plus<DataType>>;
    using DivFunction     = ElementwiseBinaryFunction<std::divides<DataType>>;
    using FloorDivFunction = ElementwiseBinaryFunction<armnn::floorDiv<DataType>>;
    using MaximumFunction = ElementwiseBinaryFunction<armnn::maximum<DataType>>;
    using MinimumFunction = ElementwiseBinaryFunction<armnn::minimum<DataType>>;
    using MulFunction     = ElementwiseBinaryFunction<std::multiplies<DataType>>;
    using SubFunction     = ElementwiseBinaryFunction<std::minus<DataType>>;
    using SqDiffFunction  = ElementwiseBinaryFunction<armnn::squaredDifference<DataType>>;
    using PowerFunction   = ElementwiseBinaryFunction<armnn::power<DataType>>;


    switch (operation)
    {
        case BinaryOperation::Add:
        {
            AddFunction(inShape0, inShape1, outShape, *input0, *input1, *output);
            break;
        }
        case BinaryOperation::Div:
        {
            if(!layerName.empty() && layerName.find("FloorDiv") != std::string::npos)
            {
                FloorDivFunction(inShape0, inShape1, outShape, *input0, *input1, *output);
            }
            else
            {
                DivFunction(inShape0, inShape1, outShape, *input0, *input1, *output);
            }
            break;
        }
        case BinaryOperation::Maximum:
        {
            MaximumFunction(inShape0, inShape1, outShape, *input0, *input1, *output);
            break;
        }
        case BinaryOperation::Minimum:
        {
            MinimumFunction(inShape0, inShape1, outShape, *input0, *input1, *output);
            break;
        }
        case BinaryOperation::Mul:
        {
            MulFunction(inShape0, inShape1, outShape, *input0, *input1, *output);
            break;
        }
        case BinaryOperation::Sub:
        {
            SubFunction(inShape0, inShape1, outShape, *input0, *input1, *output);
            break;
        }
        case BinaryOperation::SqDiff:
        {
            SqDiffFunction(inShape0, inShape1, outShape, *input0, *input1, *output);
            break;
        }
        case BinaryOperation::Power:
        {
            PowerFunction(inShape0, inShape1, outShape, *input0, *input1, *output);
            break;
        }
        default:
        {
            throw InvalidArgumentException(std::string("Unsupported binary operation ") +
                                           GetBinaryOperationAsCString(operation), CHECK_LOCATION());
        }
    }
}

RefElementwiseBinaryWorkload::RefElementwiseBinaryWorkload(const ElementwiseBinaryQueueDescriptor& desc,
                                                         const WorkloadInfo& info)
    : RefBaseWorkload<ElementwiseBinaryQueueDescriptor>(desc, info)
{}

void RefElementwiseBinaryWorkload::Execute() const
{
    Execute(m_Data.m_Inputs, m_Data.m_Outputs);
}

void RefElementwiseBinaryWorkload::ExecuteAsync(ExecutionData& executionData)
{

    WorkingMemDescriptor* workingMemDescriptor = static_cast<WorkingMemDescriptor*>(executionData.m_Data);
    Execute(workingMemDescriptor->m_Inputs, workingMemDescriptor->m_Outputs);
}

void RefElementwiseBinaryWorkload::Execute(std::vector<ITensorHandle*> inputs,
                                           std::vector<ITensorHandle*> outputs) const
{
    ARMNN_SCOPED_PROFILING_EVENT_REF_NAME_GUID("RefElementwiseBinaryWorkload_Execute");

    if (GetTensorInfo(inputs[0]).GetDataType() == DataType::Signed32)
    {
        ExecuteFunction<int32_t>(inputs, outputs, m_Data.m_Parameters.m_Operation, m_Name);
    }
    else
    {
        ExecuteFunction<float>(inputs, outputs, m_Data.m_Parameters.m_Operation, m_Name);
    }
}

} // namespace armnn
