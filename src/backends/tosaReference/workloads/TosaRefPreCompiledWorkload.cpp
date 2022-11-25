//
// Copyright © 2022 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "TosaRefPreCompiledWorkload.hpp"

namespace armnn
{

TosaRefPreCompiledWorkload::TosaRefPreCompiledWorkload(const PreCompiledQueueDescriptor& descriptor,
                                                       const WorkloadInfo& info)
    : BaseWorkload<PreCompiledQueueDescriptor>(descriptor, info)
    , m_workloadInfo(info)
{
    // Check that the workload is holding a pointer to a valid pre-compiled object
    if (m_Data.m_PreCompiledObject == nullptr)
    {
        throw InvalidArgumentException(
                "TosaRefPreCompiledWorkload requires a valid pre-compiled object (TosaSerializationHandler).");
    }
}

void TosaRefPreCompiledWorkload::Execute() const
{
    tosa::TosaSerializationHandler* handler = static_cast<tosa::TosaSerializationHandler*>(m_Data.m_PreCompiledObject);

    std::vector<std::string> inputNames = handler->GetInputs();
    std::vector<std::string> outputNames = handler->GetOutputs();

    TosaReference::IModelRunner runner;
    GraphStatus status;

    // Initialise the model runner with the TosaSerializationHandler
    status = runner.initialize(*handler);
    if(status != GraphStatus::TOSA_VALID)
    {
        throw armnn::Exception("An error has occurred while initialising the TOSA Reference Model.");
    }

    // Set the inputs
    for (uint32_t inputSlotIdx = 0; inputSlotIdx < inputNames.size(); ++inputSlotIdx)
    {
        DataType dataType = m_workloadInfo.m_InputTensorInfos[inputSlotIdx].GetDataType();
        switch (dataType)
        {
            case DataType::Float16:
                SetInput<half_float::half>(runner, inputNames[inputSlotIdx], inputSlotIdx);
                break;
            case DataType::Float32:
                SetInput<float>(runner, inputNames[inputSlotIdx], inputSlotIdx);
                break;
            case DataType::QAsymmU8:
            case DataType::QAsymmS8:
            case DataType::QSymmS8:
            case DataType::QSymmS16:
            case DataType::Signed32:
                SetInput<int32_t>(runner, inputNames[inputSlotIdx], inputSlotIdx);
                break;
            case DataType::Signed64:
                SetInput<int64_t>(runner, inputNames[inputSlotIdx], inputSlotIdx);
                break;
            case DataType::Boolean:
                SetInput<unsigned char>(runner, inputNames[inputSlotIdx], inputSlotIdx);
                break;
            default:
                throw armnn::Exception("Input data type is unsupported in TOSA Reference Backend.");
        }
    }

    // Run the TOSA Reference Model
    status = runner.run();
    if(status != GraphStatus::TOSA_VALID)
    {
        throw armnn::Exception("An error has occurred while running the TOSA Reference Model.");
    }

    // Gets the outputs
    for (uint32_t outputSlotIdx = 0; outputSlotIdx < outputNames.size(); ++outputSlotIdx)
    {
        DataType dataType = m_workloadInfo.m_OutputTensorInfos[outputSlotIdx].GetDataType();
        switch (dataType)
        {
            case DataType::Float16:
                GetOutput<half_float::half>(runner, outputNames[outputSlotIdx], outputSlotIdx);
                break;
            case DataType::Float32:
                GetOutput<float>(runner, outputNames[outputSlotIdx], outputSlotIdx);
                break;
            case DataType::QAsymmU8:
            case DataType::QAsymmS8:
            case DataType::QSymmS8:
            case DataType::QSymmS16:
            case DataType::Signed32:
                GetOutput<int32_t>(runner, outputNames[outputSlotIdx], outputSlotIdx);
                break;
            case DataType::Signed64:
                GetOutput<int64_t>(runner, outputNames[outputSlotIdx], outputSlotIdx);
                break;
            case DataType::Boolean:
                GetOutput<unsigned char>(runner, outputNames[outputSlotIdx], outputSlotIdx);
                break;
            default:
                throw armnn::Exception("Output data type is unsupported in TOSA Reference Backend.");
        }
    }
}

template <typename T>
void TosaRefPreCompiledWorkload::SetInput(TosaReference::IModelRunner& runner,
                                          std::string inputName,
                                          uint32_t inputIndex) const
{
    std::vector<T> inputData(m_Data.m_Inputs[inputIndex]->GetShape().GetNumElements());
    m_Data.m_Inputs[inputIndex]->CopyOutTo(inputData.data());

    runner.setInput<T>(inputName, inputData);
}

template <typename T>
void TosaRefPreCompiledWorkload::GetOutput(TosaReference::IModelRunner& runner,
                                           std::string outputName,
                                           uint32_t outputIndex) const
{
    std::vector<T> actualOutputs = runner.getOutput<T>(outputName);

    m_Data.m_Outputs[outputIndex]->CopyInFrom(actualOutputs.data());
}

bool TosaRefPreCompiledWorkloadValidate(std::string*)
{
    return true;
}

}    //namespace armnn
