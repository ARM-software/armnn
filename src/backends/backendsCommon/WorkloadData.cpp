//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//
#include "WorkloadData.hpp"

#include "CpuTensorHandle.hpp"

#include <DataLayoutIndexed.hpp>

#include <algorithm>
#include <iomanip>
#include <string>
#include <sstream>

#include <boost/format.hpp>
#include <boost/numeric/conversion/cast.hpp>

using namespace armnnUtils;

namespace armnn
{

//---------------------------------------------------------------
DataType GetBiasDataType(DataType inputDataType)
{
    switch (inputDataType)
    {
        case DataType::Float16:
            return DataType::Float16;
        case DataType::Float32:
            return DataType::Float32;
        case DataType::QuantisedAsymm8:
            return DataType::Signed32;
        case DataType::QuantisedSymm16:
            return DataType::Signed32;
        default:
            BOOST_ASSERT_MSG(false, "Invalid input data type");
            return DataType::Float32;
    }
}

namespace
{

//---------------------------------------------------------------
//android ndk does not support std::to_string function.
template <typename T>
std::string to_string(T value)
{
    std::ostringstream os;
    os << value;
    return os.str();
}

//---------------------------------------------------------------
void ValidatePointer(const void* ptr, std::string const& descName, std::string const& paramName)
{
    if (!ptr)
    {
        throw InvalidArgumentException(descName +  ": Invalid null pointer. The " +
                                      paramName + " parameter must be set.");
    }
}

//---------------------------------------------------------------
void ValidateTensorShapesMatch(const TensorInfo& first,
                               const TensorInfo& second,
                               std::string const& descName,
                               std::string const& firstName,
                               std::string const& secondName)
{
    if (first.GetShape() != second.GetShape())
    {
        throw InvalidArgumentException(descName + ": "
                                       + firstName + " & " + secondName + " must have identical shapes");
    }
}

//---------------------------------------------------------------
void ValidateNumInputs(const WorkloadInfo& workloadInfo, std::string const& descName, const unsigned int expectedSize)
{
    if (workloadInfo.m_InputTensorInfos.size() != expectedSize)
    {
        throw InvalidArgumentException(descName +
                                       ": Requires exactly " + to_string(expectedSize) + "input(s). " +
                                       to_string(workloadInfo.m_InputTensorInfos.size()) + " have been provided.");
    }
}

//---------------------------------------------------------------
void ValidateNumOutputs(const WorkloadInfo& workloadInfo, std::string const& descName, const unsigned int expectedSize)
{
    if (workloadInfo.m_OutputTensorInfos.size() != expectedSize)
    {
        throw InvalidArgumentException(descName +
                                       ": Requires exactly " + to_string(expectedSize) + " output(s). " +
                                       to_string(workloadInfo.m_OutputTensorInfos.size()) + " has been provided.");
    }
}

//---------------------------------------------------------------
void ValidateTensorNumDimensions(const TensorInfo&  tensor,
                                 std::string const& descName,
                                 unsigned int       numDimensions,
                                 std::string const& tensorName)
{
    if (tensor.GetNumDimensions() != numDimensions)
    {
        throw InvalidArgumentException(descName + ": Expected " + to_string(numDimensions) + " but got " +
            to_string(tensor.GetNumDimensions()) + " dimensions for " +
            tensorName + " tensor.");
    }
}

//---------------------------------------------------------------
void ValidateTensorDataType(const TensorInfo& tensor, DataType dataType,
    const std::string& descName, std::string const& tensorName)
{
    if (tensor.GetDataType() != dataType)
    {
        throw InvalidArgumentException(descName + ": Expected data type " + GetDataTypeName(dataType) + " but got " +
            GetDataTypeName(tensor.GetDataType()) + " for " + tensorName + " tensor.");
    }
}

//---------------------------------------------------------------
void ValidateTensorQuantizationSpace(const TensorInfo& first,
                                     const TensorInfo& second,
                                     const std::string& descName,
                                     std::string const& firstName,
                                     std::string const& secondName)
{
    if (!first.IsQuantized() ||
        !second.IsQuantized())
    {
        // Not a quantized type, ignore the validation
        return;
    }

    DataType firstDataType  = first.GetDataType();
    DataType secondDataType = second.GetDataType();

    if (firstDataType != secondDataType)
    {
        throw InvalidArgumentException(descName + ": " + firstName + " and " + secondName +
                                       " must be of the same quantized type, " +
                                       firstName + " is " + GetDataTypeName(firstDataType) + ", " +
                                       secondName + " is " + GetDataTypeName(secondDataType));
    }

    if (!first.IsTypeSpaceMatch(second))
    {
        throw InvalidArgumentException(descName + ": " + firstName + " and " + secondName +
                                       " must have the same quantization space, " +
                                       firstName + " has offset " + to_string(first.GetQuantizationOffset()) +
                                       " and scale " + to_string(first.GetQuantizationScale()) + ", " +
                                       secondName + " has offset " + to_string(second.GetQuantizationOffset()) +
                                       " and scale " + to_string(second.GetQuantizationScale()));
    }
}

//---------------------------------------------------------------
void ValidateBiasTensorQuantization(const TensorInfo& biasTensor, const TensorInfo& inputTensorInfo,
    const TensorInfo& weightsTensorInfo, const std::string& descName)
{
    if (biasTensor.GetQuantizationOffset() != 0)
    {
        throw InvalidArgumentException(descName + ": Expected zero quantization offset for bias tensor but got " +
            to_string(biasTensor.GetQuantizationOffset()));
    }
    const float expectedScale = inputTensorInfo.GetQuantizationScale() * weightsTensorInfo.GetQuantizationScale();
    if (std::abs(biasTensor.GetQuantizationScale() - expectedScale) > 0.00000001f)
    {
        // Print the float values with extra precision to see very small differences
        std::stringstream msg;
        msg << std::setprecision(10) << descName << ": Expected " << expectedScale <<
            " quantization scale for bias tensor (the product of the input and weight scales), but got " <<
            biasTensor.GetQuantizationScale();
        throw InvalidArgumentException(msg.str());
    }
}

//---------------------------------------------------------------
void ValidateTensors(const std::vector<ITensorHandle*>& vec,
    unsigned int numExpected,
    const std::string& descName,
    const std::string& varName)
{
    if (vec.empty() && numExpected > 0)
    {
        throw InvalidArgumentException(descName + ": Invalid empty " + varName + " array.");
    }

    for (unsigned int i = 0; i < numExpected; ++i)
    {
        if (!vec[i])
        {
            throw InvalidArgumentException(descName + ": Invalid NULL for " + varName + to_string(i));
        }
    }
}

//---------------------------------------------------------------
void ValidateBroadcastTensorShapesMatch(const TensorInfo& first,
                                        const TensorInfo& second,
                                        const TensorInfo& output,
                                        std::string const& descName,
                                        std::string const& firstName,
                                        std::string const& secondName)
{
    // Tensors must have the same number of dimensions in order to be explicit about which dimensions will get
    // broadcasted.
    if (first.GetNumDimensions() != second.GetNumDimensions())
    {
        throw InvalidArgumentException(descName  + ": Tensors "
            + firstName + " & " + secondName
            + " must have the same number of dimensions in order to be broadcasted");
    }
    uint32_t numDims = first.GetNumDimensions();
    std::vector<uint32_t> outputDims(numDims, 0u);
    for (uint32_t i = 0; i < numDims; i++)
    {
        const bool dimsNotEqual = first.GetShape()[i] != second.GetShape()[i];
        const bool dimsNotOne = (first.GetShape()[i] != 1) && (second.GetShape()[i] != 1);
        if (dimsNotEqual && dimsNotOne)
        {
            throw InvalidArgumentException("Broadcasting is not possible for incompatible shapes");
        }
        outputDims[i] = std::max(first.GetShape()[i], second.GetShape()[i]);
    }
    TensorShape broadcastShape =  TensorShape(boost::numeric_cast<unsigned int>(outputDims.size()), outputDims.data());
    if (broadcastShape != output.GetShape())
    {
        throw InvalidArgumentException(descName + ": The tensor shape resulting from adding "
                                       + firstName + " & " + secondName
                                       + " does not match the output shape");
    }
}

//---------------------------------------------------------------
/// Validates that the output tensor's quantization scale is greater than the product
/// of the two input tensors' quantization scales. This is a requirement of the implementation of
/// the quantized multiplication.
void ValidateTensorQuantizationMultiplier(const TensorInfo& inputTensor1, const TensorInfo& inputTensor2,
    const TensorInfo& outputTensorInfo, std::string const& descName,
    const std::string& inputTensor1Name, const std::string& inputTensor2Name, const std::string& outputTensorName)
{
    if (outputTensorInfo.GetDataType() == DataType::QuantisedAsymm8)
    {
        if (outputTensorInfo.GetQuantizationScale() <=
            inputTensor1.GetQuantizationScale() * inputTensor2.GetQuantizationScale())
        {
            std::stringstream msg;
            msg << descName << ": Quantization scale of " << outputTensorName << " is not greater than " <<
                "the product of the " << inputTensor1Name << " and " << inputTensor2Name << " tensors";
            throw InvalidArgumentException(msg.str());
        }
    }
}

//---------------------------------------------------------------
void ValidateDataTypes(const TensorInfo& info,
                       const std::vector<armnn::DataType>& supportedTypes,
                       std::string const& descName)
{
    auto iterator = std::find(supportedTypes.begin(), supportedTypes.end(), info.GetDataType());
    if (iterator == supportedTypes.end())
    {
        throw InvalidArgumentException(descName  + ": " + " Tensor type is not supported.");
    }
}

} //namespace

void QueueDescriptor::ValidateInputsOutputs(const std::string& descName,
    unsigned int numExpectedIn, unsigned int numExpectedOut) const
{
    ValidateTensors(m_Inputs, numExpectedIn, descName, "input");
    ValidateTensors(m_Outputs, numExpectedOut, descName, "output");
}

//---------------------------------------------------------------
void MemCopyQueueDescriptor::Validate(const WorkloadInfo& workloadInfo) const
{
    ValidateNumInputs(workloadInfo, "MemCopyQueueDescriptor", 1);
    ValidateNumOutputs(workloadInfo, "MemCopyQueueDescriptor" , 1);

    if (workloadInfo.m_InputTensorInfos.size() != workloadInfo.m_OutputTensorInfos.size())
    {
        throw InvalidArgumentException(boost::str(
            boost::format("Number of input infos (%1%) does not match the number of output infos (%2%)")
                % workloadInfo.m_InputTensorInfos.size() % workloadInfo.m_OutputTensorInfos.size()));
    }

    for (std::size_t i = 0; i < workloadInfo.m_InputTensorInfos.size(); ++i)
    {
        if (workloadInfo.m_InputTensorInfos[i].GetNumElements() !=
            workloadInfo.m_OutputTensorInfos[i].GetNumElements())
        {
            throw InvalidArgumentException(boost::str(
                boost::format("Number of elements for tensor input and output %1% does not match")
                    % i ));
        }
    }

    if (m_Inputs.size() != m_Outputs.size())
    {
        throw InvalidArgumentException(boost::str(
            boost::format("Number of inputs (%1%) does not match the number of outputs (%2%)")
                % m_Inputs.size() % m_Outputs.size()));
    }

    for (unsigned int i = 0; i < m_Inputs.size(); ++i)
    {
        if (!m_Inputs[i])
        {
            throw InvalidArgumentException(boost::str(boost::format("Invalid null input %1%") % i));
        }

        if (!m_Outputs[i])
        {
            throw InvalidArgumentException(boost::str(boost::format("Invalid null output %1%") % i));
        }
    }
}

//---------------------------------------------------------------
void ActivationQueueDescriptor::Validate(const WorkloadInfo& workloadInfo) const
{
    ValidateNumInputs(workloadInfo, "ActivationQueueDescriptor", 1);
    ValidateNumOutputs(workloadInfo, "ActivationQueueDescriptor", 1);
    ValidateTensorShapesMatch(workloadInfo.m_InputTensorInfos[0],
                              workloadInfo.m_OutputTensorInfos[0],
                              "ActivationQueueDescriptor",
                              "input",
                              "output");

    std::vector<DataType> supportedTypes = {
        DataType::Float32,
        DataType::Float16,
        DataType::QuantisedAsymm8,
        DataType::QuantisedSymm16
    };

    ValidateDataTypes(workloadInfo.m_InputTensorInfos[0],
                      supportedTypes,
                      "ActivationQueueDescriptor");

    ValidateDataTypes(workloadInfo.m_OutputTensorInfos[0],
                      {workloadInfo.m_InputTensorInfos[0].GetDataType()},
                      "ActivationQueueDescriptor");
}

//---------------------------------------------------------------
void SoftmaxQueueDescriptor::Validate(const WorkloadInfo& workloadInfo) const
{
    ValidateNumInputs(workloadInfo, "SoftmaxQueueDescriptor", 1);
    ValidateNumOutputs(workloadInfo, "SoftmaxQueueDescriptor", 1);

    ValidateTensorShapesMatch(workloadInfo.m_InputTensorInfos[0],
                              workloadInfo.m_OutputTensorInfos[0],
                              "SoftmaxQueueDescriptor",
                              "input",
                              "output");

    std::vector<DataType> supportedTypes =
    {
            DataType::Float16,
            DataType::Float32,
            DataType::QuantisedAsymm8,
            DataType::QuantisedSymm16
    };

    ValidateDataTypes(workloadInfo.m_InputTensorInfos[0],
                      supportedTypes,
                      "SoftmaxQueueDescriptor");

    ValidateDataTypes(workloadInfo.m_OutputTensorInfos[0],
                      {workloadInfo.m_InputTensorInfos[0].GetDataType()},
                      "SoftmaxQueueDescriptor");
}

//---------------------------------------------------------------
void SplitterQueueDescriptor::Validate(const WorkloadInfo& workloadInfo) const
{
    ValidateNumInputs(workloadInfo, "SplitterQueueDescriptor", 1);

    // Check the supported data types
    std::vector<DataType> supportedTypes =
    {
            DataType::Float32,
            DataType::Float16,
            DataType::Boolean,
            DataType::Signed32,
            DataType::QuantisedAsymm8,
            DataType::QuantisedSymm16
    };

    for (unsigned long i = 0; i < workloadInfo.m_OutputTensorInfos.size(); ++i)
    {
        ValidateDataTypes(workloadInfo.m_OutputTensorInfos[i],
                          supportedTypes,
                          "SplitterQueueDescriptor");
    }
    ValidateDataTypes(workloadInfo.m_OutputTensorInfos[0],
                      {workloadInfo.m_InputTensorInfos[0].GetDataType()},
                      "SplitterQueueDescriptor");

    if (workloadInfo.m_OutputTensorInfos.size() <= 0)
    {
        throw InvalidArgumentException("SplitterQueueDescriptor: At least one output needs to be provided.");
    }

    if (workloadInfo.m_OutputTensorInfos.size() != m_ViewOrigins.size())
    {
        throw InvalidArgumentException(
            "SplitterQueueDescriptor: Number of split windows "
            "has to match number of workloadInfo.m_OutputTensorInfos. "
            "Number of windows: " +
            to_string(m_ViewOrigins.size()) +
            ". Number of workloadInfo.m_OutputTensorInfos: " + to_string(workloadInfo.m_OutputTensorInfos.size()));
    }

    //The dimensionality of all the windows has to match the dimensionality (not shape) of the input.
    std::size_t inputDims = workloadInfo.m_InputTensorInfos[0].GetNumDimensions();
    for(unsigned int w = 0; w < m_ViewOrigins.size(); ++w )
    {
        //Checks that the dimensionality of input is same as the split windows.
        ViewOrigin const& e = m_ViewOrigins[w];
        if (e.m_Origin.size() != inputDims)
        {
            throw InvalidArgumentException("SplitterQueueDescriptor: Window origin have to "
                                           "have the same dimensionality as the input tensor. "
                                           "Window origin (index: " +
                                           to_string(w) + ") has " + to_string(e.m_Origin.size()) +
                                           " dimensions, the input "
                                           "tensor has " +
                                           to_string(inputDims) + " dimensions.");
        }
        for (unsigned int i = 0; i < e.m_Origin.size(); ++i)
        {
            if (e.m_Origin[i] + workloadInfo.m_OutputTensorInfos[w].GetShape()[i] >
                workloadInfo.m_InputTensorInfos[0].GetShape()[i])
            {
                throw InvalidArgumentException("SplitterQueueDescriptor: Window extent coordinates have to "
                                               "be smaller or equal than the size of the input in that coord.");
            }
        }
    }
}

//---------------------------------------------------------------
void ConcatQueueDescriptor::Validate(const WorkloadInfo& workloadInfo) const
{
    ValidateNumOutputs(workloadInfo, "ConcatQueueDescriptor", 1);

    if (m_Inputs.size() <= 0)
    {
        throw InvalidArgumentException("ConcatQueueDescriptor: At least one input needs to be provided.");
    }
    if (m_Outputs.size() <= 0)
    {
        throw InvalidArgumentException("ConcatQueueDescriptor: At least one output needs to be provided.");
    }

    if (workloadInfo.m_InputTensorInfos.size() <= 0)
    {
        throw InvalidArgumentException("ConcatQueueDescriptor: At least one TensorInfo input needs to be provided.");
    }
    if (workloadInfo.m_OutputTensorInfos.size() <= 0)
    {
        throw InvalidArgumentException("ConcatQueueDescriptor: At least one TensorInfo output needs to be provided.");
    }

    if(m_Parameters.GetConcatAxis() > workloadInfo.m_InputTensorInfos[0].GetShape().GetNumDimensions())
    {
        throw InvalidArgumentException("Invalid Concatenation Axis provided");
    }

    if (workloadInfo.m_InputTensorInfos[0].GetShape().GetNumDimensions() - m_Parameters.GetConcatAxis() == 1)
    {
        return;
    }

    if (workloadInfo.m_InputTensorInfos.size() != m_ViewOrigins.size())
    {
        throw InvalidArgumentException(
            "ConcatQueueDescriptor: Number of split windows "
            "has to match number of workloadInfo.m_InputTensorInfos. "
            "Number of windows: " +
            to_string(m_ViewOrigins.size()) +
            ". Number of workloadInfo.m_InputTensorInfos: " + to_string(workloadInfo.m_InputTensorInfos.size()));
    }

    //The dimensionality of all the windows has to match the dimensionality (not shape) of the output.
    std::size_t outputDims = workloadInfo.m_OutputTensorInfos[0].GetNumDimensions();
    for(unsigned int w = 0; w < m_ViewOrigins.size(); ++w )
    {
        //Checks that the dimensionality of output is same as the split windows.
        ViewOrigin const& e = m_ViewOrigins[w];
        if (e.m_Origin.size() != outputDims)
        {
            throw InvalidArgumentException("ConcatQueueDescriptor: Window origin have to "
                                           "have the same dimensionality as the output tensor. "
                                           "Window origin (index: " +
                                           to_string(w) + ") has " + to_string(e.m_Origin.size()) +
                                           " dimensions, the output "
                                           "tensor has " +
                                           to_string(outputDims) + " dimensions.");
        }
        //Checks that the merge windows are within the output tensor.
        for (unsigned int i = 0; i < e.m_Origin.size(); ++i)
        {
            if (e.m_Origin[i] + workloadInfo.m_InputTensorInfos[w].GetShape()[i]
                > workloadInfo.m_OutputTensorInfos[0].GetShape()[i])
            {
                throw InvalidArgumentException("ConcatQueueDescriptor: Window extent coordinates have to "
                                               "be smaller or equal than the size of the output in that coord.");
            }
        }
    }

    // Check the supported data types
    std::vector<DataType> supportedTypes =
    {
            DataType::Float32,
            DataType::Float16,
            DataType::Boolean,
            DataType::Signed32,
            DataType::QuantisedAsymm8,
            DataType::QuantisedSymm16
    };

    for (unsigned long i = 0; i < workloadInfo.m_InputTensorInfos.size(); ++i)
    {
        ValidateDataTypes(workloadInfo.m_InputTensorInfos[i],
                          supportedTypes,
                          "ConcatQueueDescriptor");
    }
    ValidateDataTypes(workloadInfo.m_OutputTensorInfos[0],
                      {workloadInfo.m_InputTensorInfos[0].GetDataType()},
                      "ConcatQueueDescriptor");
}

//---------------------------------------------------------------
void FullyConnectedQueueDescriptor::Validate(const WorkloadInfo& workloadInfo) const
{
    ValidateNumInputs(workloadInfo, "FullyConnectedQueueDescriptor", 1);
    ValidateNumOutputs(workloadInfo, "FullyConnectedQueueDescriptor", 1);
    ValidateTensorNumDimensions(workloadInfo.m_OutputTensorInfos[0], "FullyConnectedQueueDescriptor", 2, "output");

    if (!(workloadInfo.m_InputTensorInfos[0].GetNumDimensions() == 2 ||
          workloadInfo.m_InputTensorInfos[0].GetNumDimensions() == 4))
    {
        throw InvalidArgumentException("FullyConnectedQueueDescriptor: Input tensor must have 2 or 4 dimensions.");
    }

    if (m_Weight == nullptr)
    {
        throw InvalidArgumentException("FullyConnectedQueueDescriptor: Weight tensor descriptor is missing.");
    }

    ValidateTensorNumDimensions(m_Weight->GetTensorInfo(), "FullyConnectedQueueDescriptor", 2, "weight");

    if (m_Parameters.m_BiasEnabled)
    {
        if (m_Bias == nullptr)
        {
            throw InvalidArgumentException("FullyConnectedQueueDescriptor: Bias is enabled but "
                                           "bias value tensor descriptor is missing.");
        }

        // Validates type and quantization values.
        ValidateBiasTensorQuantization(m_Bias->GetTensorInfo(),
            workloadInfo.m_InputTensorInfos[0], m_Weight->GetTensorInfo(), "FullyConnectedQueueDescriptor");

        ValidateTensorDataType(m_Bias->GetTensorInfo(),
                               GetBiasDataType(workloadInfo.m_InputTensorInfos[0].GetDataType()),
                               "FullyConnectedQueueDescriptor", "bias");

        ValidateTensorNumDimensions(m_Bias->GetTensorInfo(), "FullyConnectedQueueDescriptor", 1, "bias");
    }

    ValidateTensorQuantizationMultiplier(workloadInfo.m_InputTensorInfos[0], m_Weight->GetTensorInfo(),
        workloadInfo.m_OutputTensorInfos[0], "FullyConnectedQueueDescriptor", "input", "weights", "output");

    // Check the supported data types
    std::vector<DataType> supportedTypes =
    {
            DataType::Float32,
            DataType::Float16,
            DataType::QuantisedAsymm8,
            DataType::QuantisedSymm16
    };

    ValidateDataTypes(workloadInfo.m_InputTensorInfos[0],
                      supportedTypes,
                      "FullyConnectedQueueDescriptor");

    ValidateDataTypes(workloadInfo.m_OutputTensorInfos[0],
                      {workloadInfo.m_InputTensorInfos[0].GetDataType()},
                      "FullyConnectedQueueDescriptor");
}

//---------------------------------------------------------------
void NormalizationQueueDescriptor::Validate(const WorkloadInfo& workloadInfo) const
{
    ValidateNumInputs(workloadInfo, "NormalizationQueueDescriptor", 1);
    ValidateNumOutputs(workloadInfo, "NormalizationQueueDescriptor", 1);
    ValidateTensorShapesMatch(workloadInfo.m_InputTensorInfos[0],
                              workloadInfo.m_OutputTensorInfos[0],
                              "NormalizationQueueDescriptor",
                              "input",
                              "output");
}

void AdditionQueueDescriptor::Validate(const WorkloadInfo& workloadInfo) const
{
    ValidateNumInputs(workloadInfo, "AdditionQueueDescriptor", 2);
    ValidateNumOutputs(workloadInfo, "AdditionQueueDescriptor", 1);

    std::vector<DataType> supportedTypes = {
        DataType::Float32,
        DataType::QuantisedAsymm8,
        DataType::QuantisedSymm16,
        DataType::Float16
    };

    ValidateDataTypes(workloadInfo.m_InputTensorInfos[0],
                      supportedTypes,
                      "AdditionQueueDescriptor");

    ValidateDataTypes(workloadInfo.m_InputTensorInfos[1],
                      supportedTypes,
                      "AdditionQueueDescriptor");

    ValidateDataTypes(workloadInfo.m_OutputTensorInfos[0],
                      supportedTypes,
                      "AdditionQueueDescriptor");

    ValidateBroadcastTensorShapesMatch(workloadInfo.m_InputTensorInfos[0],
                                       workloadInfo.m_InputTensorInfos[1],
                                       workloadInfo.m_OutputTensorInfos[0],
                                       "AdditionQueueDescriptor",
                                       "first input",
                                       "second input");
}

//---------------------------------------------------------------
void MultiplicationQueueDescriptor::Validate(const WorkloadInfo& workloadInfo) const
{
    ValidateNumInputs(workloadInfo, "MultiplicationQueueDescriptor", 2);
    ValidateNumOutputs(workloadInfo, "MultiplicationQueueDescriptor", 1);

    std::vector<DataType> supportedTypes = {
        DataType::Float32,
        DataType::QuantisedAsymm8,
        DataType::QuantisedSymm16,
        DataType::Float16
    };

    ValidateDataTypes(workloadInfo.m_InputTensorInfos[0],
                      supportedTypes,
                      "MultiplicationQueueDescriptor");

    ValidateDataTypes(workloadInfo.m_InputTensorInfos[1],
                      supportedTypes,
                      "MultiplicationQueueDescriptor");

    ValidateDataTypes(workloadInfo.m_OutputTensorInfos[0],
                      supportedTypes,
                      "MultiplicationQueueDescriptor");

    ValidateBroadcastTensorShapesMatch(workloadInfo.m_InputTensorInfos[0],
                                       workloadInfo.m_InputTensorInfos[1],
                                       workloadInfo.m_OutputTensorInfos[0],
                                       "MultiplicationQueueDescriptor",
                                       "first input",
                                       "second input");
}

void BatchNormalizationQueueDescriptor::Validate(const WorkloadInfo& workloadInfo) const
{
    ValidateNumInputs(workloadInfo, "BatchNormalizationQueueDescriptor", 1);
    ValidateNumOutputs(workloadInfo, "BatchNormalizationQueueDescriptor", 1);
    ValidateTensorShapesMatch(workloadInfo.m_InputTensorInfos[0],
                              workloadInfo.m_OutputTensorInfos[0],
                              "BatchNormalizationQueueDescriptor",
                              "input",
                              "output");
    ValidatePointer(m_Mean, "BatchNormalizationQueueDescriptor", "mean");
    ValidatePointer(m_Variance, "BatchNormalizationQueueDescriptor", "variance");
    ValidatePointer(m_Beta, "BatchNormalizationQueueDescriptor", "beta");
    ValidatePointer(m_Gamma, "BatchNormalizationQueueDescriptor", "gamma");


    ValidateTensorNumDimensions(m_Mean->GetTensorInfo(), "BatchNormalizationQueueDescriptor", 1, "mean");
    ValidateTensorNumDimensions(m_Variance->GetTensorInfo(), "BatchNormalizationQueueDescriptor", 1, "variance");
    ValidateTensorNumDimensions(m_Beta->GetTensorInfo(), "BatchNormalizationQueueDescriptor", 1, "beta");
    ValidateTensorNumDimensions(m_Gamma->GetTensorInfo(), "BatchNormalizationQueueDescriptor", 1, "gamma");

    ValidateTensorShapesMatch(
        m_Mean->GetTensorInfo(), m_Variance->GetTensorInfo(), "BatchNormalizationQueueDescriptor", "mean", "variance");
    ValidateTensorShapesMatch(
        m_Mean->GetTensorInfo(), m_Beta->GetTensorInfo(), "BatchNormalizationQueueDescriptor", "mean", "beta");
    ValidateTensorShapesMatch(
        m_Mean->GetTensorInfo(), m_Gamma->GetTensorInfo(), "BatchNormalizationQueueDescriptor", "mean", "gamma");
}

void Convolution2dQueueDescriptor::Validate(const WorkloadInfo& workloadInfo) const
{
    ValidateNumInputs(workloadInfo, "Convolution2dQueueDescriptor", 1);
    ValidateNumOutputs(workloadInfo, "Convolution2dQueueDescriptor", 1);

    ValidateTensorNumDimensions(workloadInfo.m_InputTensorInfos[0], "Convolution2dQueueDescriptor", 4, "input");
    ValidateTensorNumDimensions(workloadInfo.m_OutputTensorInfos[0], "Convolution2dQueueDescriptor", 4, "output");

    ValidatePointer(m_Weight, "Convolution2dQueueDescriptor", "weight");
    ValidateTensorNumDimensions(m_Weight->GetTensorInfo(), "Convolution2dQueueDescriptor", 4, "weight");
    ValidateTensorDataType(m_Weight->GetTensorInfo(), workloadInfo.m_InputTensorInfos[0].GetDataType(),
        "Convolution2dQueueDescriptor", "weight");
    if (m_Parameters.m_BiasEnabled)
    {
        ValidateTensorNumDimensions(m_Bias->GetTensorInfo(), "Convolution2dQueueDescriptor", 1, "bias");
        ValidateTensorDataType(m_Bias->GetTensorInfo(),
                               GetBiasDataType(workloadInfo.m_InputTensorInfos[0].GetDataType()),
                               "Convolution2dQueueDescriptor", "bias");
        ValidateBiasTensorQuantization(m_Bias->GetTensorInfo(),
            workloadInfo.m_InputTensorInfos[0], m_Weight->GetTensorInfo(), "Convolution2dQueueDescriptor");
    }

    ValidateTensorQuantizationMultiplier(workloadInfo.m_InputTensorInfos[0], m_Weight->GetTensorInfo(),
        workloadInfo.m_OutputTensorInfos[0], "Convolution2dQueueDescriptor", "input", "weights", "output");
}

void DepthwiseConvolution2dQueueDescriptor::Validate(const WorkloadInfo& workloadInfo) const
{
    ValidateNumInputs(workloadInfo, "DepthwiseConvolution2dQueueDescriptor", 1);
    ValidateNumOutputs(workloadInfo, "DepthwiseConvolution2dQueueDescriptor", 1);

    ValidateTensorNumDimensions(
        workloadInfo.m_InputTensorInfos[0], "DepthwiseConvolution2dQueueDescriptor", 4, "input");
    ValidateTensorNumDimensions(
        workloadInfo.m_OutputTensorInfos[0], "DepthwiseConvolution2dQueueDescriptor", 4, "output");

    ValidatePointer(m_Weight, "DepthwiseConvolution2dQueueDescriptor", "weight");
    ValidateTensorNumDimensions(m_Weight->GetTensorInfo(), "DepthwiseConvolution2dQueueDescriptor", 4, "weight");

    if (m_Parameters.m_DilationX < 1 || m_Parameters.m_DilationY < 1 )
    {
        throw InvalidArgumentException(
            boost::str(boost::format("DepthwiseConvolution2dQueueDescriptor: dilationX (provided %1%) "
                                     "and dilationY (provided %2%) cannot be smaller than 1.")
                                     % m_Parameters.m_DilationX % m_Parameters.m_DilationX));
    }

    const unsigned int channelIndex = (m_Parameters.m_DataLayout == DataLayout::NCHW) ? 1 : 3;

    // Expected weight shape: [ M, I, H, W ] - This shape does NOT depend on the data layout
    // inputChannels * channelMultiplier should be equal to outputChannels.
    const unsigned int numWeightChannelMultiplier = m_Weight->GetTensorInfo().GetShape()[0];
    const unsigned int numWeightInputChannels = m_Weight->GetTensorInfo().GetShape()[1];
    const unsigned int numWeightOutputChannels = workloadInfo.m_OutputTensorInfos[0].GetShape()[channelIndex];
    if (numWeightChannelMultiplier * numWeightInputChannels != numWeightOutputChannels)
    {
        throw InvalidArgumentException(
            boost::str(boost::format("DepthwiseConvolution2dQueueDescriptor: output_channels (provided %1%) should be "
                                     "equal to input_channels (provided %2%) multiplied by channel_multiplier "
                                     "(provided %3%).")
                                     % numWeightOutputChannels % numWeightInputChannels % numWeightChannelMultiplier));
    }

    if (m_Parameters.m_BiasEnabled)
    {
        ValidatePointer(m_Bias, "DepthwiseConvolution2dQueueDescriptor", "bias");
        ValidateTensorNumDimensions(m_Bias->GetTensorInfo(), "DepthwiseConvolution2dQueueDescriptor", 1, "bias");
        ValidateBiasTensorQuantization(m_Bias->GetTensorInfo(),
            workloadInfo.m_InputTensorInfos[0], m_Weight->GetTensorInfo(), "DepthwiseConvolution2dQueueDescriptor");

        ValidateTensorDataType(m_Bias->GetTensorInfo(),
                               GetBiasDataType(workloadInfo.m_InputTensorInfos[0].GetDataType()),
                               "DepthwiseConvolution2dQueueDescriptor", "bias");
    }

    ValidateTensorQuantizationMultiplier(workloadInfo.m_InputTensorInfos[0], m_Weight->GetTensorInfo(),
        workloadInfo.m_OutputTensorInfos[0], "DepthwiseConvolution2dQueueDescriptor", "input", "weights", "output");

    // Check the supported data types
    std::vector<DataType> supportedTypes = {
        DataType::Float32,
        DataType::QuantisedAsymm8,
        DataType::QuantisedSymm16,
        DataType::Float16
    };

    ValidateDataTypes(workloadInfo.m_InputTensorInfos[0],
                      supportedTypes,
                      "DepthwiseConvolution2dQueueDescriptor");

    ValidateDataTypes(workloadInfo.m_OutputTensorInfos[0],
                      {workloadInfo.m_InputTensorInfos[0].GetDataType()},
                      "DepthwiseConvolution2dQueueDescriptor");
}

void PermuteQueueDescriptor::Validate(const WorkloadInfo& workloadInfo) const
{
    ValidateNumInputs(workloadInfo, "PermuteQueueDescriptor", 1);
    ValidateNumOutputs(workloadInfo, "PermuteQueueDescriptor", 1);

    const PermutationVector& mapping = m_Parameters.m_DimMappings;

    const TensorInfo& input  = workloadInfo.m_InputTensorInfos[0];
    const TensorInfo& output = workloadInfo.m_OutputTensorInfos[0];

    ValidateTensorNumDimensions(input, "PermuteQueueDescriptor", mapping.GetSize(), "input");
    ValidateTensorNumDimensions(output, "PermuteQueueDescriptor", mapping.GetSize(), "output");

    for (unsigned int i = 0; i < mapping.GetSize(); ++i)
    {
        if (input.GetShape()[i] != output.GetShape()[mapping[i]])
        {
            throw InvalidArgumentException("PermuteQueueDescriptor: src dimension " + to_string(i) +
                                               " (=" + to_string(input.GetShape()[i]) + ") " +
                                               "must match dst dimension " + to_string(mapping[i]) +
                                               " (=" + to_string(output.GetShape()[mapping[i]]) + ")");
        }
    }
}

void Pooling2dQueueDescriptor::Validate(const WorkloadInfo& workloadInfo) const
{
    ValidateNumInputs(workloadInfo, "Pooling2dQueueDescriptor", 1);
    ValidateNumOutputs(workloadInfo, "Pooling2dQueueDescriptor", 1);

    ValidateTensorNumDimensions(workloadInfo.m_InputTensorInfos[0], "Pooling2dQueueDescriptor", 4, "input");
    ValidateTensorNumDimensions(workloadInfo.m_OutputTensorInfos[0], "Pooling2dQueueDescriptor", 4, "output");
}

void ResizeBilinearQueueDescriptor::Validate(const WorkloadInfo& workloadInfo) const
{
    ValidateNumInputs(workloadInfo, "ResizeBilinearQueueDescriptor", 1);
    ValidateNumOutputs(workloadInfo, "ResizeBilinearQueueDescriptor", 1);

    ValidateTensorNumDimensions(workloadInfo.m_InputTensorInfos[0], "ResizeBilinearQueueDescriptor", 4, "input");
    ValidateTensorNumDimensions(workloadInfo.m_OutputTensorInfos[0], "ResizeBilinearQueueDescriptor", 4, "output");

    // Resizes bilinear only changes width and height: batch and channel count must match.
    {
        const unsigned int inputBatchSize = workloadInfo.m_InputTensorInfos[0].GetShape()[0];
        const unsigned int outputBatchSize = workloadInfo.m_OutputTensorInfos[0].GetShape()[0];
        if (inputBatchSize != outputBatchSize)
        {
            throw InvalidArgumentException(
                boost::str(boost::format("ResizeBilinearQueueDescriptor: Input batch size (%1%) "
                    "does not match output batch size (%2%)") % inputBatchSize % outputBatchSize));
        }
    }

    {
        DataLayoutIndexed dimensionIndices(m_Parameters.m_DataLayout);
        const unsigned int inputChannelCount =
            workloadInfo.m_InputTensorInfos[0].GetShape()[dimensionIndices.GetChannelsIndex()];
        const unsigned int outputChannelCount =
            workloadInfo.m_OutputTensorInfos[0].GetShape()[dimensionIndices.GetChannelsIndex()];
        if (inputChannelCount != outputChannelCount)
        {
            throw InvalidArgumentException(
                boost::str(boost::format("ResizeBilinearQueueDescriptor: Input channel count (%1%) "
                    "does not match output channel count (%2%)") % inputChannelCount % outputChannelCount));
        }
    }
}

void FakeQuantizationQueueDescriptor::Validate(const WorkloadInfo& workloadInfo) const
{
    ValidateNumInputs(workloadInfo, "FakeQuantizationQueueDescriptor", 1);
    ValidateNumOutputs(workloadInfo, "FakeQuantizationQueueDescriptor", 1);

    ValidateTensorNumDimensions(workloadInfo.m_InputTensorInfos[0], "FakeQuantizationQueueDescriptor", 2, "input");
    ValidateTensorNumDimensions(workloadInfo.m_OutputTensorInfos[0], "FakeQuantizationQueueDescriptor", 2, "output");
    ValidateTensorShapesMatch(workloadInfo.m_InputTensorInfos[0],
        workloadInfo.m_OutputTensorInfos[0],
        "FakeQuantizationQueueDescriptor",
        "input",
        "output");
    if (m_Parameters.m_Min > m_Parameters.m_Max)
    {
        throw InvalidArgumentException("FakeQuantizationQueueDescriptor: min cannot be greater than max");
    }

}

void L2NormalizationQueueDescriptor::Validate(const WorkloadInfo& workloadInfo) const
{
    ValidateNumInputs(workloadInfo, "L2NormalizationQueueDescriptor", 1);
    ValidateNumOutputs(workloadInfo, "L2NormalizationQueueDescriptor", 1);

    ValidateTensorNumDimensions(workloadInfo.m_InputTensorInfos[0], "L2NormalizationQueueDescriptor", 4, "input");
    ValidateTensorNumDimensions(workloadInfo.m_OutputTensorInfos[0], "L2NormalizationQueueDescriptor", 4, "output");
    ValidateTensorShapesMatch(workloadInfo.m_InputTensorInfos[0],
        workloadInfo.m_OutputTensorInfos[0],
        "L2NormalizationQueueDescriptor",
        "input",
        "output");
}

void ConstantQueueDescriptor::Validate(const WorkloadInfo& workloadInfo) const
{
    ValidateNumInputs(workloadInfo, "ConstantQueueDescriptor", 0);
    ValidateNumOutputs(workloadInfo, "ConstantQueueDescriptor", 1);

    if (!m_LayerOutput)
    {
        throw InvalidArgumentException("ConstantQueueDescriptor: No const input specified");
    }

    ValidateTensorShapesMatch(m_LayerOutput->GetTensorInfo(),
        workloadInfo.m_OutputTensorInfos[0],
        "ConstantQueueDescriptor",
        "constant",
        "output");

    // Check the supported data types
    std::vector<DataType> supportedTypes =
    {
        DataType::Float32,
        DataType::Float16,
        DataType::Signed32,
        DataType::QuantisedAsymm8,
        DataType::QuantisedSymm16
    };

    ValidateDataTypes(workloadInfo.m_OutputTensorInfos[0], supportedTypes, "ConstantQueueDescriptor");
}

void ReshapeQueueDescriptor::Validate(const WorkloadInfo& workloadInfo) const
{
    ValidateNumInputs(workloadInfo, "ReshapeQueueDescriptor", 1);
    ValidateNumOutputs(workloadInfo, "ReshapeQueueDescriptor", 1);

    if (workloadInfo.m_InputTensorInfos[0].GetNumElements() != workloadInfo.m_OutputTensorInfos[0].GetNumElements())
    {
        throw InvalidArgumentException("ReshapeQueueDescriptor: Input tensor has " +
            to_string(workloadInfo.m_InputTensorInfos[0].GetNumElements()) + " but output tensor has " +
            to_string(workloadInfo.m_OutputTensorInfos[0].GetNumElements()) + " elements.");
    }

    // Check the supported data types
    std::vector<DataType> supportedTypes =
    {
        DataType::Float32,
        DataType::Float16,
        DataType::QuantisedAsymm8,
        DataType::QuantisedSymm16
    };

    ValidateDataTypes(workloadInfo.m_InputTensorInfos[0], supportedTypes, "ReshapeQueueDescriptor");
    ValidateDataTypes(workloadInfo.m_OutputTensorInfos[0], supportedTypes, "ReshapeQueueDescriptor");
}

void SpaceToBatchNdQueueDescriptor::Validate(const WorkloadInfo& workloadInfo) const
{
    ValidateNumInputs(workloadInfo, "SpaceToBatchNdQueueDescriptor", 1);
    ValidateNumOutputs(workloadInfo, "SpaceToBatchNdQueueDescriptor", 1);

    ValidateTensorNumDimensions(workloadInfo.m_InputTensorInfos[0], "SpaceToBatchNdQueueDescriptor", 4, "input");
    ValidateTensorNumDimensions(workloadInfo.m_OutputTensorInfos[0], "SpaceToBatchNdQueueDescriptor", 4, "output");

    if (m_Parameters.m_BlockShape.size() != 2)
    {
        throw InvalidArgumentException("Block Shape must contain 2 spatial dimensions");
    }

    if (m_Parameters.m_BlockShape.size() != m_Parameters.m_PadList.size())
    {
        throw InvalidArgumentException("Pad List must contain the same number of dimensions as Block Shape.");
    }

    const TensorShape inputShape = workloadInfo.m_InputTensorInfos[0].GetShape();

    std::pair<unsigned int, unsigned int> heightPad = m_Parameters.m_PadList[0];
    std::pair<unsigned int, unsigned int> widthPad = m_Parameters.m_PadList[1];

    DataLayoutIndexed dimensionIndices(m_Parameters.m_DataLayout);
    unsigned int inputHeight = inputShape[dimensionIndices.GetHeightIndex()]
                               + heightPad.first + heightPad.second;

    unsigned int inputWidth = inputShape[dimensionIndices.GetWidthIndex()]
                              + widthPad.first + widthPad.second;

    unsigned int numInputElements = inputShape[0] * inputHeight * inputWidth
                                    * inputShape[dimensionIndices.GetChannelsIndex()];

    if (workloadInfo.m_OutputTensorInfos[0].GetNumElements() != numInputElements)
    {
        throw InvalidArgumentException("SpaceToBatchNdQueueDescriptor: Input tensor has " +
            to_string(numInputElements) + " after padding but output tensor has " +
            to_string(workloadInfo.m_OutputTensorInfos[0].GetNumElements()) + " elements.");
    }

    if (inputHeight % m_Parameters.m_BlockShape[0] != 0 || inputWidth % m_Parameters.m_BlockShape[1] != 0)
    {
        throw InvalidArgumentException(
            "Input shape after padding must be divisible by Block Shape in all spatial dimensions");
    }

    std::vector<DataType> supportedTypes =
    {
            DataType::Float16,
            DataType::Float32,
            DataType::QuantisedAsymm8,
            DataType::QuantisedSymm16
    };

    ValidateDataTypes(workloadInfo.m_InputTensorInfos[0],
                      supportedTypes,
                      "SpaceToBatchNdQueueDescriptor");

    ValidateDataTypes(workloadInfo.m_OutputTensorInfos[0],
                      {workloadInfo.m_InputTensorInfos[0].GetDataType()},
                      "SpaceToBatchNdQueueDescriptor");
}

void FloorQueueDescriptor::Validate(const WorkloadInfo& workloadInfo) const
{
    const std::string floorQueueDescString = "FloorQueueDescriptor";

    ValidateNumInputs(workloadInfo,  floorQueueDescString, 1);
    ValidateNumOutputs(workloadInfo, floorQueueDescString, 1);

    std::vector<DataType> supportedTypes =
    {
            DataType::Float32,
            DataType::QuantisedSymm16
    };

    ValidateDataTypes(workloadInfo.m_InputTensorInfos[0],  supportedTypes, floorQueueDescString);
    ValidateDataTypes(workloadInfo.m_OutputTensorInfos[0], supportedTypes, floorQueueDescString);

    if (workloadInfo.m_InputTensorInfos[0] != workloadInfo.m_OutputTensorInfos[0])
    {
        throw InvalidArgumentException(floorQueueDescString + ": Input and output tensor infos do not match.");
    }
}

void LstmQueueDescriptor::Validate(const WorkloadInfo& workloadInfo) const
{
    ValidateTensorNumDimensions(workloadInfo.m_InputTensorInfos[0], "LstmQueueDescriptor", 2, "input");
    ValidateTensorNumDimensions(workloadInfo.m_OutputTensorInfos[0], "LstmQueueDescriptor", 2, "output");

    std::vector<DataType> supportedTypes = {
        DataType::Float16,
        DataType::Float32,
        DataType::QuantisedSymm16
    };

    ValidateDataTypes(workloadInfo.m_InputTensorInfos[0],
                      supportedTypes,
                      "LstmQueueDescriptor");

    ValidateDataTypes(workloadInfo.m_OutputTensorInfos[0],
                      supportedTypes,
                      "LstmQueueDescriptor");
}

void ConvertFp32ToFp16QueueDescriptor::Validate(const WorkloadInfo& workloadInfo) const
{
    ValidateNumInputs(workloadInfo, "ConvertFp32ToFp16QueueDescriptor", 1);
    ValidateNumOutputs(workloadInfo, "ConvertFp32ToFp16QueueDescriptor", 1);

    if (workloadInfo.m_InputTensorInfos[0].GetDataType() != DataType::Float32)
    {
        throw InvalidArgumentException("ConvertFp32ToFp16QueueDescriptor: Input tensor type must be Float32.");
    }

    if (workloadInfo.m_OutputTensorInfos[0].GetDataType() != DataType::Float16)
    {
        throw InvalidArgumentException("ConvertFp32ToFp16QueueDescriptor: Output tensor type must be Float16.");
    }

    ValidateTensorShapesMatch(workloadInfo.m_InputTensorInfos[0],
                              workloadInfo.m_OutputTensorInfos[0],
                              "ConvertFp32ToFp16QueueDescriptor",
                              "input",
                              "output");
}

void ConvertFp16ToFp32QueueDescriptor::Validate(const WorkloadInfo& workloadInfo) const
{
    ValidateNumInputs(workloadInfo, "ConvertFp16ToFp32QueueDescriptor", 1);
    ValidateNumOutputs(workloadInfo, "ConvertFp16ToFp32QueueDescriptor", 1);

    if (workloadInfo.m_InputTensorInfos[0].GetDataType() != DataType::Float16)
    {
        throw InvalidArgumentException("ConvertFp16ToFp32QueueDescriptor: Input tensor type must be Float16.");
    }
    if (workloadInfo.m_OutputTensorInfos[0].GetDataType() != DataType::Float32)
    {
        throw InvalidArgumentException("ConvertFp16ToFp32QueueDescriptor: Output tensor type must be Float32.");
    }

    ValidateTensorShapesMatch(workloadInfo.m_InputTensorInfos[0],
                              workloadInfo.m_OutputTensorInfos[0],
                              "ConvertFp16ToFp32QueueDescriptor",
                              "input",
                              "output");
}

void DivisionQueueDescriptor::Validate(const WorkloadInfo& workloadInfo) const
{
    ValidateNumInputs(workloadInfo, "DivisionQueueDescriptor", 2);
    ValidateNumOutputs(workloadInfo, "DivisionQueueDescriptor", 1);

    std::vector<DataType> supportedTypes = {
        DataType::Float32,
        DataType::QuantisedAsymm8,
        DataType::QuantisedSymm16,
        DataType::Float16
    };

    ValidateDataTypes(workloadInfo.m_InputTensorInfos[0],
                      supportedTypes,
                      "DivisionQueueDescriptor");

    ValidateDataTypes(workloadInfo.m_InputTensorInfos[1],
                      supportedTypes,
                      "DivisionQueueDescriptor");

    ValidateDataTypes(workloadInfo.m_OutputTensorInfos[0],
                      supportedTypes,
                      "DivisionQueueDescriptor");

    ValidateBroadcastTensorShapesMatch(workloadInfo.m_InputTensorInfos[0],
                                       workloadInfo.m_InputTensorInfos[1],
                                       workloadInfo.m_OutputTensorInfos[0],
                                       "DivisionQueueDescriptor",
                                       "first input",
                                       "second input");
}

void SubtractionQueueDescriptor::Validate(const WorkloadInfo& workloadInfo) const
{
    ValidateNumInputs(workloadInfo, "SubtractionQueueDescriptor", 2);
    ValidateNumOutputs(workloadInfo, "SubtractionQueueDescriptor", 1);

    std::vector<DataType> supportedTypes = {
        DataType::Float32,
        DataType::QuantisedAsymm8,
        DataType::QuantisedSymm16,
        DataType::Float16
    };

    ValidateDataTypes(workloadInfo.m_InputTensorInfos[0],
                      supportedTypes,
                      "SubtractionQueueDescriptor");

    ValidateDataTypes(workloadInfo.m_InputTensorInfos[1],
                      supportedTypes,
                      "SubtractionQueueDescriptor");

    ValidateDataTypes(workloadInfo.m_OutputTensorInfos[0],
                      supportedTypes,
                      "SubtractionQueueDescriptor");

    ValidateBroadcastTensorShapesMatch(workloadInfo.m_InputTensorInfos[0],
                                       workloadInfo.m_InputTensorInfos[1],
                                       workloadInfo.m_OutputTensorInfos[0],
                                       "SubtractionQueueDescriptor",
                                       "first input",
                                       "second input");
}

void MaximumQueueDescriptor::Validate(const WorkloadInfo& workloadInfo) const
{
    ValidateNumInputs(workloadInfo, "MaximumQueueDescriptor", 2);
    ValidateNumOutputs(workloadInfo, "MaximumQueueDescriptor", 1);

    std::vector<DataType> supportedTypes = {
        DataType::Float32,
        DataType::QuantisedAsymm8,
        DataType::QuantisedSymm16
    };

    ValidateDataTypes(workloadInfo.m_InputTensorInfos[0],
                      supportedTypes,
                      "MaximumQueueDescriptor");

    ValidateDataTypes(workloadInfo.m_InputTensorInfos[1],
                      supportedTypes,
                      "MaximumQueueDescriptor");

    ValidateDataTypes(workloadInfo.m_OutputTensorInfos[0],
                      supportedTypes,
                      "MaximumQueueDescriptor");

    ValidateBroadcastTensorShapesMatch(workloadInfo.m_InputTensorInfos[0],
                                       workloadInfo.m_InputTensorInfos[1],
                                       workloadInfo.m_OutputTensorInfos[0],
                                       "MaximumQueueDescriptor",
                                       "first input",
                                       "second input");
}

void MeanQueueDescriptor::Validate(const WorkloadInfo& workloadInfo) const
{
    ValidateNumInputs(workloadInfo, "MeanQueueDescriptor", 1);
    ValidateNumOutputs(workloadInfo, "MeanQueueDescriptor", 1);

    const TensorInfo& input  = workloadInfo.m_InputTensorInfos[0];
    const TensorInfo& output = workloadInfo.m_OutputTensorInfos[0];

    if (m_Parameters.m_KeepDims)
    {
        ValidateTensorNumDimensions(output, "MeanQueueDescriptor", input.GetNumDimensions(), "output");
    }
    else if (m_Parameters.m_Axis.empty())
    {
        ValidateTensorNumDimensions(output, "MeanQueueDescriptor", 1, "output");
    }
    else
    {
        auto outputDim = input.GetNumDimensions() - boost::numeric_cast<unsigned int>(m_Parameters.m_Axis.size());
        ValidateTensorNumDimensions(output,
                                    "MeanQueueDescriptor",
                                    outputDim > 0 ? outputDim : 1,
                                    "output");
    }
}

void PadQueueDescriptor::Validate(const WorkloadInfo& workloadInfo) const
{
    ValidateNumInputs(workloadInfo, "PadQueueDescriptor", 1);
    ValidateNumOutputs(workloadInfo, "PadQueueDescriptor", 1);

    const TensorInfo& input = workloadInfo.m_InputTensorInfos[0];
    const TensorInfo& output = workloadInfo.m_OutputTensorInfos[0];

    // input and output should have the same number of dimensions
    ValidateTensorNumDimensions(output, "PadQueueDescriptor", input.GetNumDimensions(), "output");
    // there should be entry in the pad list for each dimension in the input tensor
    if (m_Parameters.m_PadList.size() != input.GetNumDimensions()) {
        throw InvalidArgumentException("Pad List should contain the same number of entries as there"
                                       " are dimensions in the input tensor that is " +
                                       to_string(input.GetNumDimensions()) + " entries " +
                                       " not " + to_string(m_Parameters.m_PadList.size()) + " entries.");
    }
}

void QuantizeQueueDescriptor::Validate(const WorkloadInfo& workloadInfo) const
{
    ValidateNumInputs(workloadInfo, "QuantizeQueueDescriptor", 1);
    ValidateNumOutputs(workloadInfo, "QuantizeQueueDescriptor", 1);


    if (workloadInfo.m_InputTensorInfos[0].GetDataType() != DataType::Float32)
    {
        throw InvalidArgumentException("Quantize only accepts Float32 inputs.");
    }

    if (workloadInfo.m_OutputTensorInfos[0].GetDataType() != DataType::QuantisedAsymm8 &&
        workloadInfo.m_OutputTensorInfos[0].GetDataType() != DataType::QuantisedSymm16)
    {
        throw InvalidArgumentException("Output of quantized layer must be quantized type.");
    }
}

void BatchToSpaceNdQueueDescriptor::Validate(const WorkloadInfo& workloadInfo) const
{
    ValidateNumInputs(workloadInfo, "BatchToSpaceNdQueueDescriptor", 1);
    ValidateNumOutputs(workloadInfo, "BatchToSpaceNdQueueDescriptor", 1);
}

void StridedSliceQueueDescriptor::Validate(const WorkloadInfo& workloadInfo) const
{
    ValidateNumInputs(workloadInfo, "StridedSliceQueueDescriptor", 1);
    ValidateNumOutputs(workloadInfo, "StridedSliceQueueDescriptor", 1);

    const TensorInfo& input = workloadInfo.m_InputTensorInfos[0];
    const TensorInfo& output = workloadInfo.m_OutputTensorInfos[0];

    std::vector<DataType> supportedTypes =
    {
        DataType::Float16,
        DataType::Float32,
        DataType::QuantisedAsymm8,
        DataType::QuantisedSymm16
    };

    ValidateDataTypes(input,  supportedTypes, "StridedSliceQueueDescriptor");
    ValidateDataTypes(output, supportedTypes, "StridedSliceQueueDescriptor");

    ValidateDataTypes(output, { input.GetDataType() }, "StridedSliceQueueDescriptor");

    ValidateTensorQuantizationSpace(input, output, "StridedSliceQueueDescriptor", "input", "output");

    const uint32_t rank = input.GetNumDimensions();

    if (rank > 4)
    {
        throw InvalidArgumentException(
            "StridedSliceLayer: Input tensors with rank greater than 4 are not supported");
    }

    // Begin, End & Stride length must be of rank(input0)
    if (m_Parameters.m_Begin.size() != rank)
    {
        throw InvalidArgumentException("StridedSliceLayer: Begin length must be of rank input0("
                                       + to_string(rank) + ")");
    }

    if (m_Parameters.m_End.size() != rank)
    {
        throw InvalidArgumentException("StridedSliceLayer: End length must be of rank input0("
                                       + to_string(rank) + ")");
    }

    if (m_Parameters.m_Stride.size() != rank)
    {
        throw InvalidArgumentException("StridedSliceLayer: Stride length must be of rank input0("
                                       + to_string(rank) + ")");
    }

    // Stride entries must be non-zero
    for (auto& stride : m_Parameters.m_Stride)
    {
        if (stride == 0)
        {
            throw InvalidArgumentException("StridedSliceLayer: Stride entries must be non-zero");
        }
    }
}

void MinimumQueueDescriptor::Validate(const WorkloadInfo& workloadInfo) const
{
    ValidateNumInputs(workloadInfo, "MinimumQueueDescriptor", 2);
    ValidateNumOutputs(workloadInfo, "MinimumQueueDescriptor", 1);

    std::vector<DataType> supportedTypes = {
        DataType::Float32,
        DataType::QuantisedAsymm8,
        DataType::QuantisedSymm16
    };

    ValidateDataTypes(workloadInfo.m_InputTensorInfos[0],
                      supportedTypes,
                      "MinimumQueueDescriptor");

    ValidateDataTypes(workloadInfo.m_InputTensorInfos[1],
                      supportedTypes,
                      "MinimumQueueDescriptor");

    ValidateDataTypes(workloadInfo.m_OutputTensorInfos[0],
                      supportedTypes,
                      "MinimumQueueDescriptor");

    ValidateBroadcastTensorShapesMatch(workloadInfo.m_InputTensorInfos[0],
                                       workloadInfo.m_InputTensorInfos[1],
                                       workloadInfo.m_OutputTensorInfos[0],
                                       "MinimumQueueDescriptor",
                                       "first input",
                                       "second input");
}

void DebugQueueDescriptor::Validate(const WorkloadInfo& workloadInfo) const
{
    ValidateNumInputs(workloadInfo, "DebugQueueDescriptor", 1);
    ValidateNumOutputs(workloadInfo, "DebugQueueDescriptor", 1);
}

void EqualQueueDescriptor::Validate(const WorkloadInfo& workloadInfo) const
{
    ValidateNumInputs(workloadInfo, "EqualQueueDescriptor", 2);
    ValidateNumOutputs(workloadInfo, "EqualQueueDescriptor", 1);

    ValidateBroadcastTensorShapesMatch(workloadInfo.m_InputTensorInfos[0],
                                       workloadInfo.m_InputTensorInfos[1],
                                       workloadInfo.m_OutputTensorInfos[0],
                                       "EqualQueueDescriptor",
                                       "first input",
                                       "second input");

    if (workloadInfo.m_OutputTensorInfos[0].GetDataType() != DataType::Boolean)
    {
        throw InvalidArgumentException("EqualQueueDescriptor: Output tensor type must be Boolean.");
    }
}

void GreaterQueueDescriptor::Validate(const WorkloadInfo& workloadInfo) const
{
    ValidateNumInputs(workloadInfo, "GreaterQueueDescriptor", 2);
    ValidateNumOutputs(workloadInfo, "GreaterQueueDescriptor", 1);

    ValidateBroadcastTensorShapesMatch(workloadInfo.m_InputTensorInfos[0],
                                       workloadInfo.m_InputTensorInfos[1],
                                       workloadInfo.m_OutputTensorInfos[0],
                                       "GreaterQueueDescriptor",
                                       "first input",
                                       "second input");

    if (workloadInfo.m_OutputTensorInfos[0].GetDataType() != DataType::Boolean)
    {
        throw InvalidArgumentException("GreaterQueueDescriptor: Output tensor type must be Boolean.");
    }
}

void RsqrtQueueDescriptor::Validate(const WorkloadInfo& workloadInfo) const
{
    ValidateNumInputs(workloadInfo, "RsqrtQueueDescriptor", 1);
    ValidateNumOutputs(workloadInfo, "RsqrtQueueDescriptor", 1);
    ValidateTensorShapesMatch(workloadInfo.m_InputTensorInfos[0],
                              workloadInfo.m_OutputTensorInfos[0],
                              "RsqrtQueueDescriptor",
                              "input",
                              "output");
}

void GatherQueueDescriptor::Validate(const WorkloadInfo& workloadInfo) const
{
    ValidateNumInputs(workloadInfo, "GatherQueueDescriptor", 2);
    ValidateNumOutputs(workloadInfo, "GatherQueueDescriptor", 1);

    const TensorInfo& indices = workloadInfo.m_InputTensorInfos[1];

    if (indices.GetDataType() != DataType::Signed32)
    {
        throw InvalidArgumentException("GatherQueueDescriptor: Indices tensor type must be int32.");
    }

    const TensorInfo& params = workloadInfo.m_InputTensorInfos[0];
    const TensorInfo& output = workloadInfo.m_OutputTensorInfos[0];
    unsigned int paramsDim = params.GetNumDimensions();
    unsigned int indicesDim = indices.GetNumDimensions();
    unsigned int outputDim = paramsDim - 1 + indicesDim;

    ValidateTensorNumDimensions(output, "GatherQueueDescriptor", outputDim, "output");
}

void DetectionPostProcessQueueDescriptor::Validate(const WorkloadInfo& workloadInfo) const
{
    ValidateNumInputs(workloadInfo, "DetectionPostProcessQueueDescriptor", 2);

    if (workloadInfo.m_OutputTensorInfos.size() != 4)
    {
        throw InvalidArgumentException("DetectionPostProcessQueueDescriptor: Requires exactly four outputs. " +
                                       to_string(workloadInfo.m_OutputTensorInfos.size()) + " has been provided.");
    }

    if (m_Anchors == nullptr)
    {
        throw InvalidArgumentException("DetectionPostProcessQueueDescriptor: Anchors tensor descriptor is missing.");
    }

    const TensorInfo& boxEncodingsInfo =  workloadInfo.m_InputTensorInfos[0];
    const TensorInfo& scoresInfo =  workloadInfo.m_InputTensorInfos[1];
    const TensorInfo& anchorsInfo = m_Anchors->GetTensorInfo();
    const TensorInfo& detectionBoxesInfo = workloadInfo.m_OutputTensorInfos[0];
    const TensorInfo& detectionClassesInfo = workloadInfo.m_OutputTensorInfos[1];
    const TensorInfo& detectionScoresInfo = workloadInfo.m_OutputTensorInfos[2];
    const TensorInfo& numDetectionsInfo = workloadInfo.m_OutputTensorInfos[3];

    ValidateTensorNumDimensions(boxEncodingsInfo, "DetectionPostProcessQueueDescriptor", 3, "box encodings");
    ValidateTensorNumDimensions(scoresInfo, "DetectionPostProcessQueueDescriptor", 3, "scores");
    ValidateTensorNumDimensions(anchorsInfo, "DetectionPostProcessQueueDescriptor", 2, "anchors");

    ValidateTensorNumDimensions(detectionBoxesInfo, "DetectionPostProcessQueueDescriptor", 3, "detection boxes");
    ValidateTensorNumDimensions(detectionScoresInfo, "DetectionPostProcessQueueDescriptor", 2, "detection scores");
    ValidateTensorNumDimensions(detectionClassesInfo, "DetectionPostProcessQueueDescriptor", 2, "detection classes");
    ValidateTensorNumDimensions(numDetectionsInfo, "DetectionPostProcessQueueDescriptor", 1, "num detections");

    ValidateTensorDataType(detectionBoxesInfo, DataType::Float32,
                          "DetectionPostProcessQueueDescriptor", "detection boxes");
    ValidateTensorDataType(detectionScoresInfo, DataType::Float32,
                          "DetectionPostProcessQueueDescriptor", "detection scores");
    ValidateTensorDataType(detectionClassesInfo, DataType::Float32,
                          "DetectionPostProcessQueueDescriptor", "detection classes");
    ValidateTensorDataType(numDetectionsInfo, DataType::Float32,
                          "DetectionPostProcessQueueDescriptor", "num detections");

    if (m_Parameters.m_NmsIouThreshold <= 0.0f || m_Parameters.m_NmsIouThreshold > 1.0f)
    {
        throw InvalidArgumentException("DetectionPostProcessQueueDescriptor: Intersection over union threshold "
                                       "must be positive and less than or equal to 1.");
    }
    if (scoresInfo.GetShape()[2] != m_Parameters.m_NumClasses + 1)
    {
        throw InvalidArgumentException("DetectionPostProcessQueueDescriptor: Number of classes with background "
                                       "should be equal to number of classes + 1.");
    }
}

void DequantizeQueueDescriptor::Validate(const WorkloadInfo& workloadInfo) const
{
    ValidateNumInputs(workloadInfo, "DequantizeQueueDescriptor", 1);
    ValidateNumOutputs(workloadInfo, "DequantizeQueueDescriptor", 1);

    if (workloadInfo.m_InputTensorInfos[0].GetDataType() != DataType::QuantisedAsymm8 &&
        workloadInfo.m_InputTensorInfos[0].GetDataType() != DataType::QuantisedSymm16)
    {
        throw InvalidArgumentException("Input to dequantize layer must be quantized type.");
    }

    if (workloadInfo.m_OutputTensorInfos[0].GetDataType() != DataType::Float32)
    {
        throw InvalidArgumentException("Output of dequantize layer must be Float32 type.");
    }
}

void MergeQueueDescriptor::Validate(const WorkloadInfo& workloadInfo) const
{
    ValidateNumInputs(workloadInfo, "MergeQueueDescriptor", 2);
    ValidateNumOutputs(workloadInfo, "MergeQueueDescriptor", 1);

    ValidateTensorShapesMatch(workloadInfo.m_InputTensorInfos[0],
                              workloadInfo.m_InputTensorInfos[1],
                              "MergeQueueDescriptor",
                              "input0",
                              "input1");

    ValidateTensorShapesMatch(workloadInfo.m_InputTensorInfos[0],
                              workloadInfo.m_OutputTensorInfos[0],
                              "MergeQueueDescriptor",
                              "input0",
                              "output");

    const DataType dataType = workloadInfo.m_InputTensorInfos[0].GetDataType();
    ValidateTensorDataType(workloadInfo.m_InputTensorInfos[1], dataType, "MergeQueueDescriptor", "input1");
    ValidateTensorDataType(workloadInfo.m_OutputTensorInfos[0], dataType, "MergeQueueDescriptor", "output");
}

void SwitchQueueDescriptor::Validate(const WorkloadInfo& workloadInfo) const
{
    ValidateNumInputs(workloadInfo, "SwitchQueueDescriptor", 2);
    ValidateNumOutputs(workloadInfo, "SwitchQueueDescriptor", 2);

    std::vector<DataType> supportedTypes = {
        DataType::Float32,
        DataType::QuantisedAsymm8,
        DataType::QuantisedSymm16
    };

    ValidateDataTypes(workloadInfo.m_InputTensorInfos[0],
                      supportedTypes,
                      "SwitchQueueDescriptor");

    ValidateDataTypes(workloadInfo.m_InputTensorInfos[1],
                      supportedTypes,
                      "SwitchQueueDescriptor");

    ValidateDataTypes(workloadInfo.m_OutputTensorInfos[0],
                      supportedTypes,
                      "SwitchQueueDescriptor");

    ValidateTensorShapesMatch(workloadInfo.m_InputTensorInfos[0],
                              workloadInfo.m_OutputTensorInfos[0],
                              "SwitchQueueDescriptor",
                              "input0",
                              "output0");

    ValidateTensorShapesMatch(workloadInfo.m_InputTensorInfos[0],
                              workloadInfo.m_OutputTensorInfos[1],
                              "SwitchQueueDescriptor",
                              "input0",
                              "output1");
}

void PreCompiledQueueDescriptor::Validate(const WorkloadInfo& workloadInfo) const
{
    // This is internally generated so it should not need validation.
}

} //namespace armnn
