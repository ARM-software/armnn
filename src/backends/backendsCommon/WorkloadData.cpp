//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include <armnn/backends/TensorHandle.hpp>
#include <armnn/backends/WorkloadData.hpp>
#include <armnn/backends/WorkloadInfo.hpp>
#include <armnnUtils/DataLayoutIndexed.hpp>
#include <armnnUtils/TensorUtils.hpp>
#include <armnn/utility/NumericCast.hpp>
#include <armnn/Logging.hpp>

#include <algorithm>
#include <iomanip>
#include <string>
#include <sstream>

#include <fmt/format.h>

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
        case DataType::BFloat16:
        case DataType::Float32:
            return DataType::Float32;
        case DataType::QAsymmS8:
            return DataType::Signed32;
        case DataType::QAsymmU8:
            return DataType::Signed32;
        case DataType::QSymmS8:
            return DataType::Signed32;
        case DataType::QSymmS16:
            return DataType::Signed32;
        default:
            ARMNN_ASSERT_MSG(false, "Invalid input data type");
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
void ValidateTensorNumDimensions(const TensorInfo& tensor,
                                 std::string const& descName,
                                 unsigned int numDimensions,
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
void ValidateTensorNumElements(const TensorInfo& tensor,
                               std::string const& descName,
                               unsigned int numElements,
                               std::string const& tensorName)
{
    if (tensor.GetNumElements() != numElements)
    {
        throw InvalidArgumentException(descName + ": Expected " + to_string(numElements) + " but got " +
                                       to_string(tensor.GetNumElements()) + " elements for " +
                                       tensorName + " tensor.");
    }
}

//---------------------------------------------------------------
void ValidateTensorNumDimNumElem(const TensorInfo& tensorInfo,
                                 unsigned int numDimension,
                                 unsigned int numElements,
                                 std::string const& tensorName)
{
    const std::string functionName{"ValidateTensorNumDimNumElem"};
    ValidateTensorNumDimensions(tensorInfo, functionName, numDimension, tensorName);
    ValidateTensorNumElements(tensorInfo, functionName, numElements, tensorName);
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

void ValidPerAxisQuantizedDataType(const TensorInfo& tensor, const std::string& descName, const std::string& tensorName)
{
    if (tensor.GetDataType() != DataType::QSymmS8)
    {
        throw InvalidArgumentException(descName +
            ": Expected data type which supports per-axis quantization scheme but got " +
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
void ValidateBiasTensorQuantization(const TensorInfo& biasTensor,
                                    const TensorInfo& inputTensorInfo,
                                    const TensorInfo& weightsTensorInfo,
                                    const std::string& descName)
{
    // Helper lambda function to validate a single bias quantization scale value
    auto VerifyBiasQuantizationScale = [&descName](float biasScale, float expectedScale) -> void
    {
        constexpr float tolerance = 0.0001f;
        if (std::abs(biasScale - expectedScale) > tolerance)
        {
            // Print the float values with extra precision to see very small differences
            ARMNN_LOG(warning) << std::setprecision(6) << descName << ": Expected " << expectedScale <<
                " for bias quantization scale (product of input and weight scales), but got " <<
                biasScale << ". Using scale provided.";
        }
    };

    if (biasTensor.GetQuantizationOffset() != 0)
    {
        throw InvalidArgumentException(descName + ": Expected zero quantization offset for bias tensor but got " +
            to_string(biasTensor.GetQuantizationOffset()));
    }

    if (biasTensor.HasMultipleQuantizationScales() || weightsTensorInfo.HasMultipleQuantizationScales())
    {
        // Validate per-axis quantization scales
        const std::vector<float>& weightScales = weightsTensorInfo.GetQuantizationScales();
        const std::vector<float>& biasScales   = biasTensor.GetQuantizationScales();

        if (weightScales.size() != biasScales.size())
        {
            std::stringstream msg;
            msg << descName << ": Expected matching number of per-axis quantization scales for weights and bias, "
                << "but got different values. This is currently unsupported: weights=" << weightScales.size()
                << ", biases=" << biasScales.size();
            throw InvalidArgumentException(msg.str(), CHECK_LOCATION());
        }

        for (size_t i = 0ul; i < biasScales.size(); ++i)
        {
            const float expectedScale = inputTensorInfo.GetQuantizationScale() * weightScales[i];
            VerifyBiasQuantizationScale(biasScales[i], expectedScale);
        }
    }
    else
    {
        // Validate per-tensor quantization scale
        const float expectedScale = inputTensorInfo.GetQuantizationScale() * weightsTensorInfo.GetQuantizationScale();
        VerifyBiasQuantizationScale(biasTensor.GetQuantizationScale(), expectedScale);
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
    TensorShape broadcastShape = TensorShape(armnn::numeric_cast<unsigned int>(outputDims.size()), outputDims.data());
    if (broadcastShape != output.GetShape())
    {
        throw InvalidArgumentException(descName + ": The tensor shape resulting from adding "
                                       + firstName + " & " + secondName
                                       + " does not match the output shape");
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

//---------------------------------------------------------------
void ValidateTensorDataTypesMatch(const TensorInfo& first,
                                  const TensorInfo& second,
                                  std::string const& descName,
                                  std::string const& firstName,
                                  std::string const& secondName)
{
    if (first.GetDataType() != second.GetDataType())
    {
        throw InvalidArgumentException(descName + ": " + firstName + " & " + secondName +
                                       " must have identical data types.");
    }
}

//---------------------------------------------------------------
void ValidateTensorNumElementsMatch(const TensorInfo& first,
                                    const TensorInfo& second,
                                    std::string const& descName,
                                    std::string const& firstName,
                                    std::string const& secondName)
{
    if (first.GetNumElements() != second.GetNumElements())
    {
        throw InvalidArgumentException(descName + ": " + firstName + " & " + secondName +
                                       " must have the same number of elements.");
    }
}

void ValidateWeightDataType(const TensorInfo& inputInfo,
                            const TensorInfo& weightInfo,
                            const std::string& descName)
{
    const DataType inputType = inputInfo.GetDataType();
    if (IsQuantized8BitType(inputType))
    {
        const std::vector<DataType> validTypes =
        {
            DataType::QAsymmS8,
            DataType::QAsymmU8,
            DataType::QSymmS8
        };

        ValidateDataTypes(weightInfo, validTypes, descName);
    }
    else
    {
        ValidateTensorDataTypesMatch(inputInfo, weightInfo, descName, "input", "weight");
    }
}

void ValidatePerAxisQuantizationDimension(const TensorInfo& tensorInfo,
                                          const std::string& descName,
                                          const std::string& tensorName)
{
    const Optional<unsigned int>& quantizationDim = tensorInfo.GetQuantizationDim();
    if (!quantizationDim.has_value())
    {
        throw InvalidArgumentException(fmt::format("{0}: Quantization dimension for per-axis quantization "
                                                   "not set on tensor {1}.", descName, tensorName));
    }
}

void ValidatePerAxisQuantizationOffset(const TensorInfo& tensorInfo,
                                       const std::string& descName,
                                       const std::string& tensorName)
{
    int32_t quantizationOffset = tensorInfo.GetQuantizationOffset();
    if (quantizationOffset != 0)
    {
        throw InvalidArgumentException(fmt::format(
            "{0}: Quantization offset for per-axis quantization expected to be 0 on tensor {1}, but got: {2}",
            descName, tensorName, quantizationOffset));
    }
}

void ValidatePerAxisQuantization(const TensorInfo& inputInfo,
                                 const TensorInfo& outputInfo,
                                 const TensorInfo& weightInfo,
                                 const Optional<TensorInfo>& optionalBiasInfo,
                                 const std::string& descName)
{
    if (weightInfo.HasPerAxisQuantization())
    {
        const DataType inputDataType  = inputInfo.GetDataType();
        const DataType outputDataType = outputInfo.GetDataType();

        const bool canHavePerAxisQuantization = (IsQuantized8BitType(inputDataType)) && inputDataType == outputDataType;

        if (!canHavePerAxisQuantization)
        {
            throw InvalidArgumentException(fmt::format(
                "{0}: Per-axis quantization parameters set on tensor {1}, but data type does not support "
                "per-axis quantization.", descName, "weight"));
        }


        ValidPerAxisQuantizedDataType(weightInfo, descName, "weight");
        ValidatePerAxisQuantizationDimension(weightInfo, descName, "weight");
        ValidatePerAxisQuantizationOffset(weightInfo, descName, "weight");

        if (optionalBiasInfo.has_value())
        {
            const TensorInfo& biasInfo = optionalBiasInfo.value();
            if (!biasInfo.HasPerAxisQuantization())
            {
                throw InvalidArgumentException(fmt::format(
                        "{}: Per-axis quantization parameters not set on bias tensor, "
                        "despite being set on weight tensor.", descName));
            }

            ValidateTensorDataType(biasInfo, DataType::Signed32, descName, "bias");
            ValidatePerAxisQuantizationDimension(biasInfo, descName, "bias");
            ValidatePerAxisQuantizationOffset(biasInfo, descName, "bias");
        }
    }
}

} // anonymous namespace

void QueueDescriptor::ValidateInputsOutputs(const std::string& descName,
    unsigned int numExpectedIn, unsigned int numExpectedOut) const
{
    ValidateTensors(m_Inputs, numExpectedIn, descName, "input");
    ValidateTensors(m_Outputs, numExpectedOut, descName, "output");
}

//---------------------------------------------------------------
void MapQueueDescriptor::Validate(const WorkloadInfo& workloadInfo) const
{
    const std::string descriptorName{"MapQueueDescriptor"};

    ValidateNumInputs(workloadInfo,  descriptorName, 1);
    ValidateNumOutputs(workloadInfo, descriptorName, 0);

    for (unsigned int i = 0; i < m_Inputs.size(); ++i)
    {
        if (!m_Inputs[i])
        {
            throw InvalidArgumentException(
                fmt::format("{}: Invalid NULL input {}.", descriptorName, static_cast<int>(i)));
        }
    }
}

//---------------------------------------------------------------
void UnmapQueueDescriptor::Validate(const WorkloadInfo& workloadInfo) const
{
    const std::string descriptorName{"UnmapQueueDescriptor"};

    ValidateNumInputs(workloadInfo,  descriptorName, 1);
    ValidateNumOutputs(workloadInfo, descriptorName, 0);

    for (unsigned int i = 0; i < m_Inputs.size(); ++i)
    {
        if (!m_Inputs[i])
        {
            throw InvalidArgumentException(
                fmt::format("{}: Invalid NULL input {}.", descriptorName, static_cast<int>(i)));
        }
    }
}

//---------------------------------------------------------------
void MemCopyQueueDescriptor::Validate(const WorkloadInfo& workloadInfo) const
{
    const std::string descriptorName{"MemCopyQueueDescriptor"};

    ValidateNumInputs(workloadInfo,  descriptorName, 1);
    ValidateNumOutputs(workloadInfo, descriptorName , 1);

    const TensorInfo& inputTensorInfo  = workloadInfo.m_InputTensorInfos[0];
    const TensorInfo& outputTensorInfo = workloadInfo.m_OutputTensorInfos[0];

    ValidateTensorNumElementsMatch(inputTensorInfo, outputTensorInfo, descriptorName, "input", "output");
    ValidateTensorDataTypesMatch(inputTensorInfo, outputTensorInfo, descriptorName, "input", "output");

    if (m_Inputs.size() != m_Outputs.size())
    {
        throw InvalidArgumentException(fmt::format(
            "{0}: Number of inputs ({1}) does not match the number of outputs ({2}).",
            descriptorName, m_Inputs.size(), m_Outputs.size()));
    }

    for (unsigned int i = 0; i < m_Inputs.size(); ++i)
    {
        if (!m_Inputs[i])
        {
            throw InvalidArgumentException(fmt::format(
                "{0}: Invalid NULL input {1}.", descriptorName, i));
        }

        if (!m_Outputs[i])
        {
            throw InvalidArgumentException(fmt::format("{0}: Invalid NULL output {1}", descriptorName, i));
        }
    }
}

//---------------------------------------------------------------
void MemImportQueueDescriptor::Validate(const WorkloadInfo& workloadInfo) const
{
    ValidateNumInputs(workloadInfo, "MemImportQueueDescriptor", 1);
    ValidateNumOutputs(workloadInfo, "MemImportQueueDescriptor" , 1);

    if (workloadInfo.m_InputTensorInfos.size() != 1)
    {
        throw InvalidArgumentException(fmt::format("Number of input infos ({}) is not 1.",
                                                   workloadInfo.m_InputTensorInfos.size()));

    }

    if (workloadInfo.m_InputTensorInfos.size() != workloadInfo.m_OutputTensorInfos.size())
    {
        throw InvalidArgumentException(fmt::format(
            "Number of input infos ({0}) does not match the number of output infos ({1})",
            workloadInfo.m_InputTensorInfos.size(), workloadInfo.m_OutputTensorInfos.size()));
    }

    for (std::size_t i = 0; i < workloadInfo.m_InputTensorInfos.size(); ++i)
    {
        if (workloadInfo.m_InputTensorInfos[i].GetNumElements() !=
            workloadInfo.m_OutputTensorInfos[i].GetNumElements())
        {
            throw InvalidArgumentException(fmt::format(
                "Number of elements for tensor input and output {} does not match", i ));
        }
    }

    if (m_Inputs.size() != 1)
    {
        throw InvalidArgumentException(fmt::format("Number of inputs ({}) is not 1.", m_Inputs.size()));
    }

    if (m_Inputs.size() != m_Outputs.size())
    {
        throw InvalidArgumentException(fmt::format(
            "Number of inputs ({0}) does not match the number of outputs ({1})",
            m_Inputs.size(), m_Outputs.size()));
    }

    for (unsigned int i = 0; i < m_Inputs.size(); ++i)
    {
        if (!m_Inputs[i])
        {
            throw InvalidArgumentException(fmt::format("Invalid null input {}", i));
        }

        if (!m_Outputs[i])
        {
            throw InvalidArgumentException(fmt::format("Invalid null output {}", i));
        }
    }
}

//---------------------------------------------------------------
void MemSyncQueueDescriptor::Validate(const WorkloadInfo& workloadInfo) const
{
    ValidateNumInputs(workloadInfo, "MemSyncQueueDescriptor", 1);

    if (m_Inputs.size() != 1)
    {
        throw InvalidArgumentException(fmt::format("Number of inputs ({}) is not 1.", m_Inputs.size()));
    }

    if (m_Outputs.size() != 0)
    {
        throw InvalidArgumentException(fmt::format("Number of outputs ({}) is not 0.", m_Outputs.size()));
    }

    if (!m_Inputs[0])
    {
        throw InvalidArgumentException(fmt::format("Invalid null input 0"));
    }
}

//---------------------------------------------------------------
void ActivationQueueDescriptor::Validate(const WorkloadInfo& workloadInfo) const
{
    const std::string descriptorName{"ActivationQueueDescriptor"};

    ValidateNumInputs(workloadInfo,  descriptorName, 1);
    ValidateNumOutputs(workloadInfo, descriptorName, 1);

    const TensorInfo& inputTensorInfo  = workloadInfo.m_InputTensorInfos[0];
    const TensorInfo& outputTensorInfo = workloadInfo.m_OutputTensorInfos[0];

    std::vector<DataType> supportedTypes =
    {
        DataType::BFloat16,
        DataType::Float16,
        DataType::Float32,
        DataType::QAsymmS8,
        DataType::QAsymmU8,
        DataType::QSymmS16
    };

    ValidateDataTypes(inputTensorInfo, supportedTypes, descriptorName);
    ValidateTensorDataTypesMatch(inputTensorInfo, outputTensorInfo, descriptorName, "input", "output");
    ValidateTensorShapesMatch(inputTensorInfo, outputTensorInfo, descriptorName, "input", "output");
}

void ArgMinMaxQueueDescriptor::Validate(const WorkloadInfo& workloadInfo) const
{
    const std::string descriptorName{"ArgMinMaxQueueDescriptor"};

    ValidateNumInputs(workloadInfo,  descriptorName, 1);
    ValidateNumOutputs(workloadInfo, descriptorName, 1);

    const TensorInfo& inputTensorInfo  = workloadInfo.m_InputTensorInfos[0];
    const TensorInfo& outputTensorInfo = workloadInfo.m_OutputTensorInfos[0];

    if (outputTensorInfo.GetDataType() != DataType::Signed32 &&
        outputTensorInfo.GetDataType() != DataType::Signed64)
    {
        throw InvalidArgumentException(descriptorName + ": Output of ArgMinMax layer must be Int32 or Int64.");
    }

    std::vector<DataType> supportedInputTypes =
    {
        DataType::BFloat16,
        DataType::Float16,
        DataType::Float32,
        DataType::QAsymmS8,
        DataType::QAsymmU8,
        DataType::QSymmS16,
        DataType::Signed32,
        DataType::Signed64
    };

    ValidateDataTypes(inputTensorInfo, supportedInputTypes, descriptorName);

    auto inputShape = inputTensorInfo.GetShape();
    auto outputShape = outputTensorInfo.GetShape();

    auto inputNumDimensions = inputShape.GetNumDimensions();
    auto unsignedAxis = armnnUtils::GetUnsignedAxis(inputNumDimensions, m_Parameters.m_Axis);

    const std::string outputShapeError{": Output tensor shape does not match shape inferred from input tensor."};

    // 1D input shape results in scalar output shape
    if (inputShape.GetNumDimensions() == 1)
    {
        if (outputShape.GetNumDimensions() != 1 && outputShape[0] != 1)
        {
            throw InvalidArgumentException(descriptorName + outputShapeError);
        }
    }
    else
    {
        for (unsigned int i = 0; i < unsignedAxis; ++i)
        {
            if (outputShape[i] != inputShape[i])
            {
                throw InvalidArgumentException(descriptorName + outputShapeError);
            }
        }

        for (auto i = unsignedAxis + 1; i < inputNumDimensions; ++i)
        {
            if (outputShape[i - 1] != inputShape[i])
            {
                throw InvalidArgumentException(descriptorName + outputShapeError);
            }
        }
    }
}

void CastQueueDescriptor::Validate(const WorkloadInfo& workloadInfo) const
{
    const std::string descriptorName{"CastQueueDescriptor"};

    ValidateNumInputs(workloadInfo,  descriptorName, 1);
    ValidateNumOutputs(workloadInfo, descriptorName, 1);

    const TensorInfo& inputTensorInfo  = workloadInfo.m_InputTensorInfos[0];
    const TensorInfo& outputTensorInfo = workloadInfo.m_OutputTensorInfos[0];

    std::vector<DataType> supportedTypes =
    {
            DataType::BFloat16,
            DataType::Float16,
            DataType::Float32,
            DataType::QAsymmS8,
            DataType::QAsymmU8,
            DataType::QSymmS8,
            DataType::QSymmS16,
            DataType::Signed32,
            DataType::Signed64
    };

    ValidateDataTypes(inputTensorInfo, supportedTypes, descriptorName);
    ValidateTensorShapesMatch(inputTensorInfo, outputTensorInfo, descriptorName, "input", "output");
}

void SoftmaxQueueDescriptor::Validate(const WorkloadInfo& workloadInfo) const
{
    const std::string descriptorName{"SoftmaxQueueDescriptor"};

    ValidateNumInputs(workloadInfo,  descriptorName, 1);
    ValidateNumOutputs(workloadInfo, descriptorName, 1);

    const TensorInfo& inputTensorInfo  = workloadInfo.m_InputTensorInfos[0];
    const TensorInfo& outputTensorInfo = workloadInfo.m_OutputTensorInfos[0];

    std::vector<DataType> supportedTypes =
    {
        DataType::BFloat16,
        DataType::Float16,
        DataType::Float32,
        DataType::QAsymmS8,
        DataType::QAsymmU8,
        DataType::QSymmS16
    };

    ValidateDataTypes(inputTensorInfo, supportedTypes, descriptorName);
    ValidateTensorDataTypesMatch(inputTensorInfo, outputTensorInfo, descriptorName, "input", "output");
    ValidateTensorShapesMatch(inputTensorInfo, outputTensorInfo, descriptorName, "input", "output");
}

void SplitterQueueDescriptor::Validate(const WorkloadInfo& workloadInfo) const
{
    const std::string descriptorName{"SplitterQueueDescriptor"};

    ValidateNumInputs(workloadInfo, descriptorName, 1);

    // Check the supported data types
    std::vector<DataType> supportedTypes =
    {
        DataType::BFloat16,
        DataType::Float32,
        DataType::Float16,
        DataType::Boolean,
        DataType::Signed32,
        DataType::QAsymmS8,
        DataType::QAsymmU8,
        DataType::QSymmS16
    };

    const TensorInfo& inputTensorInfo = workloadInfo.m_InputTensorInfos[0];
    for (unsigned long i = 0ul; i < workloadInfo.m_OutputTensorInfos.size(); ++i)
    {
        const TensorInfo& outputTensorInfo = workloadInfo.m_OutputTensorInfos[i];
        ValidateDataTypes(outputTensorInfo, supportedTypes, descriptorName);

        const std::string outputName = "output_" + std::to_string(i);
        ValidateTensorDataTypesMatch(inputTensorInfo, outputTensorInfo, descriptorName, "input", outputName);
    }

    if (workloadInfo.m_OutputTensorInfos.size() <= 0)
    {
        throw InvalidArgumentException(descriptorName + ": At least one output needs to be provided.");
    }

    if (workloadInfo.m_OutputTensorInfos.size() != m_ViewOrigins.size())
    {
        throw InvalidArgumentException(
            descriptorName + ": Number of split windows "
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
            throw InvalidArgumentException(descriptorName + ": Window origin have to "
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
                throw InvalidArgumentException(descriptorName + ": Window extent coordinates have to "
                                               "be smaller or equal than the size of the input in that coord.");
            }
        }
    }
}

void ConcatQueueDescriptor::Validate(const WorkloadInfo& workloadInfo) const
{
    const std::string descriptorName{"ConcatQueueDescriptor"};

    ValidateNumOutputs(workloadInfo, descriptorName, 1);

    if (m_Inputs.size() <= 0)
    {
        throw InvalidArgumentException(descriptorName + ": At least one input needs to be provided.");
    }
    if (m_Outputs.size() <= 0)
    {
        throw InvalidArgumentException(descriptorName + ": At least one output needs to be provided.");
    }

    if (workloadInfo.m_InputTensorInfos.size() <= 0)
    {
        throw InvalidArgumentException(descriptorName + ": At least one TensorInfo input needs to be provided.");
    }
    if (workloadInfo.m_OutputTensorInfos.size() <= 0)
    {
        throw InvalidArgumentException(descriptorName + ": At least one TensorInfo output needs to be provided.");
    }

    if(m_Parameters.GetConcatAxis() > workloadInfo.m_InputTensorInfos[0].GetShape().GetNumDimensions())
    {
        throw InvalidArgumentException(descriptorName + ": Invalid concatenation axis provided.");
    }

    if (workloadInfo.m_InputTensorInfos[0].GetShape().GetNumDimensions() - m_Parameters.GetConcatAxis() == 1)
    {
        return;
    }

    if (workloadInfo.m_InputTensorInfos.size() != m_ViewOrigins.size())
    {
        throw InvalidArgumentException(
            descriptorName + ": Number of split windows "
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
            throw InvalidArgumentException(descriptorName + ": Window origin have to "
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
                throw InvalidArgumentException(descriptorName + ": Window extent coordinates have to "
                                               "be smaller or equal than the size of the output in that coord.");
            }
        }
    }

    // Check the supported data types
    std::vector<DataType> supportedTypes =
    {
        DataType::BFloat16,
        DataType::Float32,
        DataType::Float16,
        DataType::Boolean,
        DataType::Signed32,
        DataType::QAsymmS8,
        DataType::QAsymmU8,
        DataType::QSymmS16
    };

    const TensorInfo& outputTensorInfo = workloadInfo.m_OutputTensorInfos[0];
    for (unsigned long i = 0ul; i < workloadInfo.m_InputTensorInfos.size(); ++i)
    {
        const TensorInfo& inputTensorInfo = workloadInfo.m_InputTensorInfos[i];
        ValidateDataTypes(inputTensorInfo, supportedTypes, descriptorName);

        const std::string inputName = "input_" + std::to_string(i);
        ValidateTensorDataTypesMatch(inputTensorInfo, outputTensorInfo, descriptorName, inputName, "output");
    }
}

void StackQueueDescriptor::Validate(const WorkloadInfo& workloadInfo) const
{
    const std::string descriptorName{"StackQueueDescriptor"};

    ValidateNumOutputs(workloadInfo, descriptorName, 1);

    if (m_Parameters.m_NumInputs != workloadInfo.m_InputTensorInfos.size())
    {
        throw InvalidArgumentException(descriptorName + ": Must have the defined number of input tensors.");
    }

    // All inputs must have the same shape, which is defined in parameters
    const TensorShape& inputShape = m_Parameters.m_InputShape;
    for (unsigned int i = 0; i < workloadInfo.m_InputTensorInfos.size(); ++i)
    {
        if (workloadInfo.m_InputTensorInfos[i].GetShape() != inputShape)
        {
            throw InvalidArgumentException(descriptorName + ": All input tensor shapes must match the defined shape.");
        }
    }

    if (inputShape.GetNumDimensions() > 4)
    {
        throw InvalidArgumentException(descriptorName + ": Input tensor may have up to 4 dimensions.");
    }

    // m_Axis is 0-based and may take values from 0 to the number of input dimensions (inclusive),
    // since the output tensor has an additional dimension.
    if (m_Parameters.m_Axis > inputShape.GetNumDimensions())
    {
        throw InvalidArgumentException(descriptorName + ": Axis may not be greater "
                                       "than the number of input dimensions.");
    }

    // Output shape must be as inferred from the input shape
    const TensorShape& outputShape = workloadInfo.m_OutputTensorInfos[0].GetShape();
    for (unsigned int i = 0; i < m_Parameters.m_Axis; ++i)
    {
        if (outputShape[i] != inputShape[i])
        {
            throw InvalidArgumentException(descriptorName + ": Output tensor must "
                                           "match shape inferred from input tensor.");
        }
    }

    if (outputShape[m_Parameters.m_Axis] != m_Parameters.m_NumInputs)
    {
        throw InvalidArgumentException(descriptorName + ": Output tensor must "
                                       "match shape inferred from input tensor.");
    }

    for (unsigned int i = m_Parameters.m_Axis + 1; i < inputShape.GetNumDimensions() + 1; ++i)
    {
        if (outputShape[i] != inputShape[i-1])
        {
            throw InvalidArgumentException(descriptorName + ": Output tensor must "
                                           "match shape inferred from input tensor.");
        }
    }

    if (outputShape.GetNumDimensions() > 5)
    {
        throw InvalidArgumentException(descriptorName + ": Output tensor may have up to 5 dimensions.");
    }

    // Check the supported data types
    std::vector<DataType> supportedTypes =
    {
        DataType::BFloat16,
        DataType::Float32,
        DataType::Float16,
        DataType::Boolean,
        DataType::Signed32,
        DataType::QAsymmS8,
        DataType::QAsymmU8,
        DataType::QSymmS16
    };

    ValidateDataTypes(workloadInfo.m_InputTensorInfos[0], supportedTypes, descriptorName);

    for (unsigned int i = 1ul; i < workloadInfo.m_InputTensorInfos.size(); ++i)
    {
        ValidateTensorDataTypesMatch(workloadInfo.m_InputTensorInfos[0],
                                     workloadInfo.m_InputTensorInfos[i],
                                     descriptorName,
                                     "input_0",
                                     "input_" + std::to_string(i));
    }

    ValidateTensorDataTypesMatch(workloadInfo.m_InputTensorInfos[0],
                                 workloadInfo.m_OutputTensorInfos[0],
                                 descriptorName,
                                 "input_0",
                                 "output");
}

void FillQueueDescriptor::Validate(const WorkloadInfo& workloadInfo) const
{
    const std::string descriptorName{"FillQueueDescriptor"};

    ValidateNumInputs(workloadInfo,  descriptorName, 1);
    ValidateNumOutputs(workloadInfo, descriptorName, 1);

    const TensorInfo& inputTensorInfo = workloadInfo.m_InputTensorInfos[0];
    const TensorInfo& outputTensorInfo = workloadInfo.m_OutputTensorInfos[0];

    ValidateTensorNumDimensions(inputTensorInfo, descriptorName, 1, "input");

    std::vector<DataType> supportedTypes =
    {
        DataType::BFloat16,
        DataType::Float32,
        DataType::Float16,
        DataType::Signed32
    };

    ValidateDataTypes(outputTensorInfo, supportedTypes, descriptorName);
}

void FullyConnectedQueueDescriptor::Validate(const WorkloadInfo& workloadInfo) const
{
    const std::string descriptorName{"FullyConnectedQueueDescriptor"};

    uint32_t numInputs = 2;
    if (m_Parameters.m_BiasEnabled)
    {
        numInputs = 3;
    }

    ValidateNumInputs(workloadInfo, descriptorName, numInputs);
    ValidateNumOutputs(workloadInfo, descriptorName, 1);

    const TensorInfo& inputTensorInfo  = workloadInfo.m_InputTensorInfos[0];
    const TensorInfo& outputTensorInfo = workloadInfo.m_OutputTensorInfos[0];

    ValidateTensorNumDimensions(outputTensorInfo, descriptorName, 2, "output");

    if (!(inputTensorInfo.GetNumDimensions() == 2 || inputTensorInfo.GetNumDimensions() == 4))
    {
        throw InvalidArgumentException(descriptorName + ": Input tensor must have 2 or 4 dimensions.");
    }

    TensorInfo weightTensorInfo = workloadInfo.m_InputTensorInfos[1];
    ValidateTensorNumDimensions(weightTensorInfo, descriptorName, 2, "weight");

    if (m_Parameters.m_BiasEnabled)
    {
        TensorInfo biasTensorInfo = workloadInfo.m_InputTensorInfos[2];
        // Validates type and quantization values.
        ValidateBiasTensorQuantization(biasTensorInfo, inputTensorInfo, weightTensorInfo, descriptorName);
        ValidateTensorDataType(biasTensorInfo, GetBiasDataType(inputTensorInfo.GetDataType()), descriptorName, "bias");
        ValidateTensorNumDimensions(biasTensorInfo, descriptorName, 1, "bias");
    }

    // Check the supported data types
    std::vector<DataType> supportedTypes =
    {
        DataType::BFloat16,
        DataType::Float32,
        DataType::Float16,
        DataType::QAsymmS8,
        DataType::QAsymmU8,
        DataType::QSymmS16
    };

    ValidateDataTypes(inputTensorInfo, supportedTypes, descriptorName);

    // For FullyConnected, we allow to have BFloat16 input with Float32 output for optimization.
    if (inputTensorInfo.GetDataType() == DataType::BFloat16)
    {
        if (outputTensorInfo.GetDataType() != DataType::BFloat16 && outputTensorInfo.GetDataType() != DataType::Float32)
        {
            throw InvalidArgumentException(descriptorName  + ": " + " Output tensor type must be BFloat16 or Float32 "
                                           "for BFloat16 input.");
        }
    }
    else
    {
        ValidateTensorDataTypesMatch(inputTensorInfo, outputTensorInfo, descriptorName, "input", "output");
    }
}

void NormalizationQueueDescriptor::Validate(const WorkloadInfo& workloadInfo) const
{
    const std::string descriptorName{"NormalizationQueueDescriptor"};

    ValidateNumInputs(workloadInfo,  descriptorName, 1);
    ValidateNumOutputs(workloadInfo, descriptorName, 1);

    const TensorInfo& inputTensorInfo  = workloadInfo.m_InputTensorInfos[0];
    const TensorInfo& outputTensorInfo = workloadInfo.m_OutputTensorInfos[0];

    // Check the supported data types
    std::vector<DataType> supportedTypes =
    {
        DataType::BFloat16,
        DataType::Float16,
        DataType::Float32,
        DataType::QAsymmS8,
        DataType::QAsymmU8,
        DataType::QSymmS16
    };

    ValidateDataTypes(inputTensorInfo, supportedTypes, descriptorName);

    ValidateTensorDataTypesMatch(inputTensorInfo, outputTensorInfo, descriptorName, "input", "output");

    ValidateTensorShapesMatch(inputTensorInfo, outputTensorInfo, descriptorName, "input", "output");
}

void AdditionQueueDescriptor::Validate(const WorkloadInfo& workloadInfo) const
{
    const std::string descriptorName{"AdditionQueueDescriptor"};

    ValidateNumInputs(workloadInfo,  descriptorName, 2);
    ValidateNumOutputs(workloadInfo, descriptorName, 1);

    const TensorInfo& inputTensorInfo0 = workloadInfo.m_InputTensorInfos[0];
    const TensorInfo& inputTensorInfo1 = workloadInfo.m_InputTensorInfos[1];
    const TensorInfo& outputTensorInfo = workloadInfo.m_OutputTensorInfos[0];

    std::vector<DataType> supportedTypes =
    {
        DataType::BFloat16,
        DataType::Float32,
        DataType::Float16,
        DataType::QAsymmS8,
        DataType::QAsymmU8,
        DataType::QSymmS16,
        DataType::Signed32
    };

    ValidateDataTypes(inputTensorInfo0, supportedTypes, descriptorName);
    ValidateDataTypes(inputTensorInfo1, supportedTypes, descriptorName);
    ValidateDataTypes(outputTensorInfo, supportedTypes, descriptorName);

    ValidateTensorDataTypesMatch(inputTensorInfo0, inputTensorInfo1, descriptorName, "input_0", "input_1");
    ValidateTensorDataTypesMatch(inputTensorInfo1, outputTensorInfo, descriptorName, "input_1", "output");

    ValidateBroadcastTensorShapesMatch(inputTensorInfo0,
                                       inputTensorInfo1,
                                       outputTensorInfo,
                                       descriptorName,
                                       "input_0",
                                       "input_1");
}

void MultiplicationQueueDescriptor::Validate(const WorkloadInfo& workloadInfo) const
{
    const std::string descriptorName{"MultiplicationQueueDescriptor"};

    ValidateNumInputs(workloadInfo,  descriptorName, 2);
    ValidateNumOutputs(workloadInfo, descriptorName, 1);

    const TensorInfo& inputTensorInfo0 = workloadInfo.m_InputTensorInfos[0];
    const TensorInfo& inputTensorInfo1 = workloadInfo.m_InputTensorInfos[1];
    const TensorInfo& outputTensorInfo = workloadInfo.m_OutputTensorInfos[0];

    std::vector<DataType> supportedTypes =
    {
        DataType::BFloat16,
        DataType::Float16,
        DataType::Float32,
        DataType::QAsymmS8,
        DataType::QAsymmU8,
        DataType::QSymmS16,
        DataType::Signed32
    };

    ValidateDataTypes(inputTensorInfo0, supportedTypes, descriptorName);
    ValidateDataTypes(inputTensorInfo1, supportedTypes, descriptorName);
    ValidateDataTypes(outputTensorInfo, supportedTypes, descriptorName);

    ValidateTensorDataTypesMatch(inputTensorInfo0, inputTensorInfo1, descriptorName, "input_0", "input_1");
    ValidateTensorDataTypesMatch(inputTensorInfo1, outputTensorInfo, descriptorName, "input_1", "output");

    ValidateBroadcastTensorShapesMatch(inputTensorInfo0,
                                       inputTensorInfo1,
                                       outputTensorInfo,
                                       descriptorName,
                                       "input_0",
                                       "input_1");
}

void BatchNormalizationQueueDescriptor::Validate(const WorkloadInfo& workloadInfo) const
{
    const std::string descriptorName{"BatchNormalizationQueueDescriptor"};

    ValidateNumInputs(workloadInfo,  descriptorName, 1);
    ValidateNumOutputs(workloadInfo, descriptorName, 1);

    const TensorInfo& inputTensorInfo  = workloadInfo.m_InputTensorInfos[0];
    const TensorInfo& outputTensorInfo = workloadInfo.m_OutputTensorInfos[0];

    std::vector<DataType> supportedTypes =
    {
        DataType::BFloat16,
        DataType::Float16,
        DataType::Float32,
        DataType::QAsymmS8,
        DataType::QAsymmU8,
        DataType::QSymmS16
    };

    ValidateDataTypes(inputTensorInfo,  supportedTypes, descriptorName);
    ValidateDataTypes(outputTensorInfo, supportedTypes, descriptorName);

    ValidateTensorDataTypesMatch(inputTensorInfo, outputTensorInfo, descriptorName, "input", "output");
    ValidateTensorShapesMatch(inputTensorInfo, outputTensorInfo, descriptorName, "input", "output");

    ValidatePointer(m_Mean,     descriptorName, "mean");
    ValidatePointer(m_Variance, descriptorName, "variance");
    ValidatePointer(m_Beta,     descriptorName, "beta");
    ValidatePointer(m_Gamma,    descriptorName, "gamma");

    const TensorInfo& mean     = m_Mean->GetTensorInfo();
    const TensorInfo& variance = m_Variance->GetTensorInfo();
    const TensorInfo& beta     = m_Beta->GetTensorInfo();
    const TensorInfo& gamma    = m_Gamma->GetTensorInfo();

    ValidateTensorNumDimensions(mean,     descriptorName, 1, "mean");
    ValidateTensorNumDimensions(variance, descriptorName, 1, "variance");
    ValidateTensorNumDimensions(beta,     descriptorName, 1, "beta");
    ValidateTensorNumDimensions(gamma,    descriptorName, 1, "gamma");

    ValidateTensorShapesMatch(mean, variance, descriptorName, "mean", "variance");
    ValidateTensorShapesMatch(mean, beta,     descriptorName, "mean", "beta");
    ValidateTensorShapesMatch(mean, gamma,    descriptorName, "mean", "gamma");
}

void Convolution2dQueueDescriptor::Validate(const WorkloadInfo& workloadInfo) const
{
    const std::string descriptorName{"Convolution2dQueueDescriptor"};

    ValidateNumInputs(workloadInfo,  descriptorName, 1);
    ValidateNumOutputs(workloadInfo, descriptorName, 1);

    const TensorInfo& inputTensorInfo  = workloadInfo.m_InputTensorInfos[0];
    const TensorInfo& outputTensorInfo = workloadInfo.m_OutputTensorInfos[0];

    ValidateTensorNumDimensions(inputTensorInfo,  descriptorName, 4, "input");
    ValidateTensorNumDimensions(outputTensorInfo, descriptorName, 4, "output");

    ValidatePointer(m_Weight, descriptorName, "weight");

    const TensorInfo& weightTensorInfo = m_Weight->GetTensorInfo();
    ValidateTensorNumDimensions(weightTensorInfo, descriptorName, 4, "weight");

    ValidateWeightDataType(inputTensorInfo, weightTensorInfo, descriptorName);

    Optional<TensorInfo> optionalBiasTensorInfo;
    if (m_Parameters.m_BiasEnabled)
    {
        ValidatePointer(m_Bias, descriptorName, "bias");

        optionalBiasTensorInfo = MakeOptional<TensorInfo>(m_Bias->GetTensorInfo());
        const TensorInfo& biasTensorInfo = optionalBiasTensorInfo.value();

        ValidateTensorDataType(biasTensorInfo, GetBiasDataType(inputTensorInfo.GetDataType()), descriptorName, "bias");
        ValidateBiasTensorQuantization(biasTensorInfo, inputTensorInfo, weightTensorInfo, descriptorName);
    }

    if (m_Parameters.m_StrideX <= 0 || m_Parameters.m_StrideY <= 0  )
    {
        throw InvalidArgumentException(
            fmt::format("{}: strideX (provided {}) and strideY (provided {}) "
                        "cannot be either negative or 0.",
                        descriptorName, m_Parameters.m_StrideX, m_Parameters.m_StrideY));
    }

    ValidatePerAxisQuantization(inputTensorInfo,
                                outputTensorInfo,
                                weightTensorInfo,
                                optionalBiasTensorInfo,
                                descriptorName);

    std::vector<DataType> supportedTypes =
    {
        DataType::BFloat16,
        DataType::Float16,
        DataType::Float32,
        DataType::QAsymmS8,
        DataType::QAsymmU8,
        DataType::QSymmS16,
        DataType::QSymmS8
    };

    ValidateDataTypes(inputTensorInfo, supportedTypes, descriptorName);

    // For Convolution2d, we allow to have BFloat16 input with Float32 output for optimization.
    if (inputTensorInfo.GetDataType() == DataType::BFloat16)
    {
        if (outputTensorInfo.GetDataType() != DataType::BFloat16 && outputTensorInfo.GetDataType() != DataType::Float32)
        {
            throw InvalidArgumentException(descriptorName  + ": " + " Output tensor type must be BFloat16 or Float32 "
                                           "for BFloat16 input.");
        }
    }
    else
    {
        ValidateTensorDataTypesMatch(inputTensorInfo, outputTensorInfo, descriptorName, "input", "output");
    }
}

void Convolution3dQueueDescriptor::Validate(const WorkloadInfo& workloadInfo) const
{
    const std::string descriptorName{"Convolution3dQueueDescriptor"};

    uint32_t numInputs = 2;
    if (m_Parameters.m_BiasEnabled)
    {
        numInputs = 3;
    }
    ValidateNumInputs(workloadInfo,  descriptorName, numInputs);
    ValidateNumOutputs(workloadInfo, descriptorName, 1);

    const TensorInfo& inputTensorInfo  = workloadInfo.m_InputTensorInfos[0];
    const TensorInfo& outputTensorInfo = workloadInfo.m_OutputTensorInfos[0];

    ValidateTensorNumDimensions(inputTensorInfo,  descriptorName, 5, "input");
    ValidateTensorNumDimensions(outputTensorInfo, descriptorName, 5, "output");

    const TensorInfo& weightTensorInfo = workloadInfo.m_InputTensorInfos[1];
    ValidateTensorNumDimensions(weightTensorInfo, descriptorName, 5, "weight");

    ValidateWeightDataType(inputTensorInfo, weightTensorInfo, descriptorName);

    Optional<TensorInfo> optionalBiasTensorInfo;
    if (m_Parameters.m_BiasEnabled)
    {
        optionalBiasTensorInfo = MakeOptional<TensorInfo>(workloadInfo.m_InputTensorInfos[2]);
        const TensorInfo& biasTensorInfo = optionalBiasTensorInfo.value();

        ValidateTensorDataType(biasTensorInfo, GetBiasDataType(inputTensorInfo.GetDataType()), descriptorName, "bias");
        ValidateBiasTensorQuantization(biasTensorInfo, inputTensorInfo, weightTensorInfo, descriptorName);
    }

    if (m_Parameters.m_StrideX <= 0 || m_Parameters.m_StrideY <= 0 || m_Parameters.m_StrideZ <= 0 )
    {
        throw InvalidArgumentException(
                fmt::format("{}: strideX (provided {}), strideY (provided {}) or strideZ (provided {})"
                            "cannot be either negative or 0.",
                            descriptorName, m_Parameters.m_StrideX, m_Parameters.m_StrideY, m_Parameters.m_StrideZ));
    }

    ValidatePerAxisQuantization(inputTensorInfo,
                                outputTensorInfo,
                                weightTensorInfo,
                                optionalBiasTensorInfo,
                                descriptorName);

    std::vector<DataType> supportedTypes =
    {
        DataType::BFloat16,
        DataType::Float16,
        DataType::Float32,
        DataType::QAsymmS8,
        DataType::QAsymmU8,
        DataType::QSymmS16,
        DataType::QSymmS8
    };

    ValidateDataTypes(inputTensorInfo, supportedTypes, descriptorName);
    ValidateTensorDataTypesMatch(inputTensorInfo, outputTensorInfo, descriptorName, "input", "output");
}

void DepthwiseConvolution2dQueueDescriptor::Validate(const WorkloadInfo& workloadInfo) const
{
    const std::string descriptorName{"DepthwiseConvolution2dQueueDescriptor"};

    ValidateNumInputs(workloadInfo,  descriptorName, 1);
    ValidateNumOutputs(workloadInfo, descriptorName, 1);

    const TensorInfo& inputTensorInfo  = workloadInfo.m_InputTensorInfos[0];
    const TensorInfo& outputTensorInfo = workloadInfo.m_OutputTensorInfos[0];

    ValidateTensorNumDimensions(inputTensorInfo,  descriptorName, 4, "input");
    ValidateTensorNumDimensions(outputTensorInfo, descriptorName, 4, "output");

    ValidatePointer(m_Weight, descriptorName, "weight");

    const TensorInfo& weightTensorInfo = m_Weight->GetTensorInfo();
    ValidateTensorNumDimensions(weightTensorInfo, descriptorName, 4, "weight");

    if (m_Parameters.m_DilationX < 1 || m_Parameters.m_DilationY < 1 )
    {
        throw InvalidArgumentException(
            fmt::format("{}: dilationX (provided {}) and dilationY (provided {}) "
                        "cannot be smaller than 1.",
                        descriptorName, m_Parameters.m_DilationX, m_Parameters.m_DilationX));
    }

    if (m_Parameters.m_StrideX <= 0 || m_Parameters.m_StrideY <= 0  )
    {
        throw InvalidArgumentException(
            fmt::format("{}: strideX (provided {}) and strideY (provided {}) "
                        "cannot be either negative or 0.",
                        descriptorName, m_Parameters.m_StrideX, m_Parameters.m_StrideY));
    }

    const unsigned int channelIndex = (m_Parameters.m_DataLayout == DataLayout::NCHW) ? 1 : 3;

    // Expected weight shape: [ 1, H, W, I*M ] - This shape does NOT depend on the data layout
    // inputChannels * channelMultiplier should be equal to outputChannels.
    const unsigned int numWeightOutputChannels = weightTensorInfo.GetShape()[3]; // I*M=Cout
    const unsigned int numOutputChannels       = outputTensorInfo.GetShape()[channelIndex];
    if (numWeightOutputChannels != numOutputChannels)
    {
        throw InvalidArgumentException(fmt::format(
            "{0}: The weight format in armnn is expected to be [1, H, W, Cout]."
            "But 4th dimension is not equal to Cout. Cout = {1} Provided weight shape: [{2}, {3}, {4}, {5}]",
            descriptorName,
            numOutputChannels,
            weightTensorInfo.GetShape()[0],
            weightTensorInfo.GetShape()[1],
            weightTensorInfo.GetShape()[2],
            weightTensorInfo.GetShape()[3]));
    }
    if (weightTensorInfo.GetShape()[0] != 1)
    {
        throw InvalidArgumentException(fmt::format(
                "{0}: The weight format in armnn is expected to be [1, H, W, Cout]."
                "But first dimension is not equal to 1. Provided weight shape: [{1}, {2}, {3}, {4}]",
                descriptorName,
                weightTensorInfo.GetShape()[0],
                weightTensorInfo.GetShape()[1],
                weightTensorInfo.GetShape()[2],
                weightTensorInfo.GetShape()[3]));
    }

    ValidateWeightDataType(inputTensorInfo, weightTensorInfo, descriptorName);

    Optional<TensorInfo> optionalBiasTensorInfo;
    if (m_Parameters.m_BiasEnabled)
    {
        ValidatePointer(m_Bias, descriptorName, "bias");

        optionalBiasTensorInfo = MakeOptional<TensorInfo>(m_Bias->GetTensorInfo());
        const TensorInfo& biasTensorInfo = optionalBiasTensorInfo.value();

        ValidateBiasTensorQuantization(biasTensorInfo, inputTensorInfo, weightTensorInfo, descriptorName);
        ValidateTensorDataType(biasTensorInfo, GetBiasDataType(inputTensorInfo.GetDataType()), descriptorName, "bias");
    }
    ValidatePerAxisQuantization(inputTensorInfo,
                                outputTensorInfo,
                                weightTensorInfo,
                                optionalBiasTensorInfo,
                                descriptorName);

    std::vector<DataType> supportedTypes =
    {
        DataType::BFloat16,
        DataType::Float16,
        DataType::Float32,
        DataType::QAsymmS8,
        DataType::QAsymmU8,
        DataType::QSymmS16
    };

    ValidateDataTypes(inputTensorInfo, supportedTypes, descriptorName);
    ValidateTensorDataTypesMatch(inputTensorInfo, outputTensorInfo, descriptorName, "input", "output");
}

void PermuteQueueDescriptor::Validate(const WorkloadInfo& workloadInfo) const
{
    const std::string descriptorName{"PermuteQueueDescriptor"};

    ValidateNumInputs(workloadInfo,  descriptorName, 1);
    ValidateNumOutputs(workloadInfo, descriptorName, 1);

    const PermutationVector& mapping = m_Parameters.m_DimMappings;

    const TensorInfo& inputTensorInfo  = workloadInfo.m_InputTensorInfos[0];
    const TensorInfo& outputTensorInfo = workloadInfo.m_OutputTensorInfos[0];

    ValidateTensorNumDimensions(inputTensorInfo,  descriptorName, mapping.GetSize(), "input");
    ValidateTensorNumDimensions(outputTensorInfo, descriptorName, mapping.GetSize(), "output");

    for (unsigned int i = 0u; i < mapping.GetSize(); ++i)
    {
        if (inputTensorInfo.GetShape()[i] != outputTensorInfo.GetShape()[mapping[i]])
        {
            throw InvalidArgumentException(descriptorName + ": src dimension " + to_string(i) +
                                           " (=" + to_string(inputTensorInfo.GetShape()[i]) + ") " +
                                           "must match dst dimension " + to_string(mapping[i]) +
                                           " (=" + to_string(outputTensorInfo.GetShape()[mapping[i]]) + ")");
        }
    }

    ValidateTensorDataTypesMatch(inputTensorInfo, outputTensorInfo, descriptorName, "input", "output");
}

void Pooling2dQueueDescriptor::Validate(const WorkloadInfo& workloadInfo) const
{
    const std::string descriptorName{"Pooling2dQueueDescriptor"};

    ValidateNumInputs(workloadInfo,  descriptorName, 1);
    ValidateNumOutputs(workloadInfo, descriptorName, 1);

    const TensorInfo& inputTensorInfo  = workloadInfo.m_InputTensorInfos[0];
    const TensorInfo& outputTensorInfo = workloadInfo.m_OutputTensorInfos[0];

    ValidateTensorNumDimensions(inputTensorInfo,  descriptorName, 4, "input");
    ValidateTensorNumDimensions(outputTensorInfo, descriptorName, 4, "output");

    std::vector<DataType> supportedTypes =
    {
        DataType::BFloat16,
        DataType::Float32,
        DataType::Float16,
        DataType::QAsymmS8,
        DataType::QAsymmU8,
        DataType::QSymmS16
    };

    ValidateDataTypes(inputTensorInfo, supportedTypes, descriptorName);
    ValidateTensorDataTypesMatch(inputTensorInfo, outputTensorInfo, descriptorName, "input", "output");
}

void Pooling3dQueueDescriptor::Validate(const WorkloadInfo& workloadInfo) const
{
    const std::string descriptorName{"Pooling3dQueueDescriptor"};

    ValidateNumInputs(workloadInfo,  descriptorName, 1);
    ValidateNumOutputs(workloadInfo, descriptorName, 1);

    const TensorInfo& inputTensorInfo  = workloadInfo.m_InputTensorInfos[0];
    const TensorInfo& outputTensorInfo = workloadInfo.m_OutputTensorInfos[0];

    ValidateTensorNumDimensions(inputTensorInfo,  descriptorName, 5, "input");
    ValidateTensorNumDimensions(outputTensorInfo, descriptorName, 5, "output");

    std::vector<DataType> supportedTypes =
    {
        DataType::BFloat16,
        DataType::Float32,
        DataType::Float16,
        DataType::QAsymmS8,
        DataType::QAsymmU8,
        DataType::QSymmS16
    };

    ValidateDataTypes(inputTensorInfo, supportedTypes, descriptorName);
    ValidateTensorDataTypesMatch(inputTensorInfo, outputTensorInfo, descriptorName, "input", "output");
}


void ResizeBilinearQueueDescriptor::Validate(const WorkloadInfo& workloadInfo) const
{
    const std::string descriptorName{"ResizeBilinearQueueDescriptor"};

    ValidateNumInputs(workloadInfo,  descriptorName, 1);
    ValidateNumOutputs(workloadInfo, descriptorName, 1);

    const TensorInfo& inputTensorInfo  = workloadInfo.m_InputTensorInfos[0];
    const TensorInfo& outputTensorInfo = workloadInfo.m_OutputTensorInfos[0];

    ValidateTensorNumDimensions(inputTensorInfo,  descriptorName, 4, "input");
    ValidateTensorNumDimensions(outputTensorInfo, descriptorName, 4, "output");

    std::vector<DataType> supportedTypes =
    {
        DataType::BFloat16,
        DataType::Float16,
        DataType::Float32,
        DataType::QAsymmS8,
        DataType::QAsymmU8,
        DataType::QSymmS16
    };

    ValidateDataTypes(inputTensorInfo, supportedTypes, descriptorName);
    ValidateTensorDataTypesMatch(inputTensorInfo, outputTensorInfo, descriptorName, "input", "output");

    // ResizeBilinear only changes width and height: batch and channel count must match.
    const unsigned int inputBatchSize  = inputTensorInfo.GetShape()[0];
    const unsigned int outputBatchSize = outputTensorInfo.GetShape()[0];
    if (inputBatchSize != outputBatchSize)
    {
        throw InvalidArgumentException(
            fmt::format("{}: Input batch size ({}) does not match output batch size ({})",
                        descriptorName, inputBatchSize, outputBatchSize));
    }

    DataLayoutIndexed dimensionIndices(m_Parameters.m_DataLayout);
    const unsigned int inputChannelCount  = inputTensorInfo.GetShape()[dimensionIndices.GetChannelsIndex()];
    const unsigned int outputChannelCount = outputTensorInfo.GetShape()[dimensionIndices.GetChannelsIndex()];
    if (inputChannelCount != outputChannelCount)
    {
        throw InvalidArgumentException(
            fmt::format("{}: Input channel count ({}) does not match output channel count ({})",
                        descriptorName, inputChannelCount, outputChannelCount));
    }
}

void ResizeQueueDescriptor::Validate(const WorkloadInfo& workloadInfo) const
{
    const std::string descriptorName{"ResizeQueueDescriptor"};

    ValidateNumInputs(workloadInfo,  descriptorName, 1);
    ValidateNumOutputs(workloadInfo, descriptorName, 1);

    const TensorInfo& inputTensorInfo  = workloadInfo.m_InputTensorInfos[0];
    const TensorInfo& outputTensorInfo = workloadInfo.m_OutputTensorInfos[0];

    ValidateTensorNumDimensions(inputTensorInfo,  descriptorName, 4, "input");
    ValidateTensorNumDimensions(outputTensorInfo, descriptorName, 4, "output");

    std::vector<DataType> supportedTypes =
    {
        DataType::BFloat16,
        DataType::Float16,
        DataType::Float32,
        DataType::QAsymmS8,
        DataType::QAsymmU8,
        DataType::QSymmS16
    };

    ValidateDataTypes(inputTensorInfo, supportedTypes, descriptorName);
    ValidateTensorDataTypesMatch(inputTensorInfo, outputTensorInfo, descriptorName, "input", "output");

    // Resize only changes width and height: batch and channel count must match.
    const unsigned int inputBatchSize  = inputTensorInfo.GetShape()[0];
    const unsigned int outputBatchSize = outputTensorInfo.GetShape()[0];
    if (inputBatchSize != outputBatchSize)
    {
        throw InvalidArgumentException(
                fmt::format("{}: Input batch size ({}) does not match output batch size ({})",
                            descriptorName, inputBatchSize, outputBatchSize));
    }

    DataLayoutIndexed dimensionIndices(m_Parameters.m_DataLayout);
    const unsigned int inputChannelCount  = inputTensorInfo.GetShape()[dimensionIndices.GetChannelsIndex()];
    const unsigned int outputChannelCount = outputTensorInfo.GetShape()[dimensionIndices.GetChannelsIndex()];
    if (inputChannelCount != outputChannelCount)
    {
        throw InvalidArgumentException(
                fmt::format("{}: Input channel count ({}) does not match output channel count ({})",
                            descriptorName, inputChannelCount, outputChannelCount));
    }
}

void FakeQuantizationQueueDescriptor::Validate(const WorkloadInfo& workloadInfo) const
{
    const std::string descriptorName{"FakeQuantizationQueueDescriptor"};

    ValidateNumInputs(workloadInfo,  descriptorName, 1);
    ValidateNumOutputs(workloadInfo, descriptorName, 1);

    const TensorInfo& inputTensorInfo  = workloadInfo.m_InputTensorInfos[0];
    const TensorInfo& outputTensorInfo = workloadInfo.m_OutputTensorInfos[0];

    ValidateTensorNumDimensions(inputTensorInfo,  descriptorName, 2, "input");
    ValidateTensorNumDimensions(outputTensorInfo, descriptorName, 2, "output");

    ValidateTensorShapesMatch(inputTensorInfo, outputTensorInfo,  descriptorName, "input", "output");

    if (m_Parameters.m_Min > m_Parameters.m_Max)
    {
        throw InvalidArgumentException(descriptorName + ": min cannot be greater than max");
    }
}

void InstanceNormalizationQueueDescriptor::Validate(const WorkloadInfo& workloadInfo) const
{
    const std::string descriptorName{"InstanceNormalizationQueueDescriptor"};

    ValidateNumInputs(workloadInfo,  descriptorName, 1);
    ValidateNumOutputs(workloadInfo, descriptorName, 1);

    const TensorInfo& inputTensorInfo  = workloadInfo.m_InputTensorInfos[0];
    const TensorInfo& outputTensorInfo = workloadInfo.m_OutputTensorInfos[0];

    if (inputTensorInfo.GetNumDimensions() > 4)
    {
        throw InvalidArgumentException(descriptorName + ": Input tensors with rank greater than 4 are not supported.");
    }

    ValidateTensorShapesMatch(inputTensorInfo, outputTensorInfo, descriptorName, "input", "output");

    // Check the supported data types
    std::vector<DataType> supportedTypes =
        {
            DataType::BFloat16,
            DataType::Float32,
            DataType::Float16
        };

    ValidateDataTypes(inputTensorInfo,  supportedTypes, descriptorName);
    ValidateTensorDataTypesMatch(inputTensorInfo, outputTensorInfo, descriptorName, "input", "output");
}

void L2NormalizationQueueDescriptor::Validate(const WorkloadInfo& workloadInfo) const
{
    const std::string descriptorName{"L2NormalizationQueueDescriptor"};

    ValidateNumInputs(workloadInfo,  descriptorName, 1);
    ValidateNumOutputs(workloadInfo, descriptorName, 1);

    const TensorInfo& inputTensorInfo  = workloadInfo.m_InputTensorInfos[0];
    const TensorInfo& outputTensorInfo = workloadInfo.m_OutputTensorInfos[0];

    if (inputTensorInfo.GetNumDimensions() > 4)
    {
        throw InvalidArgumentException(descriptorName + ": Input tensors with rank greater than 4 are not supported.");
    }

    ValidateTensorShapesMatch(inputTensorInfo, outputTensorInfo, descriptorName, "input", "output");

    // Check the supported data types
    std::vector<DataType> supportedTypes =
    {
        DataType::BFloat16,
        DataType::Float32,
        DataType::Float16,
        DataType::QAsymmS8,
        DataType::QAsymmU8,
        DataType::QSymmS16
    };

    ValidateDataTypes(inputTensorInfo,  supportedTypes, descriptorName);
    ValidateTensorDataTypesMatch(inputTensorInfo, outputTensorInfo, descriptorName, "input", "output");
}

void LogSoftmaxQueueDescriptor::Validate(const WorkloadInfo& workloadInfo) const
{
    const std::string descriptorName{"LogSoftmaxQueueDescriptor"};

    ValidateNumInputs(workloadInfo,  descriptorName, 1);
    ValidateNumOutputs(workloadInfo, descriptorName, 1);

    const TensorInfo& inputTensorInfo  = workloadInfo.m_InputTensorInfos[0];
    const TensorInfo& outputTensorInfo = workloadInfo.m_OutputTensorInfos[0];

    ValidateTensorShapesMatch(inputTensorInfo, outputTensorInfo, descriptorName, "input", "output");

    std::vector<DataType> supportedTypes =
    {
        DataType::BFloat16,
        DataType::Float32,
        DataType::Float16,
    };

    ValidateDataTypes(inputTensorInfo,  supportedTypes, descriptorName);
    ValidateTensorDataTypesMatch(inputTensorInfo, outputTensorInfo, descriptorName, "input", "output");
}

void ConstantQueueDescriptor::Validate(const WorkloadInfo& workloadInfo) const
{
    const std::string descriptorName{"ConstantQueueDescriptor"};

    ValidateNumInputs(workloadInfo,  descriptorName, 0);
    ValidateNumOutputs(workloadInfo, descriptorName, 1);

    if (!m_LayerOutput)
    {
        throw InvalidArgumentException(descriptorName + ": No const input specified.");
    }

    const TensorInfo& outputTensorInfo = workloadInfo.m_OutputTensorInfos[0];
    ValidateTensorShapesMatch(m_LayerOutput->GetTensorInfo(), outputTensorInfo, descriptorName, "constant", "output");

    // Check the supported data types
    std::vector<DataType> supportedTypes =
    {
        DataType::BFloat16,
        DataType::Float32,
        DataType::Float16,
        DataType::QAsymmS8,
        DataType::QAsymmU8,
        DataType::QSymmS8,
        DataType::QSymmS16,
        DataType::Signed32
    };

    ValidateDataTypes(outputTensorInfo, supportedTypes, descriptorName);
}

void ReshapeQueueDescriptor::Validate(const WorkloadInfo& workloadInfo) const
{
    const std::string descriptorName{"ReshapeQueueDescriptor"};

    ValidateNumInputs(workloadInfo,  descriptorName, 1);
    ValidateNumOutputs(workloadInfo, descriptorName, 1);

    const TensorInfo& inputTensorInfo  = workloadInfo.m_InputTensorInfos[0];
    const TensorInfo& outputTensorInfo = workloadInfo.m_OutputTensorInfos[0];

    ValidateTensorNumElementsMatch(inputTensorInfo, outputTensorInfo, descriptorName, "input", "output");

    // Check the supported data types
    std::vector<DataType> supportedTypes =
    {
        DataType::BFloat16,
        DataType::Float32,
        DataType::Float16,
        DataType::QAsymmS8,
        DataType::QAsymmU8,
        DataType::QSymmS16,
        DataType::Signed32,
        DataType::Boolean
    };

    ValidateDataTypes(inputTensorInfo, supportedTypes, descriptorName);
    ValidateTensorDataTypesMatch(inputTensorInfo, outputTensorInfo, descriptorName, "input", "output");
}

void SpaceToBatchNdQueueDescriptor::Validate(const WorkloadInfo& workloadInfo) const
{
    const std::string descriptorName{"SpaceToBatchNdQueueDescriptor"};

    ValidateNumInputs(workloadInfo,  descriptorName, 1);
    ValidateNumOutputs(workloadInfo, descriptorName, 1);

    const TensorInfo& inputTensorInfo  = workloadInfo.m_InputTensorInfos[0];
    const TensorInfo& outputTensorInfo = workloadInfo.m_OutputTensorInfos[0];

    ValidateTensorNumDimensions(inputTensorInfo,  descriptorName, 4, "input");
    ValidateTensorNumDimensions(outputTensorInfo, descriptorName, 4, "output");

    if (m_Parameters.m_BlockShape.size() != 2)
    {
        throw InvalidArgumentException(descriptorName + ": Block Shape must contain 2 spatial dimensions.");
    }

    if (m_Parameters.m_BlockShape.size() != m_Parameters.m_PadList.size())
    {
        throw InvalidArgumentException(descriptorName + ": Pad List must contain the same number of "
                                       "dimensions as Block Shape.");
    }

    const TensorShape& inputShape = inputTensorInfo.GetShape();

    std::pair<unsigned int, unsigned int> heightPad = m_Parameters.m_PadList[0];
    std::pair<unsigned int, unsigned int> widthPad  = m_Parameters.m_PadList[1];

    DataLayoutIndexed dimensionIndices(m_Parameters.m_DataLayout);

    const unsigned int inputWidth  = inputShape[dimensionIndices.GetWidthIndex()] +
                                     widthPad.first + widthPad.second;
    const unsigned int inputHeight = inputShape[dimensionIndices.GetHeightIndex()] +
                                     heightPad.first + heightPad.second;

    const unsigned int numInputElements  = inputShape[0] * inputHeight * inputWidth *
                                           inputShape[dimensionIndices.GetChannelsIndex()];
    const unsigned int numOutputElements = outputTensorInfo.GetNumElements();

    if (numOutputElements != numInputElements)
    {
        throw InvalidArgumentException(descriptorName + ": Input tensor has " +
            to_string(numInputElements) + " after padding but output tensor has " +
            to_string(numOutputElements) + " elements.");
    }

    if (inputHeight % m_Parameters.m_BlockShape[0] != 0 || inputWidth % m_Parameters.m_BlockShape[1] != 0)
    {
        throw InvalidArgumentException(descriptorName + ": Input shape after padding must be "
                                       "divisible by Block Shape in all spatial dimensions");
    }

    std::vector<DataType> supportedTypes =
    {
        DataType::BFloat16,
        DataType::Float16,
        DataType::Float32,
        DataType::QAsymmS8,
        DataType::QAsymmU8,
        DataType::QSymmS16
    };

    ValidateDataTypes(inputTensorInfo, supportedTypes, descriptorName);
    ValidateTensorDataTypesMatch(inputTensorInfo, outputTensorInfo, descriptorName, "input", "output");
}

void SpaceToDepthQueueDescriptor::Validate(const WorkloadInfo& workloadInfo) const
{
    const std::string descriptorName{"SpaceToDepthQueueDescriptor"};

    ValidateNumInputs(workloadInfo,  descriptorName, 1);
    ValidateNumOutputs(workloadInfo, descriptorName, 1);

    const TensorInfo& inputTensorInfo  = workloadInfo.m_InputTensorInfos[0];
    const TensorInfo& outputTensorInfo = workloadInfo.m_OutputTensorInfos[0];

    ValidateTensorNumDimensions(inputTensorInfo,  descriptorName, 4, "input");
    ValidateTensorNumDimensions(outputTensorInfo, descriptorName, 4, "output");

    std::vector<DataType> supportedTypes =
    {
        DataType::BFloat16,
        DataType::Float32,
        DataType::Float16,
        DataType::QAsymmS8,
        DataType::QAsymmU8,
        DataType::QSymmS16
    };

    ValidateDataTypes(inputTensorInfo,  supportedTypes, descriptorName);
    ValidateDataTypes(outputTensorInfo, supportedTypes, descriptorName);

    ValidateTensorNumElementsMatch(inputTensorInfo, outputTensorInfo, descriptorName, "input", "output");

    if (m_Parameters.m_BlockSize == 0)
    {
        throw InvalidArgumentException(descriptorName + ": Block size cannot be 0.");
    }

    DataLayoutIndexed dimensionIndices(m_Parameters.m_DataLayout);
    const unsigned int wIndex = dimensionIndices.GetWidthIndex();
    const unsigned int hIndex = dimensionIndices.GetHeightIndex();
    const unsigned int cIndex = dimensionIndices.GetChannelsIndex();

    const TensorShape& inputShape = inputTensorInfo.GetShape();
    if (inputShape[hIndex] % m_Parameters.m_BlockSize != 0 || inputShape[wIndex]  % m_Parameters.m_BlockSize != 0)
    {
        throw InvalidArgumentException(descriptorName + ": Input shape must be divisible "
                                       "by block size in all spatial dimensions");
    }

    const TensorShape& outputShape = outputTensorInfo.GetShape();
    if (outputShape[cIndex] % (m_Parameters.m_BlockSize * m_Parameters.m_BlockSize) != 0)
    {
        throw InvalidArgumentException(descriptorName + ": The depth of the output tensor"
                                       "must be divisible by the square of block size." );
    }
}

void FloorQueueDescriptor::Validate(const WorkloadInfo& workloadInfo) const
{
    const std::string descriptorName{"FloorQueueDescriptor"};

    ValidateNumInputs(workloadInfo,  descriptorName, 1);
    ValidateNumOutputs(workloadInfo, descriptorName, 1);

    const TensorInfo& inputTensorInfo  = workloadInfo.m_InputTensorInfos[0];
    const TensorInfo& outputTensorInfo = workloadInfo.m_OutputTensorInfos[0];

    std::vector<DataType> supportedTypes =
    {
        DataType::BFloat16,
        DataType::Float32,
        DataType::Float16,
        DataType::QSymmS16
    };

    ValidateDataTypes(inputTensorInfo,  supportedTypes, descriptorName);
    ValidateTensorDataTypesMatch(inputTensorInfo, outputTensorInfo, descriptorName, "input", "output");
    ValidateTensorShapesMatch(inputTensorInfo, outputTensorInfo, descriptorName, "input", "output");
    ValidateTensorQuantizationSpace(inputTensorInfo, outputTensorInfo, descriptorName, "input", "output");
}

void LstmQueueDescriptor::Validate(const WorkloadInfo& workloadInfo) const
{
    // ported from android/ml/nn/common/operations/LSTM.cpp CheckInputTensorDimensions()

    const std::string descriptorName{"LstmQueueDescriptor"};

    // check dimensions of all inputs and outputs
    if (workloadInfo.m_InputTensorInfos.size() != 3)
    {
        throw InvalidArgumentException(descriptorName + ": Invalid number of inputs.");
    }
    if (workloadInfo.m_OutputTensorInfos.size() != 4)
    {
        throw InvalidArgumentException(descriptorName + ": Invalid number of outputs.");
    }

    std::vector<DataType> supportedTypes =
    {
        DataType::BFloat16,
        DataType::Float16,
        DataType::Float32,
        DataType::QSymmS16
    };

    // check for supported type of one input and match them with all the other input and output
    ValidateDataTypes(workloadInfo.m_InputTensorInfos[0], supportedTypes, descriptorName);

    // type matches all other inputs
    for (uint32_t i = 1u; i < workloadInfo.m_InputTensorInfos.size(); ++i)
    {
        ValidateTensorDataTypesMatch(workloadInfo.m_InputTensorInfos[0],
                                     workloadInfo.m_InputTensorInfos[i],
                                     descriptorName,
                                     "input_0",
                                     "input_" + std::to_string(i));
    }
    // type matches all other outputs
    for (uint32_t i = 0u; i < workloadInfo.m_OutputTensorInfos.size(); ++i)
    {
        ValidateTensorDataTypesMatch(workloadInfo.m_InputTensorInfos[0],
                                     workloadInfo.m_OutputTensorInfos[i],
                                     "LstmQueueDescriptor",
                                     "input_0",
                                     "output_" + std::to_string(i));
    }

    // Making sure clipping parameters have valid values.
    // == 0 means no clipping
    //  > 0 means clipping
    if (m_Parameters.m_ClippingThresCell < 0.0f)
    {
        throw InvalidArgumentException(descriptorName + ": negative cell clipping threshold is invalid");
    }
    if (m_Parameters.m_ClippingThresProj < 0.0f)
    {
        throw InvalidArgumentException(descriptorName + ": negative projection clipping threshold is invalid");
    }

    // Inferring batch size, number of outputs and number of cells from the inputs.
    const uint32_t n_input = workloadInfo.m_InputTensorInfos[0].GetShape()[1];
    const uint32_t n_batch = workloadInfo.m_InputTensorInfos[0].GetShape()[0];
    ValidatePointer(m_InputToOutputWeights, "Null pointer check", "InputToOutputWeights");
    const uint32_t n_cell = m_InputToOutputWeights->GetShape()[0];
    ValidatePointer(m_RecurrentToOutputWeights, "Null pointer check", "RecurrentToOutputWeights");
    const uint32_t n_output = m_RecurrentToOutputWeights->GetShape()[1];

    // input tensor
    ValidateTensorNumDimNumElem(workloadInfo.m_InputTensorInfos[0], 2, (n_batch * n_input),
                                descriptorName + " input_0");
    // outputStateInTensor
    ValidateTensorNumDimNumElem(workloadInfo.m_InputTensorInfos[1], 2, (n_batch * n_output),
                                descriptorName + " input_1");
    // outputStateInTensor
    ValidateTensorNumDimNumElem(workloadInfo.m_InputTensorInfos[2], 2, (n_batch * n_cell),
                                descriptorName + " input_2");
    // scratchBufferTensor
    unsigned int scratchBufferSize = m_Parameters.m_CifgEnabled ? n_cell * 3 : n_cell * 4;
    ValidateTensorNumDimNumElem(workloadInfo.m_OutputTensorInfos[0], 2, (n_batch * scratchBufferSize),
                                descriptorName + " output_0");
    // outputStateOutTensor
    ValidateTensorNumDimNumElem(workloadInfo.m_OutputTensorInfos[1], 2, (n_batch * n_output),
                                descriptorName + " output_1");
    // cellStateOutTensor
    ValidateTensorNumDimNumElem(workloadInfo.m_OutputTensorInfos[2], 2, (n_batch * n_cell),
                                descriptorName + " output_2");
    // outputTensor
    ValidateTensorNumDimNumElem(workloadInfo.m_OutputTensorInfos[3], 2, (n_batch * n_output),
                                descriptorName + " output_3");

    // check that dimensions of inputs/outputs and QueueDescriptor data match with each other
    if ( m_InputToInputWeights )
    {
        ValidateTensorNumDimNumElem(m_InputToInputWeights->GetTensorInfo(), 2,
                                      (n_cell * n_input), "InputLayerNormWeights");
    }

    ValidatePointer(m_InputToForgetWeights, "Null pointer check", "InputToForgetWeights");
    ValidateTensorNumDimNumElem(m_InputToForgetWeights->GetTensorInfo(), 2,
                                  (n_cell * n_input), "InputToForgetWeights");

    ValidatePointer(m_InputToCellWeights, "Null pointer check", "InputToCellWeights");
    ValidateTensorNumDimNumElem(m_InputToCellWeights->GetTensorInfo(), 2,
                                  (n_cell * n_input), "InputToCellWeights");

    if ( m_RecurrentToInputWeights )
    {
        ValidateTensorNumDimNumElem(m_RecurrentToInputWeights->GetTensorInfo(), 2,
                                      (n_cell * n_output), "RecurrentToInputWeights");
    }

    ValidatePointer(m_RecurrentToForgetWeights, "Null pointer check", "RecurrentToForgetWeights");
    ValidateTensorNumDimNumElem(m_RecurrentToForgetWeights->GetTensorInfo(), 2,
                                  (n_cell * n_output), "RecurrentToForgetWeights");

    ValidatePointer(m_RecurrentToCellWeights, "Null pointer check", "RecurrentToCellWeights");
    ValidateTensorNumDimNumElem(m_RecurrentToCellWeights->GetTensorInfo(), 2,
                                  (n_cell * n_output), "RecurrentToCellWeights");

    // Make sure the input-gate's parameters are either both present (regular
    // LSTM) or not at all (CIFG-LSTM). And CifgEnable is set accordingly.
    bool cifg_weights_all_or_none = ((m_InputToInputWeights && m_RecurrentToInputWeights &&
                                     !m_Parameters.m_CifgEnabled) ||
                                     (!m_InputToInputWeights && !m_RecurrentToInputWeights &&
                                     m_Parameters.m_CifgEnabled));
    if (!cifg_weights_all_or_none)
    {
        throw InvalidArgumentException(descriptorName + ": Input-Gate's parameters InputToInputWeights and "
                                       "RecurrentToInputWeights must either both be present (regular LSTM) "
                                       "or both not present (CIFG-LSTM). In addition CifgEnable must be set "
                                       "accordingly.");
    }

    if ( m_CellToInputWeights )
    {
        ValidateTensorNumDimNumElem(m_CellToInputWeights->GetTensorInfo(), 1,
                                      n_cell, "CellToInputWeights");
    }
    if ( m_CellToForgetWeights )
    {
        ValidateTensorNumDimNumElem(m_CellToForgetWeights->GetTensorInfo(), 1,
                                      n_cell, "CellToForgetWeights");
    }
    if ( m_CellToOutputWeights )
    {
        ValidateTensorNumDimNumElem(m_CellToOutputWeights->GetTensorInfo(), 1,
                                      n_cell, "CellToOutputWeights");
    }

    // Making sure the peephole weights are there all or none. And PeepholeEnable is set accordingly.
    bool peephole_weights_all_or_none =
            (((m_CellToInputWeights || m_Parameters.m_CifgEnabled) &&  m_CellToForgetWeights
            && m_CellToOutputWeights && m_Parameters.m_PeepholeEnabled)
            || ( !m_CellToInputWeights && !m_CellToForgetWeights
            && !m_CellToOutputWeights && !m_Parameters.m_PeepholeEnabled));
    if (!peephole_weights_all_or_none)
    {
        throw InvalidArgumentException(descriptorName + ": Invalid combination of peephole parameters.");
    }

    // Make sure the input gate bias is present only when not a CIFG-LSTM.
    if (m_Parameters.m_CifgEnabled)
    {
        if (m_InputGateBias)
        {
            throw InvalidArgumentException(descriptorName + ": InputGateBias is present and CIFG-LSTM is enabled.");
        }
    }
    else
    {
        if (!m_InputGateBias)
        {
            throw InvalidArgumentException(descriptorName + ": If CIFG-LSTM is disabled InputGateBias "
                                           "must be present.");
        }
        ValidateTensorNumDimNumElem(m_InputGateBias->GetTensorInfo(), 1,
                                      n_cell, "InputGateBias");
    }

    ValidatePointer(m_ForgetGateBias, "Null pointer check", "ForgetGateBias");
    ValidateTensorNumDimNumElem(m_ForgetGateBias->GetTensorInfo(), 1, n_cell, "ForgetGateBias");

    ValidatePointer(m_CellBias, "Null pointer check", "CellBias");
    ValidateTensorNumDimNumElem(m_CellBias->GetTensorInfo(), 1, n_cell, "CellBias");

    ValidatePointer(m_OutputGateBias, "Null pointer check", "OutputGateBias");
    ValidateTensorNumDimNumElem(m_OutputGateBias->GetTensorInfo(), 1, n_cell, "OutputGateBias");

    if (m_ProjectionWeights)
    {
        ValidateTensorNumDimNumElem(m_ProjectionWeights->GetTensorInfo(), 2,
                                      (n_cell * n_output), "ProjectionWeights");
    }
    if (m_ProjectionBias)
    {
        ValidateTensorNumDimNumElem(m_ProjectionBias->GetTensorInfo(), 1, n_output, "ProjectionBias");
    }

    // Making sure the projection tensors are consistent:
    // 1) If projection weight is not present, then projection bias should not be
    // present.
    // 2) If projection weight is present, then projection bias is optional.
    bool projecton_tensors_consistent = ((!m_ProjectionWeights && !m_ProjectionBias &&
                                        !m_Parameters.m_ProjectionEnabled)
                                        || (m_ProjectionWeights && !m_ProjectionBias &&
                                        m_Parameters.m_ProjectionEnabled)
                                        || (m_ProjectionWeights && m_ProjectionBias &&
                                        m_Parameters.m_ProjectionEnabled));
    if (!projecton_tensors_consistent)
    {
        throw InvalidArgumentException(descriptorName + ": Projection tensors are inconsistent.");
    }

    // The four layer normalization weights either all have values or none of them have values. Additionally, if
    // CIFG is used, input layer normalization weights tensor is omitted and the other layer normalization weights
    // either all have values or none of them have values. Layer normalization is used when the values of all the
    // layer normalization weights are present
    if (m_InputLayerNormWeights)
    {
        ValidateTensorNumDimNumElem(m_InputLayerNormWeights->GetTensorInfo(), 1, n_cell, "InputLayerNormWeights");
    }
    if (m_ForgetLayerNormWeights)
    {
        ValidateTensorNumDimNumElem(m_ForgetLayerNormWeights->GetTensorInfo(), 1, n_cell, "ForgetLayerNormWeights");
    }
    if (m_CellLayerNormWeights)
    {
        ValidateTensorNumDimNumElem(m_CellLayerNormWeights->GetTensorInfo(), 1, n_cell, "CellLayerNormWeights");
    }
    if (m_OutputLayerNormWeights)
    {
        ValidateTensorNumDimNumElem(m_OutputLayerNormWeights->GetTensorInfo(), 1, n_cell, "OutputLayerNormWeights");
    }

    if (m_Parameters.m_LayerNormEnabled)
    {
        if (!m_Parameters.m_CifgEnabled)
        {
            if (!m_InputLayerNormWeights)
            {
                throw InvalidArgumentException(descriptorName + ": Layer normalisation is enabled and CIFG-LSTM is "
                                               "disabled but InputLayerNormWeights are not present");
            }
            ValidateTensorNumDimNumElem(m_InputLayerNormWeights->GetTensorInfo(),
                                          1, n_cell, "InputLayerNormWeights");
        }
        else if (m_InputLayerNormWeights)
        {
            throw InvalidArgumentException(descriptorName + ":InputLayerNormWeights are present while CIFG is "
                                           "enabled");
        }

        ValidatePointer(m_ForgetLayerNormWeights, "Null pointer check layer normalisation enabled",
                        "ForgetLayerNormWeights");
        ValidateTensorNumDimNumElem(m_ForgetLayerNormWeights->GetTensorInfo(), 1, n_cell, "ForgetLayerNormWeights");

        ValidatePointer(m_OutputLayerNormWeights, "Null pointer check layer normalisation enabled",
                        "OutputLayerNormWeights");
        ValidateTensorNumDimNumElem(m_OutputLayerNormWeights->GetTensorInfo(), 1, n_cell, "OutputLayerNormWeights");

        ValidatePointer(m_CellLayerNormWeights, "Null pointer check layer normalisation enabled",
                        "CellLayerNormWeights");
        ValidateTensorNumDimNumElem(m_CellLayerNormWeights->GetTensorInfo(), 1, n_cell, "CellLayerNormWeights");
    }
    else if (m_InputLayerNormWeights || m_ForgetLayerNormWeights || m_OutputLayerNormWeights || m_CellLayerNormWeights)
    {
        throw InvalidArgumentException(descriptorName + ": Layer normalisation is disabled but one or more layer "
                                       "normalisation weights are present.");
    }
}

void ConvertBf16ToFp32QueueDescriptor::Validate(const WorkloadInfo& workloadInfo) const
{
    const std::string descriptorName{"ConvertBf16ToFp32QueueDescriptor"};

    ValidateNumInputs(workloadInfo,  descriptorName, 1);
    ValidateNumOutputs(workloadInfo, descriptorName, 1);

    const TensorInfo& inputTensorInfo  = workloadInfo.m_InputTensorInfos[0];
    const TensorInfo& outputTensorInfo = workloadInfo.m_OutputTensorInfos[0];

    if (inputTensorInfo.GetDataType() != DataType::BFloat16)
    {
        throw InvalidArgumentException(descriptorName + ": Input tensor type must be BFloat16.");
    }

    if (outputTensorInfo.GetDataType() != DataType::Float32)
    {
        throw InvalidArgumentException(descriptorName + ": Output tensor type must be Float32.");
    }

    ValidateTensorShapesMatch(inputTensorInfo, outputTensorInfo, descriptorName, "input", "output");
}

void ConvertFp32ToBf16QueueDescriptor::Validate(const WorkloadInfo& workloadInfo) const
{
    const std::string descriptorName{"ConvertFp32ToBf16QueueDescriptor"};

    ValidateNumInputs(workloadInfo,  descriptorName, 1);
    ValidateNumOutputs(workloadInfo, descriptorName, 1);

    const TensorInfo& inputTensorInfo  = workloadInfo.m_InputTensorInfos[0];
    const TensorInfo& outputTensorInfo = workloadInfo.m_OutputTensorInfos[0];

    if (inputTensorInfo.GetDataType() != DataType::Float32)
    {
        throw InvalidArgumentException(descriptorName + ": Input tensor type must be Float32.");
    }

    if (outputTensorInfo.GetDataType() != DataType::BFloat16)
    {
        throw InvalidArgumentException(descriptorName + ": Output tensor type must be BFloat16.");
    }

    ValidateTensorShapesMatch(inputTensorInfo, outputTensorInfo, descriptorName, "input", "output");
}

void ConvertFp32ToFp16QueueDescriptor::Validate(const WorkloadInfo& workloadInfo) const
{
    const std::string descriptorName{"ConvertFp32ToFp16QueueDescriptor"};

    ValidateNumInputs(workloadInfo,  descriptorName, 1);
    ValidateNumOutputs(workloadInfo, descriptorName, 1);

    const TensorInfo& inputTensorInfo  = workloadInfo.m_InputTensorInfos[0];
    const TensorInfo& outputTensorInfo = workloadInfo.m_OutputTensorInfos[0];

    if (inputTensorInfo.GetDataType() != DataType::Float32)
    {
        throw InvalidArgumentException(descriptorName + ": Input tensor type must be Float32.");
    }

    if (outputTensorInfo.GetDataType() != DataType::Float16)
    {
        throw InvalidArgumentException(descriptorName + ": Output tensor type must be Float16.");
    }

    ValidateTensorShapesMatch(inputTensorInfo, outputTensorInfo, descriptorName, "input", "output");
}

void ConvertFp16ToFp32QueueDescriptor::Validate(const WorkloadInfo& workloadInfo) const
{
    const std::string descriptorName{"ConvertFp16ToFp32QueueDescriptor"};

    ValidateNumInputs(workloadInfo,  descriptorName, 1);
    ValidateNumOutputs(workloadInfo, descriptorName, 1);

    const TensorInfo& inputTensorInfo  = workloadInfo.m_InputTensorInfos[0];
    const TensorInfo& outputTensorInfo = workloadInfo.m_OutputTensorInfos[0];

    if (inputTensorInfo.GetDataType() != DataType::Float16)
    {
        throw InvalidArgumentException(descriptorName + ": Input tensor type must be Float16.");
    }

    if (outputTensorInfo.GetDataType() != DataType::Float32)
    {
        throw InvalidArgumentException(descriptorName + ": Output tensor type must be Float32.");
    }

    ValidateTensorShapesMatch(inputTensorInfo, outputTensorInfo, descriptorName, "input", "output");
}

void DivisionQueueDescriptor::Validate(const WorkloadInfo& workloadInfo) const
{
    const std::string descriptorName{"DivisionQueueDescriptor"};

    ValidateNumInputs(workloadInfo,  descriptorName, 2);
    ValidateNumOutputs(workloadInfo, descriptorName, 1);

    const TensorInfo& inputTensorInfo0 = workloadInfo.m_InputTensorInfos[0];
    const TensorInfo& inputTensorInfo1 = workloadInfo.m_InputTensorInfos[1];
    const TensorInfo& outputTensorInfo = workloadInfo.m_OutputTensorInfos[0];

    std::vector<DataType> supportedTypes =
    {
        DataType::BFloat16,
        DataType::Float16,
        DataType::Float32,
        DataType::QAsymmS8,
        DataType::QAsymmU8,
        DataType::QSymmS16,
        DataType::Signed32
    };

    ValidateDataTypes(inputTensorInfo0, supportedTypes, descriptorName);
    ValidateDataTypes(inputTensorInfo1, supportedTypes, descriptorName);
    ValidateDataTypes(outputTensorInfo, supportedTypes, descriptorName);

    ValidateBroadcastTensorShapesMatch(inputTensorInfo0,
                                       inputTensorInfo1,
                                       outputTensorInfo,
                                       descriptorName,
                                       "input_0",
                                       "input_1");
}

void SubtractionQueueDescriptor::Validate(const WorkloadInfo& workloadInfo) const
{
    const std::string descriptorName{"SubtractionQueueDescriptor"};

    ValidateNumInputs(workloadInfo,  descriptorName, 2);
    ValidateNumOutputs(workloadInfo, descriptorName, 1);

    const TensorInfo& inputTensorInfo0 = workloadInfo.m_InputTensorInfos[0];
    const TensorInfo& inputTensorInfo1 = workloadInfo.m_InputTensorInfos[1];
    const TensorInfo& outputTensorInfo = workloadInfo.m_OutputTensorInfos[0];

    std::vector<DataType> supportedTypes =
    {
        DataType::BFloat16,
        DataType::Float16,
        DataType::Float32,
        DataType::QAsymmS8,
        DataType::QAsymmU8,
        DataType::QSymmS16,
        DataType::Signed32,
    };

    ValidateDataTypes(inputTensorInfo0, supportedTypes, descriptorName);
    ValidateDataTypes(inputTensorInfo1, supportedTypes, descriptorName);
    ValidateDataTypes(outputTensorInfo, supportedTypes, descriptorName);

    ValidateBroadcastTensorShapesMatch(inputTensorInfo0,
                                       inputTensorInfo1,
                                       outputTensorInfo,
                                       descriptorName,
                                       "input_0",
                                       "input_1");
}

void MaximumQueueDescriptor::Validate(const WorkloadInfo& workloadInfo) const
{
    const std::string descriptorName{"MaximumQueueDescriptor"};

    ValidateNumInputs(workloadInfo,  descriptorName, 2);
    ValidateNumOutputs(workloadInfo, descriptorName, 1);

    const TensorInfo& inputTensorInfo0 = workloadInfo.m_InputTensorInfos[0];
    const TensorInfo& inputTensorInfo1 = workloadInfo.m_InputTensorInfos[1];
    const TensorInfo& outputTensorInfo = workloadInfo.m_OutputTensorInfos[0];

    std::vector<DataType> supportedTypes =
    {
        DataType::BFloat16,
        DataType::Float16,
        DataType::Float32,
        DataType::QAsymmS8,
        DataType::QAsymmU8,
        DataType::QSymmS16,
        DataType::Signed32
    };

    ValidateDataTypes(inputTensorInfo0, supportedTypes, descriptorName);
    ValidateDataTypes(inputTensorInfo1, supportedTypes, descriptorName);
    ValidateDataTypes(outputTensorInfo, supportedTypes, descriptorName);

    ValidateBroadcastTensorShapesMatch(inputTensorInfo0,
                                       inputTensorInfo1,
                                       outputTensorInfo,
                                       descriptorName,
                                       "input_0",
                                       "input_1");
}

void MeanQueueDescriptor::Validate(const WorkloadInfo& workloadInfo) const
{
    const std::string descriptorName{"MeanQueueDescriptor"};

    ValidateNumInputs(workloadInfo,  descriptorName, 1);
    ValidateNumOutputs(workloadInfo, descriptorName, 1);

    const TensorInfo& inputTensorInfo  = workloadInfo.m_InputTensorInfos[0];
    const TensorInfo& outputTensorInfo = workloadInfo.m_OutputTensorInfos[0];

    std::vector<DataType> supportedTypes =
    {
        DataType::BFloat16,
        DataType::Float32,
        DataType::Float16,
        DataType::QAsymmS8,
        DataType::QAsymmU8,
        DataType::QSymmS16
    };

    // First check if input tensor data type is supported, then
    // check if this data type matches the output tensor data type
    ValidateDataTypes(inputTensorInfo,  supportedTypes, descriptorName);
    ValidateTensorDataTypesMatch(inputTensorInfo, outputTensorInfo, descriptorName, "input", "output");

    if (m_Parameters.m_KeepDims)
    {
        ValidateTensorNumDimensions(outputTensorInfo, descriptorName, inputTensorInfo.GetNumDimensions(), "output");
    }
    else if (m_Parameters.m_Axis.empty())
    {
        ValidateTensorNumDimensions(outputTensorInfo, descriptorName, 1, "output");
    }
    else
    {
        unsigned int outputDim =
            inputTensorInfo.GetNumDimensions() - armnn::numeric_cast<unsigned int>(m_Parameters.m_Axis.size());
        ValidateTensorNumDimensions(outputTensorInfo,
                                    descriptorName,
                                    outputDim > 0 ? outputDim : 1,
                                    "output");
    }
}

void PadQueueDescriptor::Validate(const WorkloadInfo& workloadInfo) const
{
    const std::string descriptorName{"PadQueueDescriptor"};

    ValidateNumInputs(workloadInfo,  descriptorName, 1);
    ValidateNumOutputs(workloadInfo, descriptorName, 1);

    const TensorInfo& inputTensorInfo  = workloadInfo.m_InputTensorInfos[0];
    const TensorInfo& outputTensorInfo = workloadInfo.m_OutputTensorInfos[0];

    // input and output should have the same number of dimensions
    ValidateTensorNumDimensions(outputTensorInfo, descriptorName, inputTensorInfo.GetNumDimensions(), "output");

    // there should be entry in the pad list for each dimension in the input tensor
    if (m_Parameters.m_PadList.size() != inputTensorInfo.GetNumDimensions()) {
        throw InvalidArgumentException(descriptorName + ":Pad List should contain the same number of entries "
                                       "as there are dimensions in the input tensor that is " +
                                       std::to_string(inputTensorInfo.GetNumDimensions()) + " entries " +
                                       " not " + std::to_string(m_Parameters.m_PadList.size()) + " entries.");
    }
}

void QuantizeQueueDescriptor::Validate(const WorkloadInfo& workloadInfo) const
{
    const std::string descriptorName{"QuantizeQueueDescriptor"};

    ValidateNumInputs(workloadInfo,  descriptorName, 1);
    ValidateNumOutputs(workloadInfo, descriptorName, 1);

    const TensorInfo& inputTensorInfo  = workloadInfo.m_InputTensorInfos[0];
    const TensorInfo& outputTensorInfo = workloadInfo.m_OutputTensorInfos[0];

    std::vector<DataType> supportedTypes =
    {
        DataType::BFloat16,
        DataType::Float32,
        DataType::Float16,
        DataType::QSymmS8,
        DataType::QAsymmS8,
        DataType::QAsymmU8,
        DataType::QSymmS16
    };

    ValidateDataTypes(inputTensorInfo, supportedTypes, descriptorName);

    if (!IsQuantizedType(outputTensorInfo.GetDataType()))
    {
        throw InvalidArgumentException(descriptorName + ": Output of quantized layer must be quantized type.");
    }
}

void BatchToSpaceNdQueueDescriptor::Validate(const WorkloadInfo& workloadInfo) const
{
    const std::string descriptorName{"BatchToSpaceNdQueueDescriptor"};

    ValidateNumInputs(workloadInfo,  descriptorName, 1);
    ValidateNumOutputs(workloadInfo, descriptorName, 1);

    const TensorInfo& inputTensorInfo  = workloadInfo.m_InputTensorInfos[0];
    const TensorInfo& outputTensorInfo = workloadInfo.m_OutputTensorInfos[0];

    std::vector<DataType> supportedTypes =
    {
        DataType::BFloat16,
        DataType::Float32,
        DataType::Float16,
        DataType::QAsymmS8,
        DataType::QAsymmU8,
        DataType::QSymmS16
    };

    ValidateDataTypes(inputTensorInfo, supportedTypes, descriptorName);
    ValidateTensorDataTypesMatch(inputTensorInfo, outputTensorInfo, descriptorName, "input", "output");
}

void StridedSliceQueueDescriptor::Validate(const WorkloadInfo& workloadInfo) const
{
    const std::string descriptorName{"StridedSliceQueueDescriptor"};

    ValidateNumInputs(workloadInfo,  descriptorName, 1);
    ValidateNumOutputs(workloadInfo, descriptorName, 1);

    const TensorInfo& inputTensorInfo  = workloadInfo.m_InputTensorInfos[0];
    const TensorInfo& outputTensorInfo = workloadInfo.m_OutputTensorInfos[0];

    std::vector<DataType> supportedTypes =
    {
        DataType::BFloat16,
        DataType::Float16,
        DataType::Float32,
        DataType::QAsymmS8,
        DataType::QAsymmU8,
        DataType::QSymmS16
    };

    ValidateDataTypes(inputTensorInfo, supportedTypes, descriptorName);
    ValidateTensorDataTypesMatch(inputTensorInfo, outputTensorInfo, descriptorName, "input", "output");

    ValidateTensorQuantizationSpace(inputTensorInfo, outputTensorInfo, descriptorName, "input", "output");

    const uint32_t rank = inputTensorInfo.GetNumDimensions();
    if (rank > 4)
    {
        throw InvalidArgumentException(descriptorName + ": Input tensors with rank greater than 4 are not supported.");
    }

    // Begin, End & Stride length must be of rank(input0)
    if (m_Parameters.m_Begin.size() != rank)
    {
        throw InvalidArgumentException(descriptorName + ": Begin length must be of rank " + std::to_string(rank));
    }

    if (m_Parameters.m_End.size() != rank)
    {
        throw InvalidArgumentException(descriptorName + ": End length must be of rank " + std::to_string(rank));
    }

    if (m_Parameters.m_Stride.size() != rank)
    {
        throw InvalidArgumentException(descriptorName + ": Stride length must be of rank " + std::to_string(rank));
    }

    // Stride entries must be non-zero
    for (auto& stride : m_Parameters.m_Stride)
    {
        if (stride == 0)
        {
            throw InvalidArgumentException(descriptorName + ": Stride entries must be non-zero.");
        }
    }
}

void MinimumQueueDescriptor::Validate(const WorkloadInfo& workloadInfo) const
{
    const std::string descriptorName{"MinimumQueueDescriptor"};

    ValidateNumInputs(workloadInfo,  descriptorName, 2);
    ValidateNumOutputs(workloadInfo, descriptorName, 1);

    const TensorInfo& inputTensorInfo0 = workloadInfo.m_InputTensorInfos[0];
    const TensorInfo& inputTensorInfo1 = workloadInfo.m_InputTensorInfos[1];
    const TensorInfo& outputTensorInfo = workloadInfo.m_OutputTensorInfos[0];

    std::vector<DataType> supportedTypes =
    {
        DataType::BFloat16,
        DataType::Float16,
        DataType::Float32,
        DataType::QAsymmS8,
        DataType::QAsymmU8,
        DataType::QSymmS16,
        DataType::Signed32
    };

    ValidateDataTypes(inputTensorInfo0, supportedTypes, descriptorName);
    ValidateDataTypes(inputTensorInfo1, supportedTypes, descriptorName);
    ValidateDataTypes(outputTensorInfo, supportedTypes, descriptorName);

    ValidateBroadcastTensorShapesMatch(inputTensorInfo0,
                                       inputTensorInfo1,
                                       outputTensorInfo,
                                       descriptorName,
                                       "input_0",
                                       "input_1");
}

void DebugQueueDescriptor::Validate(const WorkloadInfo& workloadInfo) const
{
    const std::string descriptorName{"DebugQueueDescriptor"};

    ValidateNumInputs(workloadInfo,  descriptorName, 1);
    ValidateNumOutputs(workloadInfo, descriptorName, 1);
}

void EqualQueueDescriptor::Validate(const WorkloadInfo& workloadInfo) const
{
    const std::string descriptorName{"EqualQueueDescriptor"};

    ValidateNumInputs(workloadInfo,  descriptorName, 2);
    ValidateNumOutputs(workloadInfo, descriptorName, 1);

    const TensorInfo& inputTensorInfo0 = workloadInfo.m_InputTensorInfos[0];
    const TensorInfo& inputTensorInfo1 = workloadInfo.m_InputTensorInfos[1];
    const TensorInfo& outputTensorInfo = workloadInfo.m_OutputTensorInfos[0];

    ValidateBroadcastTensorShapesMatch(inputTensorInfo0,
                                       inputTensorInfo1,
                                       outputTensorInfo,
                                       descriptorName,
                                       "input_0",
                                       "input_1");

    if (outputTensorInfo.GetDataType() != DataType::Boolean)
    {
        throw InvalidArgumentException(descriptorName + ": Output tensor type must be Boolean.");
    }
}

void GreaterQueueDescriptor::Validate(const WorkloadInfo& workloadInfo) const
{
    const std::string descriptorName{"GreaterQueueDescriptor"};

    ValidateNumInputs(workloadInfo,  descriptorName, 2);
    ValidateNumOutputs(workloadInfo, descriptorName, 1);

    const TensorInfo& inputTensorInfo0 = workloadInfo.m_InputTensorInfos[0];
    const TensorInfo& inputTensorInfo1 = workloadInfo.m_InputTensorInfos[1];
    const TensorInfo& outputTensorInfo = workloadInfo.m_OutputTensorInfos[0];

    ValidateBroadcastTensorShapesMatch(inputTensorInfo0,
                                       inputTensorInfo1,
                                       outputTensorInfo,
                                       descriptorName,
                                       "input_0",
                                       "input_1");

    if (outputTensorInfo.GetDataType() != DataType::Boolean)
    {
        throw InvalidArgumentException(descriptorName + ": Output tensor type must be Boolean.");
    }
}

void RsqrtQueueDescriptor::Validate(const WorkloadInfo& workloadInfo) const
{
    const std::string descriptorName{"RsqrtQueueDescriptor"};

    ValidateNumInputs(workloadInfo,  descriptorName, 1);
    ValidateNumOutputs(workloadInfo, descriptorName, 1);

    const TensorInfo& inputTensorInfo  = workloadInfo.m_InputTensorInfos[0];
    const TensorInfo& outputTensorInfo = workloadInfo.m_OutputTensorInfos[0];

    ValidateTensorShapesMatch(inputTensorInfo, outputTensorInfo, descriptorName, "input", "output");

    std::vector<DataType> supportedTypes =
    {
        DataType::BFloat16,
        DataType::Float16,
        DataType::Float32,
        DataType::QAsymmS8,
        DataType::QAsymmU8,
        DataType::QSymmS16
    };

    ValidateDataTypes(inputTensorInfo, supportedTypes, descriptorName);
    ValidateTensorDataTypesMatch(inputTensorInfo, outputTensorInfo, descriptorName, "input", "output");
}

void GatherQueueDescriptor::Validate(const WorkloadInfo& workloadInfo) const
{
    const std::string descriptorName{"GatherQueueDescriptor"};

    ValidateNumInputs(workloadInfo,  descriptorName, 2);
    ValidateNumOutputs(workloadInfo, descriptorName, 1);

    const TensorInfo& indicesTensorInfo = workloadInfo.m_InputTensorInfos[1];
    if (indicesTensorInfo.GetDataType() != DataType::Signed32)
    {
        throw InvalidArgumentException(descriptorName + ": Indices tensor type must be Int32.");
    }

    const TensorInfo& inputTensorInfo  = workloadInfo.m_InputTensorInfos[0];
    const TensorInfo& outputTensorInfo = workloadInfo.m_OutputTensorInfos[0];

    std::vector<DataType> supportedTypes =
    {
        DataType::BFloat16,
        DataType::Float16,
        DataType::Float32,
        DataType::QAsymmS8,
        DataType::QAsymmU8,
        DataType::QSymmS16,
        DataType::Signed32,
    };

    ValidateDataTypes(inputTensorInfo, supportedTypes, descriptorName);

    ValidateTensorDataTypesMatch(inputTensorInfo, outputTensorInfo, descriptorName, "input", "output");

    unsigned int outputDim  = inputTensorInfo.GetNumDimensions() + indicesTensorInfo.GetNumDimensions() - 1;
    ValidateTensorNumDimensions(outputTensorInfo, descriptorName, outputDim, "output");
}

void DetectionPostProcessQueueDescriptor::Validate(const WorkloadInfo& workloadInfo) const
{
    const std::string& descriptorName{"DetectionPostProcessQueueDescriptor"};

    ValidateNumInputs(workloadInfo, descriptorName, 2);

    if (workloadInfo.m_OutputTensorInfos.size() != 4)
    {
        throw InvalidArgumentException(descriptorName + ": Requires exactly four outputs. " +
                                       to_string(workloadInfo.m_OutputTensorInfos.size()) + " has been provided.");
    }

    if (m_Anchors == nullptr)
    {
        throw InvalidArgumentException(descriptorName + ": Anchors tensor descriptor is missing.");
    }

    const TensorInfo& boxEncodingsInfo =  workloadInfo.m_InputTensorInfos[0];
    const TensorInfo& scoresInfo       =  workloadInfo.m_InputTensorInfos[1];
    const TensorInfo& anchorsInfo      = m_Anchors->GetTensorInfo();

    const TensorInfo& detectionBoxesInfo   = workloadInfo.m_OutputTensorInfos[0];
    const TensorInfo& detectionClassesInfo = workloadInfo.m_OutputTensorInfos[1];
    const TensorInfo& detectionScoresInfo  = workloadInfo.m_OutputTensorInfos[2];
    const TensorInfo& numDetectionsInfo    = workloadInfo.m_OutputTensorInfos[3];

    ValidateTensorNumDimensions(boxEncodingsInfo, descriptorName, 3, "box encodings");
    ValidateTensorNumDimensions(scoresInfo, descriptorName, 3, "scores");
    ValidateTensorNumDimensions(anchorsInfo, descriptorName, 2, "anchors");

    const std::vector<DataType> supportedInputTypes =
    {
        DataType::BFloat16,
        DataType::Float32,
        DataType::Float16,
        DataType::QAsymmS8,
        DataType::QAsymmU8,
        DataType::QSymmS16
    };

    ValidateDataTypes(boxEncodingsInfo, supportedInputTypes, descriptorName);
    ValidateDataTypes(scoresInfo, supportedInputTypes, descriptorName);
    ValidateDataTypes(anchorsInfo, supportedInputTypes, descriptorName);

    ValidateTensorNumDimensions(detectionBoxesInfo, descriptorName, 3, "detection boxes");
    ValidateTensorNumDimensions(detectionScoresInfo, descriptorName, 2, "detection scores");
    ValidateTensorNumDimensions(detectionClassesInfo, descriptorName, 2, "detection classes");
    ValidateTensorNumDimensions(numDetectionsInfo, descriptorName, 1, "num detections");

    // NOTE: Output is always Float32 regardless of input type
    ValidateTensorDataType(detectionBoxesInfo, DataType::Float32, descriptorName, "detection boxes");
    ValidateTensorDataType(detectionScoresInfo, DataType::Float32, descriptorName, "detection scores");
    ValidateTensorDataType(detectionClassesInfo, DataType::Float32, descriptorName, "detection classes");
    ValidateTensorDataType(numDetectionsInfo, DataType::Float32, descriptorName, "num detections");

    if (m_Parameters.m_NmsIouThreshold <= 0.0f || m_Parameters.m_NmsIouThreshold > 1.0f)
    {
        throw InvalidArgumentException(descriptorName + ": Intersection over union threshold "
                                       "must be positive and less than or equal to 1.");
    }

    if (scoresInfo.GetShape()[2] != m_Parameters.m_NumClasses + 1)
    {
        throw InvalidArgumentException(descriptorName + ": Number of classes with background "
                                       "should be equal to number of classes + 1.");
    }
}

void DequantizeQueueDescriptor::Validate(const WorkloadInfo& workloadInfo) const
{
    const std::string& descriptorName{"DequantizeQueueDescriptor"};

    ValidateNumInputs(workloadInfo,  descriptorName, 1);
    ValidateNumOutputs(workloadInfo, descriptorName, 1);

    const TensorInfo& inputTensorInfo  = workloadInfo.m_InputTensorInfos[0];
    const TensorInfo& outputTensorInfo = workloadInfo.m_OutputTensorInfos[0];

    if (!IsQuantizedType(inputTensorInfo.GetDataType()))
    {
        throw InvalidArgumentException(descriptorName + ": Input to dequantize layer must be quantized type.");
    }

    std::vector<DataType> supportedTypes =
    {
        DataType::BFloat16,
        DataType::Float32,
        DataType::Float16
    };

    ValidateDataTypes(outputTensorInfo, supportedTypes, descriptorName);
}

void MergeQueueDescriptor::Validate(const WorkloadInfo& workloadInfo) const
{
    const std::string& descriptorName{"MergeQueueDescriptor"};

    ValidateNumInputs(workloadInfo,  descriptorName, 2);
    ValidateNumOutputs(workloadInfo, descriptorName, 1);

    const TensorInfo& inputTensorInfo0 = workloadInfo.m_InputTensorInfos[0];
    const TensorInfo& inputTensorInfo1 = workloadInfo.m_InputTensorInfos[1];
    const TensorInfo& outputTensorInfo = workloadInfo.m_OutputTensorInfos[0];

    ValidateTensorShapesMatch(inputTensorInfo0, inputTensorInfo1, descriptorName, "input_0", "input_1");
    ValidateTensorShapesMatch(inputTensorInfo0, outputTensorInfo, descriptorName, "input_0", "output");

    ValidateTensorDataTypesMatch(inputTensorInfo0, inputTensorInfo1, descriptorName, "input_0", "input_1");
    ValidateTensorDataTypesMatch(inputTensorInfo0, outputTensorInfo, descriptorName, "input_0", "output");
}

void ShapeQueueDescriptor::Validate(const WorkloadInfo& workloadInfo) const
{
    const std::string& descriptorName{"ShapeQueueDescriptor"};

    ValidateNumInputs(workloadInfo,  descriptorName, 1);
    ValidateNumOutputs(workloadInfo, descriptorName, 1);

    const TensorInfo& inputTensorInfo = workloadInfo.m_InputTensorInfos[0];
    const TensorInfo& outputTensorInfo = workloadInfo.m_OutputTensorInfos[0];

    std::vector<DataType> supportedTypes =
    {
        DataType::BFloat16,
        DataType::Float16,
        DataType::Float32,
        DataType::QAsymmS8,
        DataType::QAsymmU8,
        DataType::QAsymmS8,
        DataType::QSymmS8,
        DataType::QSymmS16,
        DataType::Signed32
    };

    ValidateDataTypes(inputTensorInfo, supportedTypes, descriptorName);
    ValidateDataTypes(outputTensorInfo, {DataType::Signed32}, descriptorName);
}

void SwitchQueueDescriptor::Validate(const WorkloadInfo& workloadInfo) const
{
    const std::string& descriptorName{"SwitchQueueDescriptor"};

    ValidateNumInputs(workloadInfo,  descriptorName, 2);
    ValidateNumOutputs(workloadInfo, descriptorName, 2);

    const TensorInfo& inputTensorInfo0 = workloadInfo.m_InputTensorInfos[0];
    const TensorInfo& inputTensorInfo1 = workloadInfo.m_InputTensorInfos[1];

    const TensorInfo& outputTensorInfo0 = workloadInfo.m_OutputTensorInfos[0];
    const TensorInfo& outputTensorInfo1 = workloadInfo.m_OutputTensorInfos[1];

    std::vector<DataType> supportedTypes =
    {
        DataType::BFloat16,
        DataType::Float32,
        DataType::QAsymmS8,
        DataType::QAsymmU8,
        DataType::QSymmS16
    };

    ValidateDataTypes(inputTensorInfo0, supportedTypes, descriptorName);
    ValidateDataTypes(inputTensorInfo1, supportedTypes, descriptorName);

    ValidateDataTypes(outputTensorInfo0, supportedTypes, descriptorName);
    ValidateDataTypes(outputTensorInfo1, supportedTypes, descriptorName);

    ValidateTensorShapesMatch(inputTensorInfo0,
                              outputTensorInfo0,
                              descriptorName,
                              "input_0",
                              "output_0");

    ValidateTensorShapesMatch(inputTensorInfo0,
                              outputTensorInfo1,
                              descriptorName,
                              "input_0",
                              "output_1");
}

void PreCompiledQueueDescriptor::Validate(const WorkloadInfo& /*workloadInfo*/) const
{
    // This is internally generated so it should not need validation.
}

void PreluQueueDescriptor::Validate(const WorkloadInfo& workloadInfo) const
{
    const std::string& descriptorName{"PreluQueueDescriptor"};

    ValidateNumInputs(workloadInfo,  descriptorName, 2);
    ValidateNumOutputs(workloadInfo, descriptorName, 1);

    const TensorInfo& inputTensorInfo  = workloadInfo.m_InputTensorInfos[0];
    const TensorInfo& alphaTensorInfo  = workloadInfo.m_InputTensorInfos[1];
    const TensorInfo& outputTensorInfo = workloadInfo.m_OutputTensorInfos[0];

    std::vector<DataType> supportedTypes
    {
        DataType::BFloat16,
        DataType::Float16,
        DataType::Float32,
        DataType::QAsymmS8,
        DataType::QAsymmU8,
        DataType::QSymmS16
    };

    ValidateDataTypes(inputTensorInfo, supportedTypes, descriptorName);
    ValidateDataTypes(alphaTensorInfo, supportedTypes, descriptorName);

    ValidateDataTypes(outputTensorInfo, supportedTypes, descriptorName);

    ValidateTensorDataTypesMatch(inputTensorInfo, alphaTensorInfo,  descriptorName, "input", "alpha");
    ValidateTensorDataTypesMatch(inputTensorInfo, outputTensorInfo, descriptorName, "input", "ouptut");

    ValidateBroadcastTensorShapesMatch(inputTensorInfo,
                                       alphaTensorInfo,
                                       outputTensorInfo,
                                       descriptorName,
                                       "input",
                                       "alpha");
}

void TransposeConvolution2dQueueDescriptor::Validate(const WorkloadInfo& workloadInfo) const
{
    const std::string descriptorName{"TransposeConvolution2dQueueDescriptor"};

    ValidateNumInputs(workloadInfo,  descriptorName, 1);
    ValidateNumOutputs(workloadInfo, descriptorName, 1);

    const TensorInfo& inputTensorInfo  = workloadInfo.m_InputTensorInfos[0];
    const TensorInfo& outputTensorInfo = workloadInfo.m_OutputTensorInfos[0];

    ValidateTensorNumDimensions(inputTensorInfo,  descriptorName, 4, "input");
    ValidateTensorNumDimensions(outputTensorInfo, descriptorName, 4, "output");

    ValidatePointer(m_Weight, descriptorName, "weight");

    const TensorInfo& weightTensorInfo = m_Weight->GetTensorInfo();
    ValidateTensorNumDimensions(weightTensorInfo, descriptorName, 4, "weight");

    ValidateWeightDataType(inputTensorInfo, weightTensorInfo, descriptorName);

    Optional<TensorInfo> optionalBiasTensorInfo;
    if (m_Parameters.m_BiasEnabled)
    {
        ValidatePointer(m_Bias, descriptorName, "bias");

        optionalBiasTensorInfo = MakeOptional<TensorInfo>(m_Bias->GetTensorInfo());
        const TensorInfo& biasTensorInfo = optionalBiasTensorInfo.value();

        ValidateTensorDataType(biasTensorInfo, GetBiasDataType(inputTensorInfo.GetDataType()), descriptorName, "bias");
        ValidateBiasTensorQuantization(biasTensorInfo, inputTensorInfo, weightTensorInfo, descriptorName);
    }

    ValidatePerAxisQuantization(inputTensorInfo,
                                outputTensorInfo,
                                weightTensorInfo,
                                optionalBiasTensorInfo,
                                descriptorName);

    std::vector<DataType> supportedTypes =
    {
        DataType::BFloat16,
        DataType::Float32,
        DataType::Float16,
        DataType::QAsymmS8,
        DataType::QAsymmU8,
        DataType::QSymmS16
    };

    ValidateDataTypes(inputTensorInfo, supportedTypes, descriptorName);
    ValidateTensorDataTypesMatch(inputTensorInfo, outputTensorInfo, descriptorName, "input", "output");
}

void TransposeQueueDescriptor::Validate(const WorkloadInfo& workloadInfo) const
{
    const std::string descriptorName{"TransposeQueueDescriptor"};

    ValidateNumInputs(workloadInfo,  descriptorName, 1);
    ValidateNumOutputs(workloadInfo, descriptorName, 1);

    const PermutationVector& mapping = m_Parameters.m_DimMappings;

    const TensorInfo& inputTensorInfo  = workloadInfo.m_InputTensorInfos[0];
    const TensorInfo& outputTensorInfo = workloadInfo.m_OutputTensorInfos[0];

    ValidateTensorNumDimensions(inputTensorInfo,  descriptorName, mapping.GetSize(), "input");
    ValidateTensorNumDimensions(outputTensorInfo, descriptorName, mapping.GetSize(), "output");

    for (unsigned int i = 0u; i < mapping.GetSize(); ++i)
    {
        if (inputTensorInfo.GetShape()[mapping[i]] != outputTensorInfo.GetShape()[i])
        {
            throw InvalidArgumentException(descriptorName + ": src dimension " + to_string(mapping[i]) +
                                           " (=" + to_string(inputTensorInfo.GetShape()[mapping[i]]) + ") " +
                                           "must match dst dimension " + to_string(i) +
                                           " (=" + to_string(outputTensorInfo.GetShape()[i]) + ")");
        }
    }

    ValidateTensorDataTypesMatch(inputTensorInfo, outputTensorInfo, descriptorName, "input", "output");
}

void ChannelShuffleQueueDescriptor::Validate(const WorkloadInfo &workloadInfo) const
{
    const std::string descriptorName{"TransposeQueueDescriptor"};

    ValidateNumInputs(workloadInfo, descriptorName, 1);
    ValidateNumOutputs(workloadInfo, descriptorName, 1);

    const TensorInfo& inputTensorInfo = workloadInfo.m_InputTensorInfos[0];
    const TensorInfo& outputTensorInfo = workloadInfo.m_OutputTensorInfos[0];

    ValidateTensorDataTypesMatch(inputTensorInfo, outputTensorInfo, descriptorName, "input", "output");
}

void QLstmQueueDescriptor::Validate(const WorkloadInfo& workloadInfo) const
{
    const std::string descriptorName{"QLstmQueueDescriptor"};

    // Validate number of inputs/outputs
    ValidateNumInputs(workloadInfo,  descriptorName, 3);
    ValidateNumOutputs(workloadInfo, descriptorName, 3);

    // Input/output tensor info
    auto inputInfo = workloadInfo.m_InputTensorInfos[0];
    auto outputStateInInfo = workloadInfo.m_InputTensorInfos[1];
    auto cellStateInInfo = workloadInfo.m_InputTensorInfos[2];

    auto outputStateOutInfo = workloadInfo.m_OutputTensorInfos[0];
    auto cellStateOutInfo = workloadInfo.m_OutputTensorInfos[1];
    auto outputInfo = workloadInfo.m_OutputTensorInfos[2];

    // Supported types for various tensors in QLSTM
    std::vector<DataType> inputOutputSupportedTypes =
    {
        DataType::QAsymmS8
    };

    std::vector<DataType> cellStateSupportedTypes =
    {
        DataType::QSymmS16
    };

    std::vector<DataType> weightsSupportedTypes =
    {
        DataType::QSymmS8
    };

    std::vector<DataType> layerNormPeepholeWeightsSupportedTypes =
    {
        DataType::QSymmS16
    };

    std::vector<DataType> biasSupportedTypes =
    {
        DataType::Signed32
    };

    // Validate types of input/output tensors
    ValidateDataTypes(inputInfo, inputOutputSupportedTypes, descriptorName);
    ValidateDataTypes(outputStateInInfo, inputOutputSupportedTypes, descriptorName);
    ValidateDataTypes(cellStateInInfo, cellStateSupportedTypes, descriptorName);

    ValidateDataTypes(outputStateOutInfo, inputOutputSupportedTypes, descriptorName);
    ValidateDataTypes(cellStateOutInfo, cellStateSupportedTypes, descriptorName);
    ValidateDataTypes(outputInfo, inputOutputSupportedTypes, descriptorName);

    // Validate matching types of input/output tensors
    ValidateTensorDataTypesMatch(inputInfo, outputStateInInfo, descriptorName, "input", "outputStateIn");
    ValidateTensorDataTypesMatch(outputStateInInfo, outputStateOutInfo, descriptorName,
                                 "outputStateIn", "outputStateOut");
    ValidateTensorDataTypesMatch(cellStateInInfo, cellStateOutInfo, descriptorName, "cellStateIn", "cellStateOut");

    // Infer number of batches, number of units, input size and output size from tensor dimensions
    const uint32_t numBatches = inputInfo.GetShape()[0];
    const uint32_t inputSize  = inputInfo.GetShape()[1];
    const uint32_t outputSize = outputStateInInfo.GetShape()[1];
    const uint32_t numUnits = cellStateInInfo.GetShape()[1];

    // Validate number of dimensions and number of elements for input/output tensors
    ValidateTensorNumDimNumElem(inputInfo, 2, (numBatches * inputSize), descriptorName + " input");
    ValidateTensorNumDimNumElem(outputStateInInfo, 2, (numBatches * outputSize), descriptorName + " outputStateIn");
    ValidateTensorNumDimNumElem(cellStateInInfo, 2, (numBatches * numUnits), descriptorName + " cellStateIn");

    ValidateTensorNumDimNumElem(outputStateOutInfo, 2, (numBatches * outputSize), descriptorName + " outputStateOut");
    ValidateTensorNumDimNumElem(cellStateOutInfo, 2, (numBatches * numUnits), descriptorName + " cellStateOut");
    ValidateTensorNumDimNumElem(outputInfo, 2, (numBatches * outputSize), descriptorName + " output");

    // Validate number of dimensions and number of elements for MANDATORY weight tensors
    ValidatePointer(m_InputToForgetWeights, descriptorName, "InputToForgetWeights");
    auto inputToForgetWeightsInfo = m_InputToForgetWeights->GetTensorInfo();
    ValidateTensorNumDimNumElem(inputToForgetWeightsInfo, 2, (numUnits * inputSize), " InputToForgetWeights");

    ValidatePointer(m_InputToCellWeights, descriptorName, "InputToCellWeights");
    auto inputToCellWeightsInfo = m_InputToCellWeights->GetTensorInfo();
    ValidateTensorNumDimNumElem(inputToCellWeightsInfo, 2, (numUnits * inputSize), " InputToCellWeights");

    ValidatePointer(m_InputToOutputWeights, descriptorName, "InputToOutputWeights");
    auto inputToOutputWeightsInfo = m_InputToOutputWeights->GetTensorInfo();
    ValidateTensorNumDimNumElem(inputToOutputWeightsInfo, 2, (numUnits * inputSize), " InputToOutputWeights");

    ValidatePointer(m_RecurrentToForgetWeights, descriptorName, "RecurrentToForgetWeights");
    auto recurrentToForgetWeightsInfo = m_RecurrentToForgetWeights->GetTensorInfo();
    ValidateTensorNumDimNumElem(recurrentToForgetWeightsInfo, 2, (numUnits * outputSize),
                                " RecurrentToForgetWeights");

    ValidatePointer(m_RecurrentToCellWeights, descriptorName, "RecurrentToCellWeights");
    auto recurrentToCellWeightsInfo = m_RecurrentToCellWeights->GetTensorInfo();
    ValidateTensorNumDimNumElem(recurrentToCellWeightsInfo, 2, (numUnits * outputSize), " RecurrentToCellWeights");

    ValidatePointer(m_RecurrentToOutputWeights, descriptorName, "RecurrentToOutputWeights");
    auto recurrentToOutputWeightsInfo = m_RecurrentToOutputWeights->GetTensorInfo();
    ValidateTensorNumDimNumElem(recurrentToOutputWeightsInfo, 2, (numUnits * outputSize), " RecurrentToCellWeights");

    // Validate data types for MANDATORY weights tensors (all should match each other)
    ValidateDataTypes(inputToForgetWeightsInfo, weightsSupportedTypes, descriptorName);

    ValidateTensorDataTypesMatch(inputToForgetWeightsInfo, inputToCellWeightsInfo, descriptorName,
                                 "inputToForgetWeights", "inputToCellWeights");
    ValidateTensorDataTypesMatch(inputToForgetWeightsInfo, inputToOutputWeightsInfo, descriptorName,
                                 "inputToForgetWeights", "inputToOutputWeights");

    ValidateTensorDataTypesMatch(inputToForgetWeightsInfo, recurrentToForgetWeightsInfo, descriptorName,
                                 "inputToForgetWeights", "recurrentToForgeteights");
    ValidateTensorDataTypesMatch(inputToForgetWeightsInfo, recurrentToCellWeightsInfo, descriptorName,
                                 "inputToForgetWeights", "recurrentToCellWeights");
    ValidateTensorDataTypesMatch(inputToForgetWeightsInfo, recurrentToOutputWeightsInfo, descriptorName,
                                 "inputToForgetWeights", "recurrentToOutputWeights");

    // Validate number of dimensions and number of elements for MANDATORY bias tensors
    ValidatePointer(m_ForgetGateBias, descriptorName, "ForgetGateBias");
    auto forgetGateBiasInfo = m_ForgetGateBias->GetTensorInfo();
    ValidateTensorNumDimNumElem(forgetGateBiasInfo, 1, numUnits, " ForgetGateBias");

    ValidatePointer(m_CellBias, descriptorName, "CellBias");
    auto cellBiasInfo = m_CellBias->GetTensorInfo();
    ValidateTensorNumDimNumElem(cellBiasInfo, 1, numUnits, " CellBias");

    ValidatePointer(m_OutputGateBias, descriptorName, "OutputGateBias");
    auto outputGateBiasInfo = m_OutputGateBias->GetTensorInfo();
    ValidateTensorNumDimNumElem(outputGateBiasInfo, 1, numUnits, " OutputGateBias");

    // Validate data types for MANDATORY bias tensors
    ValidateDataTypes(forgetGateBiasInfo, biasSupportedTypes, descriptorName);

    ValidateTensorDataTypesMatch(forgetGateBiasInfo, cellBiasInfo, descriptorName,
                                 "forgetGateBias", "cellBias");
    ValidateTensorDataTypesMatch(forgetGateBiasInfo, outputGateBiasInfo, descriptorName,
                                 "forgetGateBias", "outputGateBias");

    // Validate OPTIONAL params: CIFG (inputToInputWeights, recurrentToInputWeights, inputGateBias)
    const bool allCifgParamsPresentOrNot = ((m_InputToInputWeights && m_RecurrentToInputWeights && m_InputGateBias &&
                                             !m_Parameters.m_CifgEnabled) ||
                                            (!m_InputToInputWeights && !m_RecurrentToInputWeights &&
                                             !m_InputGateBias && m_Parameters.m_CifgEnabled));

    if (!allCifgParamsPresentOrNot)
    {
        throw InvalidArgumentException(descriptorName +
                ": InputToInputWeights, RecurrentToInputWeights and InputGateBias must either all be present "
                "(CIFG disabled) or not be present at all (CIFG enabled). m_Parameters.m_CifgEnabled should be "
                "set appropriately.");
    }

    if (!m_Parameters.m_CifgEnabled)
    {
        // Validate number of dimensions and number of elements
        auto inputToInputWeightsInfo = m_InputToInputWeights->GetTensorInfo();
        ValidateTensorNumDimNumElem(inputToInputWeightsInfo, 2, (numUnits * inputSize), " InputToInputWeights");

        auto recurrentToInputWeightsInfo = m_RecurrentToInputWeights->GetTensorInfo();
        ValidateTensorNumDimNumElem(recurrentToInputWeightsInfo, 2, (numUnits * outputSize),
                                    " RecurrentToInputWeights");

        auto inputGateBiasInfo = m_InputGateBias->GetTensorInfo();
        ValidateTensorNumDimNumElem(inputGateBiasInfo, 1, numUnits, " InputGateBias");

        // Validate data types
        ValidateTensorDataTypesMatch(inputToForgetWeightsInfo, inputToInputWeightsInfo, descriptorName,
                                     "inputToForgetWeights", "inputToInputWeights");
        ValidateTensorDataTypesMatch(inputToForgetWeightsInfo, recurrentToInputWeightsInfo, descriptorName,
                                     "inputToForgetWeights", "recurrentToInputWeights");
        ValidateTensorDataTypesMatch(forgetGateBiasInfo, inputGateBiasInfo, descriptorName,
                                     "forgetGateBias", "inputGateBias");
    }

    // Validate OPTIONAL params: Peephole (cellToInputWeights, cellToForgetWeights, cellToOutputWeights)
    bool allPeepholeWeightsPresentOrNot =
            (((m_CellToInputWeights || m_Parameters.m_CifgEnabled) && m_CellToForgetWeights
              && m_CellToOutputWeights && m_Parameters.m_PeepholeEnabled)
             || (!m_CellToInputWeights && !m_CellToForgetWeights
                 && !m_CellToOutputWeights && !m_Parameters.m_PeepholeEnabled));

    if (!allPeepholeWeightsPresentOrNot)
    {
        throw InvalidArgumentException(descriptorName +
                ": CellToInputWeights, CellToForgetWeights and CellToOutputWeights should all be present (Peephole "
                "enabled) or not be present at all (Peephole disabled). CellToInputWeights should only be present "
                "when Peephole is enabled and CIFG is disabled. m_Parameters.m_PeepholeEnabled should be set "
                "appropriately.");
    }

    if (m_Parameters.m_PeepholeEnabled)
    {
        auto cellToForgetWeightsInfo = m_CellToForgetWeights->GetTensorInfo();
        ValidateTensorNumDimNumElem(cellToForgetWeightsInfo, 1, numUnits, " cellToForgetWeights");
        ValidateDataTypes(cellToForgetWeightsInfo, layerNormPeepholeWeightsSupportedTypes, descriptorName);

        auto cellToOutputWeightsInfo = m_CellToOutputWeights->GetTensorInfo();
        ValidateTensorNumDimNumElem(cellToOutputWeightsInfo, 1, numUnits, " cellToOutputWeights");
        ValidateTensorDataTypesMatch(cellToForgetWeightsInfo, cellToOutputWeightsInfo, descriptorName,
                                     "cellToForgetWeight", "cellToOutputWeights");

        if (!m_Parameters.m_CifgEnabled)
        {
            auto cellToInputWeightsInfo = m_CellToInputWeights->GetTensorInfo();
            ValidateTensorNumDimNumElem(cellToInputWeightsInfo, 1, numUnits, " cellToInputWeights");
            ValidateTensorDataTypesMatch(cellToForgetWeightsInfo, cellToInputWeightsInfo, descriptorName,
                                         "cellToForgetWeights", "cellToInputWeights");
        }
    }

    // Validate OPTIONAL params: Layer Norm Weights
    bool allLayerNormWeightsPresentOrNot =
            (((m_InputLayerNormWeights || m_Parameters.m_CifgEnabled) && m_ForgetLayerNormWeights
              && m_CellLayerNormWeights && m_OutputLayerNormWeights && m_Parameters.m_LayerNormEnabled)
             || (!m_InputLayerNormWeights && !m_ForgetLayerNormWeights && !m_CellLayerNormWeights
                 && !m_OutputLayerNormWeights && !m_Parameters.m_LayerNormEnabled));

    if (!allLayerNormWeightsPresentOrNot)
    {
        throw InvalidArgumentException(descriptorName +
                                       ": InputLayerNormWeights, ForgetLayerNormWeights, m_OutputLayerNormWeights "
                                       "and CellLayerNormWeights should all be present (Layer Norm enabled) or not "
                                       "be present at all (Layer Norm disabled). InputLayerNormWeights should "
                                       "only be present when Layer Norm is enabled and CIFG is disabled. "
                                       "m_Parameters.m_LayerNormEnabled should be set appropriately.");
    }

    if (m_Parameters.m_LayerNormEnabled)
    {
        auto forgetLayerNormWeightsInfo = m_ForgetLayerNormWeights->GetTensorInfo();
        ValidateTensorNumDimNumElem(forgetLayerNormWeightsInfo, 1, numUnits, " forgetLayerNormWeights");
        ValidateDataTypes(forgetLayerNormWeightsInfo, layerNormPeepholeWeightsSupportedTypes, descriptorName);

        auto cellLayerNormWeightsInfo = m_CellLayerNormWeights->GetTensorInfo();
        ValidateTensorNumDimNumElem(cellLayerNormWeightsInfo, 1, numUnits, " cellLayerNormWeights");
        ValidateTensorDataTypesMatch(forgetLayerNormWeightsInfo, cellLayerNormWeightsInfo, descriptorName,
                                     "forgetLayerNormWeights", "cellLayerNormWeights");

        auto outputLayerNormWeightsInfo = m_OutputLayerNormWeights->GetTensorInfo();
        ValidateTensorNumDimNumElem(outputLayerNormWeightsInfo, 1, numUnits, " outputLayerNormWeights");
        ValidateTensorDataTypesMatch(forgetLayerNormWeightsInfo, outputLayerNormWeightsInfo, descriptorName,
                                     "forgetLayerNormWeights", "outputLayerNormWeights");

        if (!m_Parameters.m_CifgEnabled)
        {
            auto inputLayerNormWeightsInfo = m_InputLayerNormWeights->GetTensorInfo();
            ValidateTensorNumDimNumElem(inputLayerNormWeightsInfo, 1, numUnits, " inputLayerNormWeights");
            ValidateTensorDataTypesMatch(forgetLayerNormWeightsInfo, inputLayerNormWeightsInfo, descriptorName,
                                         "forgetLayerNormWeights", "inputLayerNormWeights");
        }
    }

    // Validate OPTIONAL params: Projection (projectionWeights, projectionBias)
    bool correctProjectionTensorsPresent =
            ((!m_ProjectionWeights && !m_ProjectionBias && !m_Parameters.m_ProjectionEnabled) ||
            (m_ProjectionWeights && !m_ProjectionBias && m_Parameters.m_ProjectionEnabled) ||
            (m_ProjectionWeights && m_ProjectionBias && m_Parameters.m_ProjectionEnabled));

    if (!correctProjectionTensorsPresent)
    {
        throw InvalidArgumentException(descriptorName +
                                       ": If projection is enabled, ProjectionWeights should be present and "
                                       "ProjectionBias is optional. If projection is disabled, neither "
                                       "ProjectionWeights nor ProjectionBias should be present.");
    }

    if (m_Parameters.m_ProjectionEnabled)
    {
        auto projectionWeightsInfo = m_ProjectionWeights->GetTensorInfo();
        ValidateTensorNumDimNumElem(projectionWeightsInfo, 2, (numUnits * outputSize), "ProjectionWeights");
        ValidateDataTypes(projectionWeightsInfo, weightsSupportedTypes, descriptorName);

        if (m_ProjectionBias)
        {
            auto projectionBiasInfo = m_ProjectionBias->GetTensorInfo();
            ValidateTensorNumDimNumElem(projectionBiasInfo, 1, outputSize, "ProjectionBias");
            ValidateDataTypes(projectionBiasInfo, biasSupportedTypes, descriptorName);
        }

    }
    else if ((outputInfo.GetQuantizationScale() != m_Parameters.m_HiddenStateScale) &&
              outputInfo.GetQuantizationOffset() != m_Parameters.m_HiddenStateZeroPoint) {
        throw InvalidArgumentException(descriptorName +
                                       ": If projection is disabled, output quantization info (scale, offset) "
                                       "should match HiddenStateScale and HiddenStateZeroPoint.");
    }

}

void QuantizedLstmQueueDescriptor::Validate(const WorkloadInfo& workloadInfo) const
{
    const std::string descriptorName{"QuantizedLstmQueueDescriptor"};

    // Validate number of inputs/outputs
    ValidateNumInputs(workloadInfo,  descriptorName, 3);
    ValidateNumOutputs(workloadInfo, descriptorName, 2);

    // Input/output tensor infos
    auto inputInfo = workloadInfo.m_InputTensorInfos[0];
    auto cellStateInInfo = workloadInfo.m_InputTensorInfos[1];
    auto outputStateInInfo = workloadInfo.m_InputTensorInfos[2];

    auto cellStateOutInfo = workloadInfo.m_OutputTensorInfos[0];
    auto outputStateOutInfo = workloadInfo.m_OutputTensorInfos[1];

    std::vector<DataType> inputOutputSupportedTypes =
    {
        DataType::QAsymmU8
    };

    std::vector<DataType> cellStateSupportedTypes =
    {
        DataType::QSymmS16
    };

    std::vector<DataType> weightsSupportedTypes =
    {
        DataType::QAsymmU8
    };

    std::vector<DataType> biasSupportedTypes =
    {
        DataType::Signed32
    };

    // Validate types of input/output tensors
    ValidateDataTypes(inputInfo, inputOutputSupportedTypes, descriptorName);
    ValidateDataTypes(cellStateInInfo, cellStateSupportedTypes, descriptorName);
    ValidateDataTypes(outputStateInInfo, inputOutputSupportedTypes, descriptorName);

    ValidateDataTypes(cellStateOutInfo, cellStateSupportedTypes, descriptorName);
    ValidateDataTypes(outputStateOutInfo, inputOutputSupportedTypes, descriptorName);

    // Validate matching types of input/output tensors
    ValidateTensorDataTypesMatch(inputInfo, outputStateInInfo, descriptorName, "input", "outputStateIn");
    ValidateTensorDataTypesMatch(outputStateInInfo, outputStateOutInfo, descriptorName,
                                 "outputStateIn", "outputStateOut");
    ValidateTensorDataTypesMatch(cellStateInInfo, cellStateOutInfo, descriptorName, "cellStateIn", "cellStateOut");

    // Validate matching quantization info for input/output tensors
    ValidateTensorQuantizationSpace(inputInfo, outputStateInInfo, descriptorName, "input", "outputStateIn");
    ValidateTensorQuantizationSpace(inputInfo, outputStateOutInfo, descriptorName, "input", "outputStateOut");
    ValidateTensorQuantizationSpace(cellStateInInfo, cellStateOutInfo, descriptorName, "cellStateIn", "cellStateOut");

    // Infer number of batches, input size and output size from tensor dimensions
    const uint32_t numBatches = inputInfo.GetShape()[0];
    const uint32_t inputSize  = inputInfo.GetShape()[1];
    const uint32_t outputSize = cellStateInInfo.GetShape()[1];

    // Validate number of dimensions and number of elements for input/output tensors
    ValidateTensorNumDimNumElem(inputInfo, 2, (numBatches * inputSize), descriptorName + " input");
    ValidateTensorNumDimNumElem(cellStateInInfo, 2, (numBatches * outputSize), descriptorName + " cellStateIn");
    ValidateTensorNumDimNumElem(outputStateInInfo, 2, (numBatches * outputSize), descriptorName + " outputStateIn");
    ValidateTensorNumDimNumElem(cellStateOutInfo, 2, (numBatches * outputSize), descriptorName + " cellStateOut");
    ValidateTensorNumDimNumElem(outputStateOutInfo, 2, (numBatches * outputSize), descriptorName + " outputStateOut");

    // Validate number of dimensions and number of elements for weights tensors
    ValidatePointer(m_InputToInputWeights, descriptorName, "InputToInputWeights");
    auto inputToInputWeightsInfo = m_InputToInputWeights->GetTensorInfo();
    ValidateTensorNumDimNumElem(inputToInputWeightsInfo, 2, (outputSize * inputSize), " InputToInputWeights");

    ValidatePointer(m_InputToForgetWeights, descriptorName, "InputToForgetWeights");
    auto inputToForgetWeightsInfo = m_InputToForgetWeights->GetTensorInfo();
    ValidateTensorNumDimNumElem(inputToForgetWeightsInfo, 2, (outputSize * inputSize), " InputToForgetWeights");

    ValidatePointer(m_InputToCellWeights, descriptorName, "InputToCellWeights");
    auto inputToCellWeightsInfo = m_InputToCellWeights->GetTensorInfo();
    ValidateTensorNumDimNumElem(inputToCellWeightsInfo, 2, (outputSize * inputSize), " InputToCellWeights");

    ValidatePointer(m_InputToOutputWeights, descriptorName, "InputToOutputWeights");
    auto inputToOutputWeightsInfo = m_InputToOutputWeights->GetTensorInfo();
    ValidateTensorNumDimNumElem(inputToOutputWeightsInfo, 2, (outputSize * inputSize), " InputToOutputWeights");

    ValidatePointer(m_RecurrentToInputWeights, descriptorName, "RecurrentToInputWeights");
    auto recurrentToInputWeightsInfo = m_RecurrentToInputWeights->GetTensorInfo();
    ValidateTensorNumDimNumElem(recurrentToInputWeightsInfo, 2, (outputSize * outputSize), " RecurrentToInputWeights");

    ValidatePointer(m_RecurrentToForgetWeights, descriptorName, "RecurrentToForgetWeights");
    auto recurrentToForgetWeightsInfo = m_RecurrentToForgetWeights->GetTensorInfo();
    ValidateTensorNumDimNumElem(recurrentToForgetWeightsInfo, 2, (outputSize * outputSize),
                                " RecurrentToForgetWeights");

    ValidatePointer(m_RecurrentToCellWeights, descriptorName, "RecurrentToCellWeights");
    auto recurrentToCellWeightsInfo = m_RecurrentToCellWeights->GetTensorInfo();
    ValidateTensorNumDimNumElem(recurrentToCellWeightsInfo, 2, (outputSize * outputSize), " RecurrentToCellWeights");

    ValidatePointer(m_RecurrentToOutputWeights, descriptorName, "RecurrentToOutputWeights");
    auto recurrentToOutputWeightsInfo = m_RecurrentToOutputWeights->GetTensorInfo();
    ValidateTensorNumDimNumElem(recurrentToOutputWeightsInfo, 2, (outputSize * outputSize), " RecurrentToCellWeights");

    // Validate data types for weights tensors (all should match each other)
    ValidateDataTypes(inputToInputWeightsInfo, weightsSupportedTypes, descriptorName);

    ValidateTensorDataTypesMatch(inputToInputWeightsInfo, inputToForgetWeightsInfo, descriptorName,
                                 "inputToInputWeights", "inputToForgetWeights");
    ValidateTensorDataTypesMatch(inputToInputWeightsInfo, inputToCellWeightsInfo, descriptorName,
                                 "inputToInputWeights", "inputToCellWeights");
    ValidateTensorDataTypesMatch(inputToInputWeightsInfo, inputToOutputWeightsInfo, descriptorName,
                                 "inputToInputWeights", "inputToOutputWeights");

    ValidateTensorDataTypesMatch(inputToInputWeightsInfo, recurrentToInputWeightsInfo, descriptorName,
                                 "inputToInputWeights", "recurrentToInputWeights");
    ValidateTensorDataTypesMatch(inputToInputWeightsInfo, recurrentToForgetWeightsInfo, descriptorName,
                                 "inputToInputWeights", "recurrentToForgeteights");
    ValidateTensorDataTypesMatch(inputToInputWeightsInfo, recurrentToCellWeightsInfo, descriptorName,
                                 "inputToInputWeights", "recurrentToCellWeights");
    ValidateTensorDataTypesMatch(inputToInputWeightsInfo, recurrentToOutputWeightsInfo, descriptorName,
                                 "inputToInputWeights", "recurrentToOutputWeights");

    // Validate matching quantization info for weight tensors (all should match each other)
    ValidateTensorQuantizationSpace(inputToInputWeightsInfo, inputToForgetWeightsInfo,
                                    descriptorName, "inputToInputWeights", "inputToForgetWeights");
    ValidateTensorQuantizationSpace(inputToInputWeightsInfo, inputToCellWeightsInfo,
                                    descriptorName, "inputToInputWeights", "inputToCellWeights");
    ValidateTensorQuantizationSpace(inputToInputWeightsInfo, inputToOutputWeightsInfo,
                                    descriptorName, "inputToInputWeights", "inputToOutputWeights");

    ValidateTensorQuantizationSpace(inputToInputWeightsInfo, recurrentToInputWeightsInfo,
                                    descriptorName, "inputToInputWeights", "recurrentToInputWeights");
    ValidateTensorQuantizationSpace(inputToInputWeightsInfo, recurrentToForgetWeightsInfo,
                                    descriptorName, "inputToInputWeights", "recurrentToForgetWeights");
    ValidateTensorQuantizationSpace(inputToInputWeightsInfo, recurrentToCellWeightsInfo,
                                    descriptorName, "inputToInputWeights", "recurrentToCellWeights");
    ValidateTensorQuantizationSpace(inputToInputWeightsInfo, recurrentToOutputWeightsInfo,
                                    descriptorName, "inputToInputWeights", "recurrentToOutputWeights");

    // Validate number of dimensions and number of elements in bias tensors
    ValidatePointer(m_InputGateBias, descriptorName, "InputGateBias");
    auto inputGateBiasInfo = m_InputGateBias->GetTensorInfo();
    ValidateTensorNumDimNumElem(inputGateBiasInfo, 1, outputSize, " InputGateBias");

    ValidatePointer(m_ForgetGateBias, descriptorName, "ForgetGateBias");
    auto forgetGateBiasInfo = m_ForgetGateBias->GetTensorInfo();
    ValidateTensorNumDimNumElem(forgetGateBiasInfo, 1, outputSize, " ForgetGateBias");

    ValidatePointer(m_CellBias, descriptorName, "CellBias");
    auto cellBiasInfo = m_CellBias->GetTensorInfo();
    ValidateTensorNumDimNumElem(cellBiasInfo, 1, outputSize, " CellBias");

    ValidatePointer(m_OutputGateBias, descriptorName, "OutputGateBias");
    auto outputGateBiasInfo = m_OutputGateBias->GetTensorInfo();
    ValidateTensorNumDimNumElem(outputGateBiasInfo, 1, outputSize, " OutputGateBias");

    // Validate data types for bias tensors (all should match each other)
    ValidateDataTypes(inputGateBiasInfo, biasSupportedTypes, descriptorName);

    ValidateTensorDataTypesMatch(inputGateBiasInfo, forgetGateBiasInfo, descriptorName,
                                 "inputGateBias", "forgetGateBias");
    ValidateTensorDataTypesMatch(inputGateBiasInfo, cellBiasInfo, descriptorName,
                                 "inputGateBias", "cellBias");
    ValidateTensorDataTypesMatch(inputGateBiasInfo, outputGateBiasInfo, descriptorName,
                                 "inputGateBias", "outputGateBias");

    // Validate bias tensor quantization info
    ValidateBiasTensorQuantization(inputGateBiasInfo, inputInfo, inputToInputWeightsInfo, descriptorName);
    ValidateBiasTensorQuantization(forgetGateBiasInfo, inputInfo, inputToInputWeightsInfo, descriptorName);
    ValidateBiasTensorQuantization(cellBiasInfo, inputInfo, inputToInputWeightsInfo, descriptorName);
    ValidateBiasTensorQuantization(outputGateBiasInfo, inputInfo, inputToInputWeightsInfo, descriptorName);
}

void AbsQueueDescriptor::Validate(const WorkloadInfo& workloadInfo) const
{
    const std::string descriptorName{"AbsQueueDescriptor"};

    ValidateNumInputs(workloadInfo,  descriptorName, 1);
    ValidateNumOutputs(workloadInfo, descriptorName, 1);

    const TensorInfo& inputTensorInfo  = workloadInfo.m_InputTensorInfos[0];
    const TensorInfo& outputTensorInfo = workloadInfo.m_OutputTensorInfos[0];

    ValidateTensorShapesMatch(inputTensorInfo, outputTensorInfo, descriptorName, "input", "output");

    std::vector<DataType> supportedTypes =
    {
        DataType::BFloat16,
        DataType::Float16,
        DataType::Float32,
        DataType::QAsymmS8,
        DataType::QAsymmU8,
        DataType::QSymmS16,
        DataType::Signed32
    };

    ValidateDataTypes(inputTensorInfo, supportedTypes, descriptorName);
    ValidateTensorDataTypesMatch(inputTensorInfo, outputTensorInfo, descriptorName, "input", "output");
}

void SliceQueueDescriptor::Validate(const WorkloadInfo& workloadInfo) const
{
    const std::string descriptorName{"SliceQueueDescriptor"};

    ValidateNumInputs(workloadInfo,  descriptorName, 1);
    ValidateNumOutputs(workloadInfo, descriptorName, 1);

    const TensorInfo& inputTensorInfo  = workloadInfo.m_InputTensorInfos[0];
    const TensorInfo& outputTensorInfo = workloadInfo.m_OutputTensorInfos[0];

    ValidateTensorDataTypesMatch(inputTensorInfo, outputTensorInfo, descriptorName, "input", "output");

    const unsigned int rank = inputTensorInfo.GetNumDimensions();
    if (rank > 4)
    {
        throw InvalidArgumentException(descriptorName + ": Input tensors with rank greater than 4 are not supported.");
    }

    ValidateTensorNumDimensions(outputTensorInfo, descriptorName, rank, "output");

    // Check if m_Begin and m_Size have the expected length
    if (m_Parameters.m_Begin.size() != rank)
    {
        throw InvalidArgumentException(descriptorName +
            ": Length of begin offset descriptor must equal rank " + std::to_string(rank));
    }
    if (m_Parameters.m_Size.size() != rank)
    {
        throw InvalidArgumentException(descriptorName +
            ": Length of size descriptor must equal rank " + std::to_string(rank));
    }

    // Check if the shape of the output tensor matches m_Size
    const TensorShape& outputShape = outputTensorInfo.GetShape();
    for (unsigned int i = 0u; i < rank; ++i)
    {
        if (m_Parameters.m_Size[i] != outputShape[i])
        {
            throw InvalidArgumentException(descriptorName + ": Size descriptor does not match output tensor.");
        }
    }

    // Check if the sum of begin offset and size in a given dimension
    // does not exceed the size of corresponding input
    const TensorShape& inputShape  = inputTensorInfo.GetShape();
    for(unsigned int i = 0u; i < rank; ++i)
    {
        if (m_Parameters.m_Begin[i] + m_Parameters.m_Size[i] > inputShape[i])
        {
            throw InvalidArgumentException(descriptorName + ": Sum of begin offset and size for dimension " +
                std::to_string(i) + " exceeds input size.");
        }
    }
}

void DepthToSpaceQueueDescriptor::Validate(const WorkloadInfo& workloadInfo) const
{
    const std::string descriptorName{"DepthToSpaceQueueDescriptor"};

    ValidateNumInputs(workloadInfo,  descriptorName, 1);
    ValidateNumOutputs(workloadInfo, descriptorName, 1);

    const TensorInfo& inputInfo  = workloadInfo.m_InputTensorInfos[0];
    const TensorInfo& outputInfo = workloadInfo.m_OutputTensorInfos[0];

    ValidateTensorNumDimensions(inputInfo,  descriptorName, 4, "input");
    ValidateTensorNumDimensions(outputInfo, descriptorName, 4, "output");

    std::vector<DataType> supportedTypes =
    {
        DataType::BFloat16,
        DataType::Float32,
        DataType::Float16,
        DataType::QAsymmS8,
        DataType::QAsymmU8,
        DataType::QSymmS16
    };

    ValidateDataTypes(inputInfo,  supportedTypes, descriptorName);
    ValidateDataTypes(outputInfo, supportedTypes, descriptorName);

    ValidateTensorNumElementsMatch(inputInfo, outputInfo, descriptorName, "input", "output");

    if (m_Parameters.m_BlockSize == 0)
    {
        throw InvalidArgumentException(descriptorName + ": Block size cannot be 0.");
    }

    DataLayoutIndexed dimensionIndices(m_Parameters.m_DataLayout);
    const unsigned int wIndex = dimensionIndices.GetWidthIndex();
    const unsigned int hIndex = dimensionIndices.GetHeightIndex();
    const unsigned int cIndex = dimensionIndices.GetChannelsIndex();

    const TensorShape& outputShape = outputInfo.GetShape();
    if (outputShape[hIndex] % m_Parameters.m_BlockSize != 0 || outputShape[wIndex]  % m_Parameters.m_BlockSize != 0)
    {
        throw InvalidArgumentException(descriptorName + ": Output width and height shape"
                                       "must be divisible by block size.");
    }

    const TensorShape& inputShape = inputInfo.GetShape();
    if (inputShape[cIndex] % (m_Parameters.m_BlockSize * m_Parameters.m_BlockSize) != 0)
    {
        throw InvalidArgumentException(descriptorName + ": The depth of the input tensor"
                                       "must be divisible by the square of block size." );
    }
}

void ComparisonQueueDescriptor::Validate(const WorkloadInfo& workloadInfo) const
{
    const std::string descriptorName{"ComparisonQueueDescriptor"};

    ValidateNumInputs(workloadInfo,  descriptorName, 2);
    ValidateNumOutputs(workloadInfo, descriptorName, 1);

    const TensorInfo& inputTensorInfo0 = workloadInfo.m_InputTensorInfos[0];
    const TensorInfo& inputTensorInfo1 = workloadInfo.m_InputTensorInfos[1];
    const TensorInfo& outputTensorInfo = workloadInfo.m_OutputTensorInfos[0];

    ValidateBroadcastTensorShapesMatch(inputTensorInfo0,
                                       inputTensorInfo1,
                                       outputTensorInfo,
                                       descriptorName,
                                       "input_0",
                                       "input_1");

    if (outputTensorInfo.GetDataType() != DataType::Boolean)
    {
        throw InvalidArgumentException(descriptorName + ": Output tensor type must be Boolean.");
    }
}

void ElementwiseUnaryQueueDescriptor::Validate(const WorkloadInfo& workloadInfo) const
{
    const std::string descriptorName{"ElementwiseUnaryQueueDescriptor"};

    ValidateNumInputs(workloadInfo,  descriptorName, 1);
    ValidateNumOutputs(workloadInfo, descriptorName, 1);

    const TensorInfo& inputTensorInfo = workloadInfo.m_InputTensorInfos[0];
    const TensorInfo& outputTensorInfo = workloadInfo.m_OutputTensorInfos[0];

    ValidateTensorShapesMatch(inputTensorInfo, outputTensorInfo, descriptorName, "input", "output");

    std::vector<DataType> supportedTypes =
    {
        DataType::BFloat16,
        DataType::Float16,
        DataType::Float32,
        DataType::QAsymmS8,
        DataType::QAsymmU8,
        DataType::QSymmS16,
        DataType::Signed32
    };

    std::vector<DataType> logicalSupportedTypes =
    {
        DataType::Boolean
    };

    if (m_Parameters.m_Operation == UnaryOperation::LogicalNot)
    {
        ValidateDataTypes(inputTensorInfo, logicalSupportedTypes, descriptorName);
    }
    else
    {
        ValidateDataTypes(inputTensorInfo, supportedTypes, descriptorName);
    }


    ValidateTensorDataTypesMatch(inputTensorInfo, outputTensorInfo, descriptorName, "input", "output");
}

void RankQueueDescriptor::Validate(const WorkloadInfo& workloadInfo) const
{
    const std::string descriptorName{"RankQueueDescriptor"};

    ValidateNumInputs(workloadInfo,  descriptorName, 1);
    ValidateNumOutputs(workloadInfo, descriptorName, 1);

    const TensorInfo& inputTensorInfo  = workloadInfo.m_InputTensorInfos[0];
    const TensorInfo& outputTensorInfo = workloadInfo.m_OutputTensorInfos[0];

    ValidateTensorNumDimensions(outputTensorInfo, descriptorName, 1, "output");
    ValidateTensorNumElements(outputTensorInfo, descriptorName, 1, "output");

    std::vector<DataType> supportedTypes =
    {
        DataType::BFloat16,
        DataType::Float16,
        DataType::Float32,
        DataType::QAsymmS8,
        DataType::QAsymmU8,
        DataType::QSymmS8,
        DataType::QSymmS16,
        DataType::Signed32
    };

    ValidateDataTypes(inputTensorInfo, supportedTypes, descriptorName);
    ValidateDataTypes(outputTensorInfo, { DataType::Signed32 }, descriptorName);
}

void LogicalBinaryQueueDescriptor::Validate(const WorkloadInfo& workloadInfo) const
{
    const std::string descriptorName{"LogicalBinaryQueueDescriptor"};

    ValidateNumInputs(workloadInfo,  descriptorName, 2);
    ValidateNumOutputs(workloadInfo, descriptorName, 1);

    const TensorInfo& inputTensorInfo0 = workloadInfo.m_InputTensorInfos[0];
    const TensorInfo& inputTensorInfo1 = workloadInfo.m_InputTensorInfos[1];
    const TensorInfo& outputTensorInfo = workloadInfo.m_OutputTensorInfos[0];

    ValidateBroadcastTensorShapesMatch(inputTensorInfo0,
                                       inputTensorInfo1,
                                       outputTensorInfo,
                                       descriptorName,
                                       "input_0",
                                       "input_1");

    if (inputTensorInfo0.GetDataType() != DataType::Boolean)
    {
        throw InvalidArgumentException(descriptorName + ": Input tensor 0 type must be Boolean.");
    }

    if (inputTensorInfo1.GetDataType() != DataType::Boolean)
    {
        throw InvalidArgumentException(descriptorName + ": Input tensor 1 type must be Boolean.");
    }

    if (outputTensorInfo.GetDataType() != DataType::Boolean)
    {
        throw InvalidArgumentException(descriptorName + ": Output tensor type must be Boolean.");
    }
}

void ReduceQueueDescriptor::Validate(const WorkloadInfo& workloadInfo) const
{
    const std::string descriptorName{"ReduceQueueDescriptor"};

    ValidateNumInputs(workloadInfo,  descriptorName, 1);
    ValidateNumOutputs(workloadInfo, descriptorName, 1);

    const TensorInfo& inputTensorInfo  = workloadInfo.m_InputTensorInfos[0];
    const TensorInfo& outputTensorInfo = workloadInfo.m_OutputTensorInfos[0];

    std::vector<DataType> supportedTypes =
    {
        DataType::BFloat16,
        DataType::Float16,
        DataType::Float32,
        DataType::QAsymmS8,
        DataType::QAsymmU8,
        DataType::QSymmS16,
        DataType::Signed32
    };

    ValidateDataTypes(inputTensorInfo, supportedTypes, descriptorName);
    ValidateTensorDataTypesMatch(inputTensorInfo, outputTensorInfo, descriptorName, "input", "output");
}

void UnidirectionalSequenceLstmQueueDescriptor::Validate(const WorkloadInfo& workloadInfo) const
{
    // Modified from LstmQueueDescriptor::Validate to support UnidirectionalSequenceLstm

    const std::string descriptorName{"UnidirectionalSequenceLstmQueueDescriptor"};

    // check dimensions of all inputs and outputs
    if (workloadInfo.m_InputTensorInfos.size() != 3)
    {
        throw InvalidArgumentException(descriptorName + ": Invalid number of inputs.");
    }
    if (workloadInfo.m_OutputTensorInfos.size() != 1)
    {
        throw InvalidArgumentException(descriptorName + ": Invalid number of outputs.");
    }

    std::vector<DataType> supportedTypes =
    {
        DataType::Float32
    };

    // check for supported type of one input and match them with all the other input and output
    ValidateDataTypes(workloadInfo.m_InputTensorInfos[0], supportedTypes, descriptorName);

    // type matches all other inputs
    for (uint32_t i = 1u; i < workloadInfo.m_InputTensorInfos.size(); ++i)
    {
        ValidateTensorDataTypesMatch(workloadInfo.m_InputTensorInfos[0],
                                     workloadInfo.m_InputTensorInfos[i],
                                     descriptorName,
                                     "input_0",
                                     "input_" + std::to_string(i));
    }
    // type matches all other outputs
    for (uint32_t i = 0u; i < workloadInfo.m_OutputTensorInfos.size(); ++i)
    {
        ValidateTensorDataTypesMatch(workloadInfo.m_InputTensorInfos[0],
                                     workloadInfo.m_OutputTensorInfos[i],
                                     "LstmQueueDescriptor",
                                     "input_0",
                                     "output_" + std::to_string(i));
    }

    // Making sure clipping parameters have valid values.
    // == 0 means no clipping
    //  > 0 means clipping
    if (m_Parameters.m_ClippingThresCell < 0.0f)
    {
        throw InvalidArgumentException(descriptorName + ": negative cell clipping threshold is invalid");
    }
    if (m_Parameters.m_ClippingThresProj < 0.0f)
    {
        throw InvalidArgumentException(descriptorName + ": negative projection clipping threshold is invalid");
    }

    unsigned int batchIndx = 0;
    unsigned int inputIndx = 1;
    uint32_t timeStep = 1;
    unsigned int timeIndx = 1;
    inputIndx = 2;
    if (m_Parameters.m_TimeMajor)
    {
        batchIndx = 1;
        timeIndx = 0;

    }
    timeStep = workloadInfo.m_InputTensorInfos[0].GetShape()[timeIndx];

    // Inferring batch size, number of outputs and number of cells from the inputs.
    const uint32_t n_input = workloadInfo.m_InputTensorInfos[0].GetShape()[inputIndx];
    const uint32_t n_batch = workloadInfo.m_InputTensorInfos[0].GetShape()[batchIndx];
    ValidatePointer(m_InputToOutputWeights, "Null pointer check", "InputToOutputWeights");
    const uint32_t n_cell = m_InputToOutputWeights->GetShape()[0];
    ValidatePointer(m_RecurrentToOutputWeights, "Null pointer check", "RecurrentToOutputWeights");
    const uint32_t n_output = m_RecurrentToOutputWeights->GetShape()[1];

    // input tensor
    ValidateTensorNumDimNumElem(workloadInfo.m_InputTensorInfos[0], 3, (timeStep * n_batch * n_input),
                                descriptorName + " input_0");
    // outputStateInTensor
    ValidateTensorNumDimNumElem(workloadInfo.m_InputTensorInfos[1], 2, (n_batch * n_output),
                                descriptorName + " input_1");
    // outputStateInTensor
    ValidateTensorNumDimNumElem(workloadInfo.m_InputTensorInfos[2], 2, (n_batch * n_cell),
                                descriptorName + " input_2");

    // outputTensor
    ValidateTensorNumDimNumElem(workloadInfo.m_OutputTensorInfos[0], 3, (timeStep * n_batch * n_output),
                                descriptorName + " output_0");

    // check that dimensions of inputs/outputs and QueueDescriptor data match with each other
    if ( m_InputToInputWeights )
    {
        ValidateTensorNumDimNumElem(m_InputToInputWeights->GetTensorInfo(), 2,
                                      (n_cell * n_input), "InputLayerNormWeights");
    }

    ValidatePointer(m_InputToForgetWeights, "Null pointer check", "InputToForgetWeights");
    ValidateTensorNumDimNumElem(m_InputToForgetWeights->GetTensorInfo(), 2,
                                  (n_cell * n_input), "InputToForgetWeights");

    ValidatePointer(m_InputToCellWeights, "Null pointer check", "InputToCellWeights");
    ValidateTensorNumDimNumElem(m_InputToCellWeights->GetTensorInfo(), 2,
                                  (n_cell * n_input), "InputToCellWeights");

    if ( m_RecurrentToInputWeights )
    {
        ValidateTensorNumDimNumElem(m_RecurrentToInputWeights->GetTensorInfo(), 2,
                                      (n_cell * n_output), "RecurrentToInputWeights");
    }

    ValidatePointer(m_RecurrentToForgetWeights, "Null pointer check", "RecurrentToForgetWeights");
    ValidateTensorNumDimNumElem(m_RecurrentToForgetWeights->GetTensorInfo(), 2,
                                  (n_cell * n_output), "RecurrentToForgetWeights");

    ValidatePointer(m_RecurrentToCellWeights, "Null pointer check", "RecurrentToCellWeights");
    ValidateTensorNumDimNumElem(m_RecurrentToCellWeights->GetTensorInfo(), 2,
                                  (n_cell * n_output), "RecurrentToCellWeights");

    // Make sure the input-gate's parameters are either both present (regular
    // LSTM) or not at all (CIFG-LSTM). And CifgEnable is set accordingly.
    bool cifg_weights_all_or_none = ((m_InputToInputWeights && m_RecurrentToInputWeights &&
                                     !m_Parameters.m_CifgEnabled) ||
                                     (!m_InputToInputWeights && !m_RecurrentToInputWeights &&
                                     m_Parameters.m_CifgEnabled));
    if (!cifg_weights_all_or_none)
    {
        throw InvalidArgumentException(descriptorName + ": Input-Gate's parameters InputToInputWeights and "
                                       "RecurrentToInputWeights must either both be present (regular LSTM) "
                                       "or both not present (CIFG-LSTM). In addition CifgEnable must be set "
                                       "accordingly.");
    }

    if ( m_CellToInputWeights )
    {
        ValidateTensorNumDimNumElem(m_CellToInputWeights->GetTensorInfo(), 1,
                                      n_cell, "CellToInputWeights");
    }
    if ( m_CellToForgetWeights )
    {
        ValidateTensorNumDimNumElem(m_CellToForgetWeights->GetTensorInfo(), 1,
                                      n_cell, "CellToForgetWeights");
    }
    if ( m_CellToOutputWeights )
    {
        ValidateTensorNumDimNumElem(m_CellToOutputWeights->GetTensorInfo(), 1,
                                      n_cell, "CellToOutputWeights");
    }

    // Making sure the peephole weights are there all or none. And PeepholeEnable is set accordingly.
    bool peephole_weights_all_or_none =
            (((m_CellToInputWeights || m_Parameters.m_CifgEnabled) &&  m_CellToForgetWeights
            && m_CellToOutputWeights && m_Parameters.m_PeepholeEnabled)
            || ( !m_CellToInputWeights && !m_CellToForgetWeights
            && !m_CellToOutputWeights && !m_Parameters.m_PeepholeEnabled));
    if (!peephole_weights_all_or_none)
    {
        throw InvalidArgumentException(descriptorName + ": Invalid combination of peephole parameters.");
    }

    // Make sure the input gate bias is present only when not a CIFG-LSTM.
    if (m_Parameters.m_CifgEnabled)
    {
        if (m_InputGateBias)
        {
            throw InvalidArgumentException(descriptorName + ": InputGateBias is present and CIFG-LSTM is enabled.");
        }
    }
    else
    {
        if (!m_InputGateBias)
        {
            throw InvalidArgumentException(descriptorName + ": If CIFG-LSTM is disabled InputGateBias "
                                           "must be present.");
        }
        ValidateTensorNumDimNumElem(m_InputGateBias->GetTensorInfo(), 1,
                                      n_cell, "InputGateBias");
    }

    ValidatePointer(m_ForgetGateBias, "Null pointer check", "ForgetGateBias");
    ValidateTensorNumDimNumElem(m_ForgetGateBias->GetTensorInfo(), 1, n_cell, "ForgetGateBias");

    ValidatePointer(m_CellBias, "Null pointer check", "CellBias");
    ValidateTensorNumDimNumElem(m_CellBias->GetTensorInfo(), 1, n_cell, "CellBias");

    ValidatePointer(m_OutputGateBias, "Null pointer check", "OutputGateBias");
    ValidateTensorNumDimNumElem(m_OutputGateBias->GetTensorInfo(), 1, n_cell, "OutputGateBias");

    if (m_ProjectionWeights)
    {
        ValidateTensorNumDimNumElem(m_ProjectionWeights->GetTensorInfo(), 2,
                                      (n_cell * n_output), "ProjectionWeights");
    }
    if (m_ProjectionBias)
    {
        ValidateTensorNumDimNumElem(m_ProjectionBias->GetTensorInfo(), 1, n_output, "ProjectionBias");
    }

    // Making sure the projection tensors are consistent:
    // 1) If projection weight is not present, then projection bias should not be
    // present.
    // 2) If projection weight is present, then projection bias is optional.
    bool projecton_tensors_consistent = ((!m_ProjectionWeights && !m_ProjectionBias &&
                                        !m_Parameters.m_ProjectionEnabled)
                                        || (m_ProjectionWeights && !m_ProjectionBias &&
                                        m_Parameters.m_ProjectionEnabled)
                                        || (m_ProjectionWeights && m_ProjectionBias &&
                                        m_Parameters.m_ProjectionEnabled));
    if (!projecton_tensors_consistent)
    {
        throw InvalidArgumentException(descriptorName + ": Projection tensors are inconsistent.");
    }

    // The four layer normalization weights either all have values or none of them have values. Additionally, if
    // CIFG is used, input layer normalization weights tensor is omitted and the other layer normalization weights
    // either all have values or none of them have values. Layer normalization is used when the values of all the
    // layer normalization weights are present
    if (m_InputLayerNormWeights)
    {
        ValidateTensorNumDimNumElem(m_InputLayerNormWeights->GetTensorInfo(), 1, n_cell, "InputLayerNormWeights");
    }
    if (m_ForgetLayerNormWeights)
    {
        ValidateTensorNumDimNumElem(m_ForgetLayerNormWeights->GetTensorInfo(), 1, n_cell, "ForgetLayerNormWeights");
    }
    if (m_CellLayerNormWeights)
    {
        ValidateTensorNumDimNumElem(m_CellLayerNormWeights->GetTensorInfo(), 1, n_cell, "CellLayerNormWeights");
    }
    if (m_OutputLayerNormWeights)
    {
        ValidateTensorNumDimNumElem(m_OutputLayerNormWeights->GetTensorInfo(), 1, n_cell, "OutputLayerNormWeights");
    }

    if (m_Parameters.m_LayerNormEnabled)
    {
        if (!m_Parameters.m_CifgEnabled)
        {
            if (!m_InputLayerNormWeights)
            {
                throw InvalidArgumentException(descriptorName + ": Layer normalisation is enabled and CIFG-LSTM is "
                                               "disabled but InputLayerNormWeights are not present");
            }
            ValidateTensorNumDimNumElem(m_InputLayerNormWeights->GetTensorInfo(),
                                          1, n_cell, "InputLayerNormWeights");
        }
        else if (m_InputLayerNormWeights)
        {
            throw InvalidArgumentException(descriptorName + ":InputLayerNormWeights are present while CIFG is "
                                           "enabled");
        }

        ValidatePointer(m_ForgetLayerNormWeights, "Null pointer check layer normalisation enabled",
                        "ForgetLayerNormWeights");
        ValidateTensorNumDimNumElem(m_ForgetLayerNormWeights->GetTensorInfo(), 1, n_cell, "ForgetLayerNormWeights");

        ValidatePointer(m_OutputLayerNormWeights, "Null pointer check layer normalisation enabled",
                        "OutputLayerNormWeights");
        ValidateTensorNumDimNumElem(m_OutputLayerNormWeights->GetTensorInfo(), 1, n_cell, "OutputLayerNormWeights");

        ValidatePointer(m_CellLayerNormWeights, "Null pointer check layer normalisation enabled",
                        "CellLayerNormWeights");
        ValidateTensorNumDimNumElem(m_CellLayerNormWeights->GetTensorInfo(), 1, n_cell, "CellLayerNormWeights");
    }
    else if (m_InputLayerNormWeights || m_ForgetLayerNormWeights || m_OutputLayerNormWeights || m_CellLayerNormWeights)
    {
        throw InvalidArgumentException(descriptorName + ": Layer normalisation is disabled but one or more layer "
                                       "normalisation weights are present.");
    }
}


} // namespace armnn