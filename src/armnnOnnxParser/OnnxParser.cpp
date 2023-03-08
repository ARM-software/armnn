//
// Copyright © 2017,2022-2023 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//
#include "OnnxParser.hpp"

#include "armnnOnnxParser/Version.hpp"

#include <armnn/Descriptors.hpp>
#include <armnn/utility/Assert.hpp>
#include <armnn/utility/NumericCast.hpp>
#include <ParserHelper.hpp>
#include <VerificationHelpers.hpp>

#include <fmt/format.h>

#include <google/protobuf/text_format.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>

#include <iostream>
#include <numeric>
#include <armnnUtils/Permute.hpp>

using namespace armnn;

namespace armnnOnnxParser
{

IOnnxParser::IOnnxParser() : pOnnxParserImpl(new OnnxParserImpl()) {}

IOnnxParser::~IOnnxParser() = default;

IOnnxParser* IOnnxParser::CreateRaw()
{
    return new IOnnxParser();
}

IOnnxParserPtr IOnnxParser::Create()
{
    return IOnnxParserPtr(CreateRaw(), &IOnnxParser::Destroy);
}

void IOnnxParser::Destroy(IOnnxParser* parser)
{
    delete parser;
}

armnn::INetworkPtr IOnnxParser::CreateNetworkFromBinaryFile(const char* graphFile)
{
    return pOnnxParserImpl->CreateNetworkFromBinaryFile(graphFile);
}

armnn::INetworkPtr IOnnxParser::CreateNetworkFromBinary(const std::vector<uint8_t>& binaryContent)
{
    return pOnnxParserImpl->CreateNetworkFromBinary(binaryContent);
}

armnn::INetworkPtr IOnnxParser::CreateNetworkFromBinary(const std::vector<uint8_t>& binaryContent,
                                                        const std::map<std::string, armnn::TensorShape>& inputShapes)
{
    return pOnnxParserImpl->CreateNetworkFromBinary(binaryContent, inputShapes);
}

armnn::INetworkPtr IOnnxParser::CreateNetworkFromTextFile(const char* graphFile)
{
    return pOnnxParserImpl->CreateNetworkFromTextFile(graphFile);
}

armnn::INetworkPtr IOnnxParser::CreateNetworkFromString(const std::string& protoText)
{
    return pOnnxParserImpl->CreateNetworkFromString(protoText);
}

armnn::INetworkPtr IOnnxParser::CreateNetworkFromBinaryFile(
    const char* graphFile,
    const std::map<std::string, armnn::TensorShape>& inputShapes)
{
    return pOnnxParserImpl->CreateNetworkFromBinaryFile(graphFile, inputShapes);
}

armnn::INetworkPtr IOnnxParser::CreateNetworkFromTextFile(const char* graphFile,
                                                          const std::map<std::string, armnn::TensorShape>& inputShapes)
{
    return pOnnxParserImpl->CreateNetworkFromTextFile(graphFile, inputShapes);
}

armnn::INetworkPtr IOnnxParser::CreateNetworkFromString(const std::string& protoText,
                                                        const std::map<std::string, armnn::TensorShape>& inputShapes)
{
    return pOnnxParserImpl->CreateNetworkFromString(protoText, inputShapes);
}

BindingPointInfo IOnnxParser::GetNetworkInputBindingInfo(const std::string& name) const
{
    return pOnnxParserImpl->GetNetworkInputBindingInfo(name);
}

BindingPointInfo IOnnxParser::GetNetworkOutputBindingInfo(const std::string& name) const
{
    return pOnnxParserImpl->GetNetworkOutputBindingInfo(name);
}

namespace
{
void CheckValidDataType(std::initializer_list<onnx::TensorProto::DataType> validInputTypes,
                        const onnx::TensorProto::DataType actualValue,
                        const char* validExpr,
                        std::string nodeName,
                        std::string tensorName,
                        const armnn::CheckLocation& location)
{
    bool isValid = std::any_of(validInputTypes.begin(),
                               validInputTypes.end(),
                               [&actualValue](onnx::TensorProto::DataType x) { return x == actualValue; } );
    if (!isValid)
    {
        throw ParseException(
            fmt::format("Datatype {} is not valid for tensor '{}' of node '{}', not in {{{}}}. {}",
                        onnx::TensorProto::DataType_Name(actualValue),
                        tensorName,
                        nodeName,
                        validExpr,
                        location.AsString()));
    }
}

#define CHECK_VALID_DATATYPE(NODE, TENSOR, ACTUAL, ...) \
CheckValidDataType({__VA_ARGS__}, ACTUAL, #__VA_ARGS__, NODE, TENSOR, CHECK_LOCATION())

using StrTypeListPair = std::pair<const char*, std::initializer_list<onnx::TensorProto::DataType>>;
#define STR_LIST(...) StrTypeListPair(#__VA_ARGS__, {__VA_ARGS__})

template <typename Callable>
void ReadMandatoryNodeAttributeImpl(const onnx::NodeProto& node,
                                    const std::string& attribName,
                                    onnx::AttributeProto::AttributeType expectedType,
                                    Callable callable)
{
  auto attribs = node.attribute();
  int attriNum = 0;
  while (attriNum < node.attribute_size())
  {
      if (attribs.Get(attriNum).name() == attribName)
      {
          if (attribs.Get(attriNum).type() == expectedType)
          {
              callable(attribs.Get(attriNum));
          }
          else
          {
              throw ParseException(fmt::format("Attribute {} of node {} expected to have {} as "
                                               "onnx::AttributeProto::AttributeType, but found {} instead {}",
                                               attribName,
                                               node.name(),
                                               onnx::AttributeProto::AttributeType_Name(expectedType),
                                               onnx::AttributeProto::AttributeType_Name(attribs.Get(attriNum).type()),
                                               CHECK_LOCATION().AsString()));
          }
          break;
      }
      ++attriNum;
  }
  if (attriNum == node.attribute_size())
  {
      throw ParseException(fmt::format("Could not find required attribute {} in node {} {}",
                                       attribName, node.name(), CHECK_LOCATION().AsString()));
  }
}

template <typename Callable>
void ReadOptionalNodeAttributeImpl(const onnx::NodeProto& node,
                                   const std::string& attribName,
                                   onnx::AttributeProto::AttributeType expectedType,
                                   Callable callable)
{
    auto attribs = node.attribute();
    for (int attriNum = 0; attriNum < node.attribute_size(); ++attriNum)
    {
        if (attribs.Get(attriNum).name() == attribName)
        {
            if (attribs.Get(attriNum).type() == expectedType)
            {
                callable(attribs.Get(attriNum));
            }
            else
            {
                throw ParseException(
                    fmt::format("Attribute {} of node {} expected to have {} as onnx::AttributeProto::AttributeType, "
                                "but found {} instead {}",
                                attribName,
                                node.name(),
                                onnx::AttributeProto::AttributeType_Name(expectedType),
                                onnx::AttributeProto::AttributeType_Name(attribs.Get(attriNum).type()),
                                CHECK_LOCATION().AsString()));
            }
        }
    }
}

int ReadMandatoryNodeIntAttribute(const onnx::NodeProto& node,
                                    const std::string& name)
{
    int attribValue = 0;
    ReadMandatoryNodeAttributeImpl(node, name, onnx::AttributeProto::INT,
                                  [&attribValue](const onnx::AttributeProto& attrValue)
                                      {
                                          attribValue = CHECKED_INT32(attrValue.i());
                                      });
    return attribValue;
}

int64_t ReadOptionalNodeInt64Attribute(const onnx::NodeProto& node,
                                       const std::string& name,
                                       const int64_t defaultValue = 0)
{
    int64_t attribValue = defaultValue;
    ReadOptionalNodeAttributeImpl(node, name, onnx::AttributeProto::INT,
                                  [&attribValue](const onnx::AttributeProto& attrValue)
                                      {
                                          attribValue = attrValue.i();
                                      });
    return attribValue;
}

std::vector<uint32_t> ReadMandatoryNodeUint32ListAttribute(const onnx::NodeProto& node,
                                                           const std::string& name)
{
    std::vector<uint32_t> attriList;
    ReadMandatoryNodeAttributeImpl(node, name, onnx::AttributeProto::INTS,
        [&attriList](const onnx::AttributeProto& attrValue)
    {
        for (int attriNum = 0; attriNum < attrValue.ints_size(); ++attriNum)
        {
            attriList.push_back(CHECKED_NON_NEGATIVE(CHECKED_INT32(attrValue.ints().Get(attriNum))));
        }
    });
    return attriList;
}

uint32_t ReadOptionalNodeUint32Attribute(const onnx::NodeProto& node,
                                         const std::string& name,
                                         const uint32_t defaultVal = 0u)
{
    uint32_t attribValue = defaultVal;
    ReadOptionalNodeAttributeImpl(node, name, onnx::AttributeProto::INT,
        [&attribValue](const onnx::AttributeProto& attrValue)
    {
        attribValue = CHECKED_NON_NEGATIVE(CHECKED_INT32((attrValue.i())));
    });
    return attribValue;
}

std::vector<uint32_t> ReadOptionalNodeUint32ListAttribute(const onnx::NodeProto& node,
                                                          const std::string& name)
{
    std::vector<uint32_t> attriList;
    ReadOptionalNodeAttributeImpl(node, name, onnx::AttributeProto::INTS,
        [&attriList](const onnx::AttributeProto& attrValue)
    {
        for (int attriNum = 0; attriNum < attrValue.ints_size(); ++attriNum)
        {
            attriList.push_back(CHECKED_NON_NEGATIVE(CHECKED_INT32(attrValue.ints().Get(attriNum))));
        }
    });

    return attriList;
}

float ReadOptionalNodeFloatAttribute(const onnx::NodeProto& node,
                                     const std::string& name,
                                     const float defaultValue = 0.0f)
{
    float attribValue = defaultValue;
    ReadOptionalNodeAttributeImpl(node, name, onnx::AttributeProto::FLOAT,
        [&attribValue](const onnx::AttributeProto& attrValue)
    {
        attribValue = attrValue.f();
    });
    return attribValue;
}

std::string ReadOptionalNodeStringAttribute(const onnx::NodeProto& node, const std::string& name)
{
    std::string attribValue = "";
    ReadOptionalNodeAttributeImpl(node, name, onnx::AttributeProto::STRING,
        [&attribValue](const onnx::AttributeProto& attrValue)
    {
        attribValue = attrValue.s();
    });
    return attribValue;
}

armnn::TensorInfo ToTensorInfo(const std::string& name, std::vector<unsigned int>& shape, int data_type)
{
    DataType type;
    switch(data_type)
    {
        case onnx::TensorProto::FLOAT:
        {
          type = DataType::Float32;
          break;
        }
        case onnx::TensorProto::INT32:
        case onnx::TensorProto::INT64:
        {
            type = DataType::Signed32;
            break;
        }
        default:
        {
            throw ParseException(
                fmt::format("'{}' is not a currently supported datatype for tensor {}."
                            " Supported dataTypes are FLOAT, INT32 and INT64.  {}",
                            onnx::TensorProto::DataType_Name(static_cast<onnx::TensorProto::DataType>(data_type)),
                            name,
                            CHECK_LOCATION().AsString() ));
        }
    }

    // Scalar Tensor
    if (shape.empty())
    {
        return TensorInfo(TensorShape(Dimensionality::Scalar), type);
    }

    // Dynamic Tensor
    if(std::find(shape.begin(), shape.end(), 0) != shape.end())
    {
        return TensorInfo(TensorShape(Dimensionality::NotSpecified), type);
    }

    return TensorInfo(TensorShape(static_cast<unsigned int>(shape.size()), shape.data()), type);
}

armnn::TensorInfo ToTensorInfo(const onnx::ValueInfoProto& info)
{
  const onnx::TensorShapeProto onnxShape = info.type().tensor_type().shape();
  std::vector<unsigned int> shapeDims;
  for (int i = 0; i < onnxShape.dim_size(); ++i)
  {
      shapeDims.push_back(CHECKED_NON_NEGATIVE(CHECKED_INT32(onnxShape.dim(i).dim_value())));
  }

  return ToTensorInfo(info.name(), shapeDims, info.type().tensor_type().elem_type());
}

armnn::TensorInfo ToTensorInfo(const onnx::TensorProto& tensor)
{
  std::vector<unsigned int> shapeDims;

  for (auto dim: tensor.dims())
  {
      shapeDims.push_back(CHECKED_NON_NEGATIVE(CHECKED_INT32(dim)));
  }

  return ToTensorInfo(tensor.name(), shapeDims, tensor.data_type());
}

std::string TensorInfoAsString(const TensorInfo& info,
                               const std::string& name,
                               const onnx::TensorProto::DataType& type)
{
    const TensorShape shape = info.GetShape();
    std::stringstream ss;
    ss << "tensor '" << name << "' contains "
       << onnx::TensorProto::DataType_Name(type)
       << " and has shape [";

    for (uint32_t i = 0; i < shape.GetNumDimensions() - 1; ++i)
    {
        ss << shape[i] << ", ";
    }
    ss << shape[shape.GetNumDimensions() - 1] << "]";
    return ss.str();
}

void CalcPadding(uint32_t inputSize,
                 uint32_t filterSize,
                 uint32_t stride,
                 uint32_t dilation,
                 uint32_t* paddingFront,
                 uint32_t* paddingBack,
                 bool isUpper)
{
    uint32_t outputSize = (inputSize + stride - 1) / stride;
    uint32_t dilatedSize = filterSize + (dilation - 1) * (filterSize - 1);
    uint32_t temp = (outputSize - 1) * stride + dilatedSize;
    *paddingFront = (temp - inputSize) / 2;
    *paddingBack = *paddingFront;
    if((temp - inputSize) % 2 == 1)
    {
        if (isUpper)
        {
            *paddingBack += 1;
        }
        else
        {
            *paddingFront += 1;
        }
    }
}

TensorInfo ComputeReshapeInfo(const TensorShape& targetShapeTensor,
                              const TensorShape& inShape,
                              const std::string& outName,
                              DataType dataType = DataType::Float32)
{
    std::vector<int> targetDims;
    for(uint i = 0; i < targetShapeTensor.GetNumDimensions(); ++i)
    {
        int val = CHECKED_INT32(targetShapeTensor[i]);
        if(val == 0)
        {
            targetDims.push_back(static_cast<int>(inShape[static_cast<uint>(i)]));
        }
        else
        {
            targetDims.push_back(val);
        }
    }

    std::vector<unsigned int> outDims(targetDims.begin(), targetDims.end());
    const auto stretchDim = std::find(targetDims.begin(), targetDims.end(), -1);
    if (stretchDim != targetDims.end())
    {
        if (std::find(std::next(stretchDim), targetDims.end(), -1) != targetDims.end())
        {
            std::stringstream ss;
            ss << "[ ";
            for(uint i = 0; i < targetDims.size() - 1; ++i)
            {
                ss << targetDims[i] << ", ";
            }
            ss << targetDims[targetDims.size() - 1] << " ]";

            throw ParseException(
                fmt::format("Error during creation of reshaped tensor '{}'. At most one component of shape can be "
                            " -1 and here, shape is {} {}",
                            outName,
                            ss.str(),
                            CHECK_LOCATION().AsString()));
        }

        auto targetNumElements = armnn::numeric_cast<unsigned int>(std::accumulate(targetDims.begin(), targetDims.end(),
            -1, std::multiplies<int32_t>()));
        auto stretchIndex = static_cast<size_t>(std::distance(targetDims.begin(), stretchDim));
        outDims[stretchIndex] = inShape.GetNumElements() / targetNumElements;
    }
    TensorShape outShape = TensorShape{static_cast<unsigned int>(outDims.size()), outDims.data()};
    return TensorInfo(outShape, dataType);
}

} //namespace

const std::map<std::string, OnnxParserImpl::OperationParsingFunction> OnnxParserImpl::m_ParserFunctions = {
    { "BatchNormalization",    &OnnxParserImpl::ParseBatchNormalization},
    { "GlobalAveragePool",     &OnnxParserImpl::ParseGlobalAveragePool},
    { "AveragePool",           &OnnxParserImpl::ParseAveragePool },
    { "Clip",                  &OnnxParserImpl::ParseClip },
    { "Constant",              &OnnxParserImpl::ParseConstant },
    { "MaxPool",               &OnnxParserImpl::ParseMaxPool },
    { "Reshape",               &OnnxParserImpl::ParseReshape },
    { "Sigmoid",               &OnnxParserImpl::ParseSigmoid },
    { "Tanh",                  &OnnxParserImpl::ParseTanh },
    { "Relu",                  &OnnxParserImpl::ParseRelu },
    { "LeakyRelu",             &OnnxParserImpl::ParseLeakyRelu },
    { "Conv",                  &OnnxParserImpl::ParseConv },
    { "Add",                   &OnnxParserImpl::ParseAdd },
    { "Flatten",               &OnnxParserImpl::ParseFlatten },
    { "Shape",                 &OnnxParserImpl::ParseShape },
    { "Gather",                &OnnxParserImpl::ParseGather },
    { "Unsqueeze",             &OnnxParserImpl::ParseUnsqueeze },
    { "Concat",                &OnnxParserImpl::ParseConcat },
    { "Gemm",                  &OnnxParserImpl::ParseGemm }
};

template<typename TypePair, typename Location>
void OnnxParserImpl::ValidateInputs(const onnx::NodeProto& node,
                                TypePair validInputs,
                                const Location& location)
{
    for(auto input : node.input())
    {
        CheckValidDataType(validInputs.second,
                           m_TensorsInfo[input].m_dtype,
                           validInputs.first,
                           node.name(),
                           input,
                           location);
    }
}

#define VALID_INPUTS(NODE, VALID_INPUTS) \
    OnnxParserImpl::ValidateInputs(NODE, \
                               VALID_INPUTS, \
                               CHECK_LOCATION())

std::vector<TensorInfo> OnnxParserImpl::ComputeOutputInfo(std::vector<std::string> outNames,
                                                          const IConnectableLayer* layer,
                                                          std::vector<TensorShape> inputShapes,
                                                          const onnx::TensorProto::DataType& dataType)
{
    ARMNN_ASSERT(! outNames.empty());
    bool needCompute = std::any_of(outNames.begin(),
                                   outNames.end(),
                                   [this](std::string name)
                                   {
                                       return (m_TensorsInfo.count(name) == 0 ||
                                               m_TensorsInfo[name].m_info == nullptr ||
                                               m_TensorsInfo[name].m_info->GetShape().GetDimensionality() ==
                                               Dimensionality::NotSpecified);
                                   });
    std::vector<TensorInfo> outInfo;
    //if the output info(s) are not here, we need to compute them
    std::vector<TensorShape> inferredShapes;
    DataType armnnType = DataType::Float32;
    if(needCompute) {
        inferredShapes = layer->InferOutputShapes(inputShapes);
        ARMNN_ASSERT(inferredShapes.size() == outNames.size());
        switch (dataType) {
            case onnx::TensorProto::FLOAT: {
                armnnType = DataType::Float32;
                break;
            }
            case onnx::TensorProto::INT32:
            case onnx::TensorProto::INT64: {
                armnnType = DataType::Signed32;
                break;
            }
            default: {
                throw ParseException(
                    fmt::format("'{}' is not a currently supported datatype for {}."
                                " Supported dataTypes are FLOAT, INT32 and INT64.  {}",
                                onnx::TensorProto::DataType_Name(static_cast<onnx::TensorProto::DataType>(dataType)),
                                layer->GetName(),
                                CHECK_LOCATION().AsString()));
            }
        }
    }
    for (uint i = 0; i < outNames.size(); ++i)
    {
        if(needCompute)
        {
            m_TensorsInfo[outNames[i]] = OnnxTensor();
            m_TensorsInfo[outNames[i]].m_info = std::make_unique<TensorInfo>(
                TensorInfo(inferredShapes[i], armnnType));
            m_TensorsInfo[outNames[i]].m_dtype = dataType;
        }
        outInfo.push_back(*m_TensorsInfo[outNames[i]].m_info);
    }
    return outInfo;
}

OnnxParserImpl::OnnxParserImpl()
    : m_Network(nullptr, nullptr)
{
}

void OnnxParserImpl::ResetParser()
{
    m_Network = armnn::INetworkPtr(nullptr, nullptr);
    m_Graph = nullptr;
    m_InputInfos.clear();
    m_OutputInfos.clear();
}

void OnnxParserImpl::Cleanup()
{
    m_TensorConnections.clear();
    m_TensorsInfo.clear();
    m_OutputsMap.clear();
    m_OutputsFusedAndUsed.clear();
    m_InputShapes.clear();
}

template<typename T>
std::pair<armnn::ConstTensor, std::unique_ptr<T[]>>
CreateConstTensorImpl(const T* bufferPtr,
                      armnn::TensorInfo& tensorInfo,
                      const armnn::Optional<armnn::PermutationVector&> permutationVector)
{
    ARMNN_ASSERT_MSG(bufferPtr != nullptr, fmt::format("Buffer for permutation is null").c_str());

    std::unique_ptr<T[]> data(new T[tensorInfo.GetNumElements()]);

    if (permutationVector.has_value() && permutationVector.value().GetSize() > 0)
    {
        tensorInfo = armnnUtils::Permuted(tensorInfo, permutationVector.value());
        armnnUtils::Permute(tensorInfo.GetShape(), permutationVector.value(),
                            reinterpret_cast<const T*>(bufferPtr), data.get(), sizeof(T));
    }
    else
    {
        ::memcpy(data.get(), bufferPtr, tensorInfo.GetNumBytes());
    }

    return std::make_pair(ConstTensor(tensorInfo, data.get()), std::move(data));
}

std::pair<ConstTensor, std::unique_ptr<float[]>>
OnnxParserImpl::CreateConstTensor(const std::string name,
                                  armnn::Optional<armnn::PermutationVector&> permutationVector)
{
    TensorInfo tensorInfo = *m_TensorsInfo[name].m_info;
    onnx::TensorProto onnxTensor = *m_TensorsInfo[name].m_tensor;

    //ONNX can have Float16 and double constant nodes but ArmNN only supports float32
    CHECK_VALID_DATATYPE(name, onnxTensor.name(),
                         static_cast<onnx::TensorProto::DataType>(onnxTensor.data_type()), onnx::TensorProto::FLOAT);

    // Makes sure IsConstant flag is set.
    tensorInfo.SetConstant();

    // Const tensors requires at least a list of values
    if (tensorInfo.GetNumElements() == 0)
    {
        throw ParseException(fmt::format("No tensor data found for Const tensor '{}' {}",
                                         name,
                                         CHECK_LOCATION().AsString()));
    }

    auto srcData = onnxTensor.float_data().data();
    // Copy the value list entries into the destination
    if (!onnxTensor.has_raw_data())
    {
        if(tensorInfo.GetNumElements() != static_cast<uint>(onnxTensor.float_data_size()))
        {
            throw ParseException(
                fmt::format("The number of data provided ({}) does not match the tensor '{}' number of "
                            "elements ({}) {}",
                            onnxTensor.float_data_size(),
                            name,
                            tensorInfo.GetNumElements(),
                            CHECK_LOCATION().AsString()));
        }
        return CreateConstTensorImpl<float>(srcData, tensorInfo, permutationVector);
    }
    else
    {
        return CreateConstTensorImpl<float>(reinterpret_cast<const float*>(onnxTensor.raw_data().c_str()),
                                            tensorInfo,
                                            permutationVector);
    }
}

std::pair<ConstTensor, std::unique_ptr<int32_t[]>>
OnnxParserImpl::CreateInt64ConstTensor(const std::string name,
                                       armnn::Optional<armnn::PermutationVector&> permutationVector)
{
    TensorInfo tensorInfo = *m_TensorsInfo[name].m_info;
    onnx::TensorProto onnxTensor = *m_TensorsInfo[name].m_tensor;

    CHECK_VALID_DATATYPE(name, onnxTensor.name(),
                         static_cast<onnx::TensorProto::DataType>(onnxTensor.data_type()), onnx::TensorProto::INT64);

    // Makes sure IsConstant flag is set.
    tensorInfo.SetConstant();
    uint numElements = tensorInfo.GetNumElements();

    // Const tensors requires at least a list of values
    if (numElements == 0)
    {
        throw ParseException(fmt::format("No tensor data found for Const tensor '{}' {}",
                                         name,
                                         CHECK_LOCATION().AsString()));
    }

    // Copy the value list entries into the destination
    if (!onnxTensor.has_raw_data())
    {
        auto srcData = onnxTensor.int64_data().data();
        if(numElements != static_cast<uint>(onnxTensor.int64_data_size()))
        {
            throw ParseException(
                fmt::format("The number of data provided ({}) does not match the tensor '{}' number of "
                            "elements ({}) {}",
                            onnxTensor.int64_data_size(),
                            name,
                            tensorInfo.GetNumElements(),
                            CHECK_LOCATION().AsString()));
        }

        std::vector<int32_t> int32Data;
        for(uint i = 0; i < numElements; i++)
        {
            int32_t int32Value = CHECKED_INT32(srcData[i]);
            int32Data.push_back(int32Value);
        }

        return CreateConstTensorImpl<int32_t>(int32Data.data(), tensorInfo, permutationVector);
    }
    else
    {
        auto srcData = reinterpret_cast<const int64_t*>(onnxTensor.raw_data().c_str());
        std::vector<int32_t> int32Data;
        for(uint i = 0; i < numElements; i++)
        {
            int32_t int32Value = CHECKED_INT32(srcData[i]);
            int32Data.push_back(int32Value);
        }
        return CreateConstTensorImpl<int32_t>(int32Data.data(), tensorInfo, permutationVector);
    }
}

ModelPtr OnnxParserImpl::LoadModelFromTextFile(const char* graphFile)
{
    FILE* fd = fopen(graphFile, "r");

    if (fd == nullptr)
    {
        throw FileNotFoundException(fmt::format("Invalid (null) filename {}", CHECK_LOCATION().AsString()));
    }

    // Parse the file into a message
    ModelPtr     modelProto = std::make_unique<onnx::ModelProto>();
    using google::protobuf::io::FileInputStream;
    std::unique_ptr<FileInputStream> input = std::make_unique<FileInputStream>(fileno(fd));
    bool                 success = google::protobuf::TextFormat::Parse(input.get(), modelProto.get());
    fclose(fd);

    if (!success)
    {
        std::stringstream error;
        error << "Failed to parse graph file";
        throw ParseException(fmt::format("{} {}", error.str(), CHECK_LOCATION().AsString()));
    }
    return modelProto;
}

INetworkPtr OnnxParserImpl::CreateNetworkFromTextFile(const char* graphFile)
{
    ResetParser();
    ModelPtr modelProto = LoadModelFromTextFile(graphFile);
    return CreateNetworkFromModel(*modelProto);
}

INetworkPtr OnnxParserImpl::CreateNetworkFromTextFile(const char* graphFile,
                                                      const std::map<std::string, armnn::TensorShape>& inputShapes)
{
    ResetParser();
    m_InputShapes = inputShapes;
    ModelPtr modelProto = LoadModelFromTextFile(graphFile);
    return CreateNetworkFromModel(*modelProto);
}

INetworkPtr OnnxParserImpl::CreateNetworkFromBinary(const std::vector<uint8_t>& binaryContent)
{
    ResetParser();
    ModelPtr modelProto = LoadModelFromBinary(binaryContent);
    return CreateNetworkFromModel(*modelProto);
}

INetworkPtr OnnxParserImpl::CreateNetworkFromBinary(const std::vector<uint8_t>& binaryContent,
                                                    const std::map<std::string, armnn::TensorShape>& inputShapes)
{
    ResetParser();
    m_InputShapes = inputShapes;
    ModelPtr modelProto = LoadModelFromBinary(binaryContent);
    return CreateNetworkFromModel(*modelProto);
}

ModelPtr OnnxParserImpl::LoadModelFromBinary(const std::vector<uint8_t>& binaryContent)
{
    if (binaryContent.size() == 0)
    {
        throw ParseException(fmt::format("Missing binary content", CHECK_LOCATION().AsString()));
    }
    // Parse the file into a message
    ModelPtr modelProto = std::make_unique<onnx::ModelProto>();

    google::protobuf::io::CodedInputStream codedStream(binaryContent.data(), static_cast<int>(binaryContent.size()));
    codedStream.SetTotalBytesLimit(INT_MAX);
    bool success = modelProto.get()->ParseFromCodedStream(&codedStream);

    if (!success)
    {
        std::stringstream error;
        error << "Failed to parse graph";
        throw ParseException(fmt::format("{} {}", error.str(), CHECK_LOCATION().AsString()));
    }
    return modelProto;
}

ModelPtr OnnxParserImpl::LoadModelFromBinaryFile(const char* graphFile)
{
    FILE* fd = fopen(graphFile, "rb");

    if (fd == nullptr)
    {
        throw FileNotFoundException(fmt::format("Invalid (null) filename {}", CHECK_LOCATION().AsString()));
    }

    // Parse the file into a message
    ModelPtr modelProto = std::make_unique<onnx::ModelProto>();

    google::protobuf::io::FileInputStream  inStream(fileno(fd));
    google::protobuf::io::CodedInputStream codedStream(&inStream);
    codedStream.SetTotalBytesLimit(INT_MAX);
    bool success = modelProto.get()->ParseFromCodedStream(&codedStream);
    fclose(fd);

    if (!success)
    {
        std::stringstream error;
        error << "Failed to parse graph file";
        throw ParseException(fmt::format("{} {}", error.str(), CHECK_LOCATION().AsString()));
    }
    return modelProto;

}

INetworkPtr OnnxParserImpl::CreateNetworkFromBinaryFile(const char* graphFile)
{
    ResetParser();
    ModelPtr modelProto = LoadModelFromBinaryFile(graphFile);
    return CreateNetworkFromModel(*modelProto);
}

INetworkPtr OnnxParserImpl::CreateNetworkFromBinaryFile(const char* graphFile,
                                                        const std::map<std::string, armnn::TensorShape>& inputShapes)
{
    ResetParser();
    m_InputShapes = inputShapes;
    ModelPtr modelProto = LoadModelFromBinaryFile(graphFile);
    return CreateNetworkFromModel(*modelProto);
}

ModelPtr OnnxParserImpl::LoadModelFromString(const std::string& protoText)
{
    if (protoText == "")
    {
        throw InvalidArgumentException(fmt::format("Invalid (empty) string for model parameter {}",
                                                   CHECK_LOCATION().AsString()));
    }
    // Parse the string into a message
    ModelPtr modelProto = std::make_unique<onnx::ModelProto>();
    bool success = google::protobuf::TextFormat::ParseFromString(protoText, modelProto.get());
    if (!success)
    {
        std::stringstream error;
        error << "Failed to parse graph file";
        throw ParseException(fmt::format("{} {}", error.str(), CHECK_LOCATION().AsString()));
    }
    return modelProto;
}

INetworkPtr OnnxParserImpl::CreateNetworkFromString(const std::string& protoText)
{
    ResetParser();
    ModelPtr modelProto = LoadModelFromString(protoText);
    return CreateNetworkFromModel(*modelProto);
}

INetworkPtr OnnxParserImpl::CreateNetworkFromString(const std::string& protoText,
                                                    const std::map<std::string, armnn::TensorShape>& inputShapes)
{
    ResetParser();
    m_InputShapes = inputShapes;
    ModelPtr modelProto = LoadModelFromString(protoText);
    return CreateNetworkFromModel(*modelProto);
}

INetworkPtr OnnxParserImpl::CreateNetworkFromModel(onnx::ModelProto& model)
{
    m_Network = INetwork::Create();
    try
    {
        m_Graph = std::make_unique<onnx::GraphProto>(*model.mutable_graph());
        LoadGraph();
    }
    catch (const ParseException& e)
    {
        Cleanup();
        throw e;
    }
    Cleanup();
    return std::move(m_Network);
}

void OnnxParserImpl::LoadGraph()
{
    ARMNN_ASSERT(m_Graph.get() != nullptr);

    //Fill m_TensorsInfo with the shapes and value of every tensor
    SetupInfo(m_Graph->mutable_output());
    SetupInfo(m_Graph->mutable_input());
    SetupInfo(m_Graph->mutable_value_info());

    for (auto tensor : m_Graph->initializer())
    {
        m_TensorsInfo[tensor.name()].m_tensor = std::make_unique<const onnx::TensorProto>(tensor);
        m_TensorsInfo[tensor.name()].m_info = std::make_unique<TensorInfo>(ToTensorInfo(tensor));
        m_TensorsInfo[tensor.name()].m_dtype =
            static_cast<onnx::TensorProto::DataType>(tensor.data_type());
    }

    SetupInputLayers();
    SetupOutputLayers();

    //Detect FullyConnected layers with bias and update the FusedAndUsed map acccordingly
    DetectFullyConnected();

    //Parsing the graph
    for(size_t nodeIndex = 0; nodeIndex < static_cast<size_t>(m_Graph->node_size()); nodeIndex++)
    {
        auto node = m_Graph->node(static_cast<int>(nodeIndex));
        const std::string& operation = node.op_type();

        // check which layers we handled already (add and matmul fused as FC)
        if (operation == "MatMul" )
        {
            if(m_OutputsFusedAndUsed[nodeIndex].inputForNodes != m_OutputsFusedAndUsed[nodeIndex].fusedWithNodes.size())
            {
                //Node which can not be fused as a FullyConnected layer (used in layers as a simple matmul output)
                AddFullyConnected(node);
            }
        }
        else if (!(m_OutputsFusedAndUsed[nodeIndex].fusedWithNodes.empty()) && operation == "Add")
        {
            int matmulIndex = static_cast<int> (m_OutputsFusedAndUsed[nodeIndex].fusedWithNodes[0]);
            AddFullyConnected(m_Graph->node(matmulIndex), &node);
        }
        else if (m_OutputsFusedAndUsed[nodeIndex].fusedWithNodes.empty()) //node is not part of a fused layer
        {
            auto it = m_ParserFunctions.find(operation);
            if (it != m_ParserFunctions.end())
            {
                auto func = it->second;
                (this->*func)(node);
            }
            else
            {
                throw ParseException(fmt::format("Unsupported operation {} for node '{}' {}",
                                                 operation,
                                                 node.name(),
                                                 CHECK_LOCATION().AsString()));
            }
        }
    }

    //Making the connections between outputs and inputs of each layers
    for (const auto& tensorCon : m_TensorConnections)
    {
        if (tensorCon.second.outputSlot != nullptr)
        {
            for (size_t inputSlotIdx = 0; inputSlotIdx < tensorCon.second.inputSlots.size(); ++inputSlotIdx)
            {
                tensorCon.second.outputSlot->Connect(*(tensorCon.second.inputSlots[inputSlotIdx]));
            }
        }
    }

    // Get output info.
    for(int outputIndex = 0; outputIndex < m_Graph->output_size(); ++outputIndex)
    {
        auto output = m_Graph->output(outputIndex);
        m_OutputInfos[output.name()] = *m_TensorsInfo[output.name()].m_info;
    }
}

void OnnxParserImpl::SetupInfo(const google::protobuf::RepeatedPtrField<onnx::ValueInfoProto >* list)
{
    for (auto tensor : *list)
    {
        m_TensorsInfo[tensor.name()] = OnnxTensor();
        m_TensorsInfo[tensor.name()].m_info = std::make_unique<TensorInfo>(ToTensorInfo(tensor));
        m_TensorsInfo[tensor.name()].m_dtype =
            static_cast<onnx::TensorProto::DataType>(tensor.type().tensor_type().elem_type());
    }
}

void OnnxParserImpl::DetectFullyConnected()
{
    m_OutputsFusedAndUsed = std::vector<UsageSummary> (static_cast<size_t>(m_Graph->node_size()), UsageSummary());
    auto matmulAndConstant = [&](const std::string& constInput,
                                 const std::string& matmulInput,
                                 int& nodeIndex)
    {
        auto matmulIt = m_OutputsMap.find(matmulInput);
        if(matmulIt != m_OutputsMap.end()  && matmulIt->second.first->op_type() == "MatMul"
            && m_TensorsInfo[constInput].isConstant())
        {
            nodeIndex = matmulIt->second.second;
            return true;
        }
        return false;
    };

    for(int nodeIndex = 0; nodeIndex < m_Graph->node_size(); nodeIndex++)
    {
        const onnx::NodeProto* node = &m_Graph->node(nodeIndex);
        for (const std::string& output : node->output())
        {
            m_OutputsMap[output] = std::make_pair(node, nodeIndex);
        }

        for (const std::string& input : node->input()) //count how many time a node is used as input
        {
            auto matmulIt = m_OutputsMap.find(input);
            if(matmulIt != m_OutputsMap.end()){
                ++m_OutputsFusedAndUsed[static_cast<size_t>(matmulIt->second.second)].inputForNodes; //node used
            }
        }

        if (node->op_type() == "Add")
        {
            int matmulIndex = 0;
            if (matmulAndConstant(node->input(0), node->input(1), matmulIndex) ||
                matmulAndConstant(node->input(1), node->input(0), matmulIndex))
            {
                //matmul and add were fused
                m_OutputsFusedAndUsed[static_cast<size_t>(matmulIndex)].fusedWithNodes
                                                                       .push_back(static_cast<size_t>(nodeIndex));

                m_OutputsFusedAndUsed[static_cast<size_t>(nodeIndex)].fusedWithNodes
                                                                     .push_back(static_cast<size_t>(matmulIndex));
            }
        }
    }

    for (auto output: m_Graph->output()) { //Add usages as output of the graph in count of usages
        auto matmulIt = m_OutputsMap.find(output.name());
        if(matmulIt != m_OutputsMap.end()){
            ++m_OutputsFusedAndUsed[static_cast<size_t>(matmulIt->second.second)].inputForNodes;
        }
    }
}

template<typename Location>
void OnnxParserImpl::GetInputAndParam(const onnx::NodeProto& node,
                                      std::string* inputName,
                                      std::string* constName,
                                      const Location& location)
{
    int cstIndex;
    if (m_TensorsInfo[node.input(0)].isConstant())
    {
        cstIndex = 0;
    }
    else if (m_TensorsInfo[node.input(1)].isConstant())
    {
        cstIndex = 1;
    }
    else
    {
        throw ParseException(fmt::format("One of the input tensors ('{}' or '{}') should be constant in node '{}' {}",
                                         node.input(0),
                                         node.input(1),
                                         node.name(),
                                         location.AsString()));
    }
    if(constName)
    {
        *constName = node.input(cstIndex);
    }
    if(inputName)
    {
        *inputName = node.input(!cstIndex);
    }
}

template<typename Location>
void OnnxParserImpl::To1DTensor(const std::string& name, const Location& location)
{
    TensorShape shape = m_TensorsInfo[name].m_info->GetShape();
    std::vector<uint32_t> newShape;
    for(uint i = 0; i < shape.GetNumDimensions() - 1; ++i)
    {
        if(shape[i] != 1)
        {
            throw ParseException(
                fmt::format("Only tensors with shape [1, ..., 1, X] can be converted to 1D and {} {}",
                            TensorInfoAsString(*m_TensorsInfo[name].m_info, name, m_TensorsInfo[name].m_dtype),
                            location.AsString()));
        }
    }
    newShape.push_back(shape[shape.GetNumDimensions() - 1]);

    m_TensorsInfo[name].m_info->SetShape(TensorShape(static_cast<unsigned int>(newShape.size()), newShape.data()));
}

void OnnxParserImpl::AddConvLayerWithDepthwiseConv(const onnx::NodeProto& node, const Convolution2dDescriptor& convDesc)
{
    ARMNN_ASSERT(node.op_type() == "Conv");

    DepthwiseConvolution2dDescriptor desc;
    desc.m_PadLeft      = convDesc.m_PadLeft;
    desc.m_PadRight     = convDesc.m_PadRight;
    desc.m_PadTop       = convDesc.m_PadTop;
    desc.m_PadBottom    = convDesc.m_PadBottom;
    desc.m_StrideX      = convDesc.m_StrideX;
    desc.m_StrideY      = convDesc.m_StrideY;
    desc.m_BiasEnabled  = convDesc.m_BiasEnabled;

    armnn::IConnectableLayer* layer = m_Network->AddDepthwiseConvolution2dLayer(desc, node.name().c_str());
    std::string permuteStr = "permute_" + node.input(1);
    std::vector<std::string> tensorIndexes= {node.input(0), permuteStr};

    auto weightTensor = CreateConstTensor(node.input(1));
    IConnectableLayer* weightsLayer = m_Network->AddConstantLayer(weightTensor.first);

    // weights come in as [O,1,H,W] from ONNX and need to be converted to ArmNNs depthwise weights layout [1,H,W,O]
    armnn::PermutationVector perVec {3, 0, 1, 2};
    TensorInfo weightsPermuted = armnnUtils::Permuted(weightTensor.first.GetInfo(), perVec);

    // Inserts NewLayer so layers don't need to be re-sorted.
    IConnectableLayer* permuteLayer = m_Network->AddPermuteLayer(PermuteDescriptor(perVec),
                                                                 "permute_layer");
    permuteLayer->GetOutputSlot(0).SetTensorInfo(weightsPermuted);
    permuteLayer->GetOutputSlot(0).Connect(layer->GetInputSlot(1u));

    weightsLayer->GetOutputSlot(0).SetTensorInfo(weightTensor.first.GetInfo());
    weightsLayer->GetOutputSlot(0).Connect(permuteLayer->GetInputSlot(0u));

    if (node.input_size() == 3)
    {
        if(!m_TensorsInfo[node.input(2)].isConstant())
        {
            throw ParseException(fmt::format("Bias '{}' should be constant in Conv layer '{}' {}",
                                             node.input(2),
                                             node.name(),
                                             CHECK_LOCATION().AsString()));
        }

        desc.m_BiasEnabled = true;
        auto biasTensor = CreateConstTensor(node.input(2));
        tensorIndexes.emplace_back(node.input(2));

        IConnectableLayer* biasLayer = m_Network->AddConstantLayer(biasTensor.first);
        biasLayer->GetOutputSlot(0).SetTensorInfo(biasTensor.first.GetInfo());
        biasLayer->GetOutputSlot(0).Connect(layer->GetInputSlot(2u));
    }

    ARMNN_ASSERT(layer != nullptr);

    auto outputInfo = ComputeOutputInfo({ node.output(0) }, layer,
                                        { m_TensorsInfo[node.input(0)].m_info->GetShape(),
                                          weightsPermuted.GetShape() });

    layer->GetOutputSlot(0).SetTensorInfo(outputInfo[0]);

    // register the input connection slots for the layer, connections are made after all layers have been created
    // only the tensors for the inputs are relevant, exclude the const tensors
    RegisterInputSlots(layer, tensorIndexes);

    // register the output connection slots for the layer, connections are made after all layers have been created
    RegisterOutputSlots(layer, {node.output(0)});
}

void OnnxParserImpl::AddFullyConnected(const onnx::NodeProto& matmulNode, const onnx::NodeProto* addNode)
{
    // find matmul inputs
    std::string inputName;
    std::string weightName;
    std::string biasName;
    std::string outputName;
    CHECK_VALID_SIZE(static_cast<size_t>(matmulNode.input_size()), 2);
    CHECK_VALID_SIZE(static_cast<size_t>(matmulNode.output_size()), 1);
    VALID_INPUTS(matmulNode, STR_LIST(onnx::TensorProto::FLOAT));

    GetInputAndParam(matmulNode, &inputName, &weightName, CHECK_LOCATION());

    TensorInfo inputInfo = *m_TensorsInfo[inputName].m_info;
    TensorInfo weightInfo = *m_TensorsInfo[weightName].m_info;
    TensorInfo biasInfo;

    std::vector<std::string> inputNames;

    FullyConnectedDescriptor desc;
    desc.m_BiasEnabled = addNode != nullptr;

    IConnectableLayer* layer = nullptr;
    if(desc.m_BiasEnabled)
    {
        // find bias const
        CHECK_VALID_SIZE(static_cast<size_t>(addNode->input_size()), 2);
        CHECK_VALID_SIZE(static_cast<size_t>(addNode->output_size()), 1);
        VALID_INPUTS(*addNode, STR_LIST(onnx::TensorProto::FLOAT));

        GetInputAndParam(*addNode, nullptr, &biasName, CHECK_LOCATION());

        //Output shape is [1, weights[1]] and 1d vec in ONNX can be [1,X] so we convert biases to "armnn" 1D
        To1DTensor(biasName, CHECK_LOCATION());
        biasInfo = *m_TensorsInfo[biasName].m_info;

        if (weightInfo.GetShape()[1] != biasInfo.GetShape()[0])
        {
            throw ParseException(
                fmt::format("Shape of weights '{}' and bias of following Add node '{}' do not match : {}"
                            " and {} ( /!\\ bias should be a 1D tensor) {}",
                            weightName,
                            addNode->name(),
                            TensorInfoAsString(*m_TensorsInfo[weightName].m_info, weightName,
                                               m_TensorsInfo[weightName].m_dtype),
                            TensorInfoAsString(*m_TensorsInfo[biasName].m_info, biasName,
                                               m_TensorsInfo[biasName].m_dtype ),
                            CHECK_LOCATION().AsString()));
        }

        inputNames = { inputName, weightName, biasName };
        outputName = addNode->output(0);
    }
    else
    {
        inputNames = { inputName, weightName };
        outputName = matmulNode.output(0);
    }

    // Just add a FullyConnected layer, weights and biases are handled as inputs now.
    layer = m_Network->AddFullyConnectedLayer(desc, matmulNode.name().c_str());
    ARMNN_ASSERT(layer != nullptr);

    if (inputInfo.GetNumDimensions() > 2)
    {
        // Add reshape to flatten to 2D [batch_size, input_size],
        // where "input_size" corresponds to the number of inputs to the layer,
        // matching the second dimension of weights,
        // and "batch_size" is calculated by dividing the number of elements by "input_size".
        std::vector<unsigned int> reshapedDimensions(2);
        reshapedDimensions[1] = weightInfo.GetShape()[0];
        reshapedDimensions[0] = inputInfo.GetNumElements() / reshapedDimensions[1];

        if (inputInfo.GetNumElements() % reshapedDimensions[1] != 0)
        {
            throw ParseException(
                    fmt::format("Failed to deduce input tensor shape from filter size {} {}",
                                reshapedDimensions[1],
                                CHECK_LOCATION().AsString()));
        }

        TensorInfo reshapedTensorInfo = inputInfo;
        reshapedTensorInfo.SetShape(armnn::TensorShape{ 2, reshapedDimensions.data() });
        inputInfo = reshapedTensorInfo;

        ReshapeDescriptor reshapeDescriptor;
        reshapeDescriptor.m_TargetShape = reshapedTensorInfo.GetShape();

        std::string reshapeLayerName = fmt::format("Reshape_for:{}", layer->GetName());
        IConnectableLayer* reshapeLayer = m_Network->AddReshapeLayer(reshapeDescriptor, reshapeLayerName.c_str());

        reshapeLayer->GetOutputSlot(0).SetTensorInfo(reshapedTensorInfo);
        reshapeLayer->GetOutputSlot(0).Connect(layer->GetInputSlot(0));

        RegisterInputSlots(reshapeLayer, {inputName});
        inputNames[0] = reshapeLayerName;
    }

    auto outputInfo = ComputeOutputInfo({ outputName },
                                        layer,
                                        { inputInfo.GetShape(),
                                          weightInfo.GetShape() });
    layer->GetOutputSlot(0).SetTensorInfo(outputInfo[0]);

    RegisterInputSlots(layer, inputNames);

    // Add constant layer to store weights/biases and connect to FullyConnected layer..
    if(m_TensorsInfo[weightName].isConstant())
    {
        IConnectableLayer* weightsLayer = m_Network->AddConstantLayer(CreateConstTensor(weightName).first);

        weightInfo.SetConstant();
        weightsLayer->GetOutputSlot(0).SetTensorInfo(weightInfo);
        weightsLayer->GetOutputSlot(0).Connect(layer->GetInputSlot(1u));
    }

    if(desc.m_BiasEnabled && m_TensorsInfo[biasName].isConstant())
    {
        IConnectableLayer* biasLayer = m_Network->AddConstantLayer(CreateConstTensor(biasName).first);

        biasInfo.SetConstant();
        biasLayer->GetOutputSlot(0).SetTensorInfo(biasInfo);
        biasLayer->GetOutputSlot(0).Connect(layer->GetInputSlot(2u));
    }

    if (outputInfo[0].GetNumDimensions() > 2)
    {
        // Calculate reshape to flatten to 2D [batch_size, input_size]
        std::vector<unsigned int> reshapedDimensions(2);
        reshapedDimensions[1] = weightInfo.GetShape()[1];
        reshapedDimensions[0] = outputInfo[0].GetNumElements() / reshapedDimensions[1];

        if (outputInfo[0].GetNumElements() % reshapedDimensions[1] != 0)
        {
            throw ParseException(
                    fmt::format("Failed to deduce output tensor shape from filter size {} {}",
                                reshapedDimensions[1],
                                CHECK_LOCATION().AsString()));
        }

        armnn::TensorInfo reshapedOutputTensorInfo = outputInfo[0];
        reshapedOutputTensorInfo.SetShape(armnn::TensorShape{ 2, reshapedDimensions.data() });
        layer->GetOutputSlot(0).SetTensorInfo(reshapedOutputTensorInfo);

        ReshapeDescriptor desc;
        desc.m_TargetShape = outputInfo[0].GetShape();

        std::string reshapeLayerName = fmt::format("ExpandDims_for:{}", layer->GetName());
        IConnectableLayer* reshapeLayer = m_Network->AddReshapeLayer(desc, reshapeLayerName.c_str());

        layer->GetOutputSlot(0).Connect(reshapeLayer->GetInputSlot(0));
        reshapeLayer->GetOutputSlot(0).SetTensorInfo(outputInfo[0]);

        RegisterInputSlots(reshapeLayer, {layer->GetName()});
        layer = reshapeLayer;
    }

    RegisterOutputSlots(layer, { outputName });
}

void OnnxParserImpl::AddPoolingLayer(const onnx::NodeProto& node, Pooling2dDescriptor& desc)
{

    CHECK_VALID_SIZE(static_cast<size_t>(node.input_size()), 1);
    CHECK_VALID_SIZE(static_cast<size_t>(node.output_size()), 1);

    VALID_INPUTS(node, STR_LIST(onnx::TensorProto::FLOAT));

    std::vector<uint32_t> kernel_shape = ReadMandatoryNodeUint32ListAttribute(node, "kernel_shape"); //size of pool win
    std::vector<uint32_t> strides = ReadOptionalNodeUint32ListAttribute(node, "strides");
    std::vector<uint32_t> pads = ReadOptionalNodeUint32ListAttribute(node, "pads");

    desc.m_OutputShapeRounding = OutputShapeRounding::Floor;
    desc.m_PoolWidth  = kernel_shape[1];
    desc.m_PoolHeight = kernel_shape[0];

    if(strides.empty())
    {
        desc.m_StrideX    = 1;
        desc.m_StrideY    = 1;
    }
    else
    {
        desc.m_StrideX    = strides[1];
        desc.m_StrideY    = strides[0];
    }

    //Check new padding version first
    if(pads.empty())
    {
        //Check deprecated version
        std::string paddingString = ReadOptionalNodeStringAttribute(node, "auto_pad");
        if(paddingString != "VALID" && paddingString != "" && paddingString != "NOTSET")
        {
            bool isUpper;
            if( paddingString == "SAME_LOWER")
            {
                isUpper = false;
            }
            else if (paddingString == "SAME_UPPER")
            {
                isUpper = true;
            }
            else
            {
                throw ParseException(fmt::format("Invalid auto_pad attribute for node {}. "
                                                 "Only SAME_UPPER, SAME_LOWER or VALID supported and found {} {}",
                                                 node.name(),
                                                 paddingString,
                                                 CHECK_LOCATION().AsString()));
            }
            auto inputInfo = *m_TensorsInfo[node.input(0)].m_info;
            uint32_t inputHeight = inputInfo.GetShape()[2];
            uint32_t inputWidth  = inputInfo.GetShape()[3];
            CalcPadding(inputHeight,
                        desc.m_PoolHeight,
                        desc.m_StrideY,
                        1u,
                        &desc.m_PadTop,
                        &desc.m_PadBottom,
                        isUpper);
            CalcPadding(inputWidth,
                        desc.m_PoolWidth,
                        desc.m_StrideX,
                        1u,
                        &desc.m_PadLeft,
                        &desc.m_PadRight,
                        isUpper);
        }
    }
    else
    {
        desc.m_PadTop     = pads[0];
        desc.m_PadLeft    = pads[1];
        desc.m_PadBottom  = pads[2];
        desc.m_PadRight   = pads[3];
    }

    IConnectableLayer* layer = m_Network->AddPooling2dLayer(desc, node.name().c_str());
    ARMNN_ASSERT(layer != nullptr);

    auto outputInfo = ComputeOutputInfo({node.output(0)}, layer, {m_TensorsInfo[node.input(0)].m_info->GetShape()});
    layer->GetOutputSlot(0).SetTensorInfo(outputInfo[0]);

    // register the input connection slots for the layer, connections are made after all layers have been created
    // only the tensors for the inputs are relevant, exclude the const tensors
    RegisterInputSlots(layer, {node.input(0)});

    // register the output connection slots for the layer, connections are made after all layers have been created
    RegisterOutputSlots(layer, {node.output(0)});
}

std::pair<std::string, std::string> OnnxParserImpl::AddPrepareBroadcast(const std::string& input0,
                                                                        const std::string& input1)
{
    std::pair<std::string, std::string> inputs = std::make_pair(input0, input1);

    TensorShape input0Shape = m_TensorsInfo[input0].m_info->GetShape();
    TensorShape input1Shape = m_TensorsInfo[input1].m_info->GetShape();

    if(input1Shape.GetNumDimensions() < input0Shape.GetNumDimensions())
    {
        auto outputName = fmt::format("reshape_output_{}", input1);
        PrependForBroadcast(outputName, input1, input0);
        inputs.second = outputName;
    }
    else if(input0Shape.GetNumDimensions() < input1Shape.GetNumDimensions())
    {
        auto outputName = fmt::format("reshape_output_{}", input0);
        PrependForBroadcast(outputName, input0, input1);
        inputs.first = outputName;
    }
    return inputs;
}

void OnnxParserImpl::CreateConstantLayer(const std::string& tensorName, const std::string& layerName)
{
    auto armnnTensor = CreateConstTensor(tensorName);
    IConnectableLayer* layer = m_Network->AddConstantLayer(armnnTensor.first, layerName.c_str());
    layer->GetOutputSlot(0).SetTensorInfo(armnnTensor.first.GetInfo());
    RegisterOutputSlots(layer, {tensorName});
}

void OnnxParserImpl::CreateInt64ConstantLayer(const std::string& tensorName, const std::string& layerName)
{
    auto armnnTensor = CreateInt64ConstTensor(tensorName);
    IConnectableLayer* layer = m_Network->AddConstantLayer(armnnTensor.first, layerName.c_str());
    layer->GetOutputSlot(0).SetTensorInfo(armnnTensor.first.GetInfo());
    RegisterOutputSlots(layer, {tensorName});
}

void OnnxParserImpl::CreateReshapeLayer(const std::string& inputName,
                                        const std::string& outputName,
                                        const std::string& layerName)
{
    const TensorInfo outputTensorInfo = *m_TensorsInfo[outputName].m_info;
    ReshapeDescriptor reshapeDesc;
    reshapeDesc.m_TargetShape = outputTensorInfo.GetShape();

    IConnectableLayer* layer = m_Network->AddReshapeLayer(reshapeDesc, layerName.c_str());
    ARMNN_ASSERT(layer != nullptr);
    layer->GetOutputSlot(0).SetTensorInfo(outputTensorInfo);

    // register the input connection slots for the layer, connections are made after all layers have been created
    // only the tensors for the inputs are relevant, exclude the const tensors
    RegisterInputSlots(layer, {inputName});

    // register the output connection slots for the layer, connections are made after all layers have been created
    RegisterOutputSlots(layer, {outputName});
}

void OnnxParserImpl::ParseActivation(const onnx::NodeProto& node, const armnn::ActivationFunction func)
{
    CHECK_VALID_SIZE(static_cast<size_t>(node.input_size()), 1, 3);
    CHECK_VALID_SIZE(static_cast<size_t>(node.output_size()), 1);

    VALID_INPUTS(node, STR_LIST(onnx::TensorProto::FLOAT));

    ActivationDescriptor desc;
    desc.m_Function = func;

    if (func == ActivationFunction::BoundedReLu)
    {
        if (node.input_size() == 1 && node.attribute_size() > 0)
        {
            desc.m_A = ReadOptionalNodeFloatAttribute(node, "max", std::numeric_limits<float>::max());
            desc.m_B = ReadOptionalNodeFloatAttribute(node, "min", std::numeric_limits<float>::lowest());
        }
        else
        {
            desc.m_A = node.input(2).empty() ? std::numeric_limits<float>::max() : std::stof(node.input(2));
            desc.m_B = node.input(1).empty() ? std::numeric_limits<float>::lowest() : std::stof(node.input(1));
        }
    }

    IConnectableLayer* const layer = m_Network->AddActivationLayer(desc, node.name().c_str());
    ARMNN_ASSERT(layer != nullptr);

    auto outputInfo = ComputeOutputInfo({ node.output(0)}, layer, {m_TensorsInfo[node.input(0)].m_info->GetShape()});
    layer->GetOutputSlot(0).SetTensorInfo(outputInfo[0]);

    // register the input connection slots for the layer, connections are made after all layers have been created
    // only the tensors for the inputs are relevant, exclude the const tensors
    RegisterInputSlots(layer, {node.input(0)});

    // register the output connection slots for the layer, connections are made after all layers have been created
    RegisterOutputSlots(layer, {node.output(0)});
}

void OnnxParserImpl::ParseClip(const onnx::NodeProto& node)
{
    ParseActivation(node, ActivationFunction::BoundedReLu);
}

void OnnxParserImpl::ParseSigmoid(const onnx::NodeProto& node)
{
    ParseActivation(node, ActivationFunction::Sigmoid);
}

void OnnxParserImpl::ParseTanh(const onnx::NodeProto& node)
{
    ParseActivation(node, ActivationFunction::TanH);
}

void OnnxParserImpl::ParseRelu(const onnx::NodeProto& node)
{
    ParseActivation(node, ActivationFunction::ReLu);
}

void OnnxParserImpl::ParseLeakyRelu(const onnx::NodeProto& node)
{
    ParseActivation(node, ActivationFunction::LeakyReLu);
}

void OnnxParserImpl::ParseAdd(const onnx::NodeProto& node)
{
    CHECK_VALID_SIZE(static_cast<size_t>(node.input_size()), 2);
    CHECK_VALID_SIZE(static_cast<size_t>(node.output_size()), 1);

    VALID_INPUTS(node, STR_LIST(onnx::TensorProto::FLOAT));

    // TODO: unify broadcast validation code across layers
    // tracked by: IVGCVSW-1576

    // Checking broadcast compatibility : only scalar or 1D tensors
    auto inputs = AddPrepareBroadcast(node.input(0), node.input(1));
    auto input0 = *m_TensorsInfo[inputs.first].m_info;
    auto input1 = *m_TensorsInfo[inputs.second].m_info;
    ARMNN_ASSERT(input0.GetNumDimensions() == input1.GetNumDimensions());

    unsigned int numDims = input0.GetNumDimensions();
    for (unsigned int i = 0; i < numDims; i++)
    {
        unsigned int dim0 = input0.GetShape()[i];
        unsigned int dim1 = input1.GetShape()[i];
        if (dim0 != dim1 && dim0 != 1 && dim1 != 1)
        {
            throw ParseException(
                fmt::format("Broadcast is only supported for scalar or 1D tensors in Add node '{}'. "
                            "Input dimensions should either match or one should be of size 1 and here, "
                            "{} and {} {}",
                            node.name(),
                            TensorInfoAsString(*m_TensorsInfo[inputs.first].m_info, inputs.first,
                                               m_TensorsInfo[inputs.first].m_dtype),
                            TensorInfoAsString(*m_TensorsInfo[inputs.second].m_info, inputs.second,
                                               m_TensorsInfo[inputs.second].m_dtype),
                            CHECK_LOCATION().AsString()));
        }
    }


    IConnectableLayer* layer = m_Network->AddElementwiseBinaryLayer(BinaryOperation::Add, node.name().c_str());
    ARMNN_ASSERT(layer != nullptr);

    auto outputInfo = ComputeOutputInfo({ node.output(0) }, layer,
                                        { m_TensorsInfo[inputs.first].m_info->GetShape(),
                                          m_TensorsInfo[inputs.second].m_info->GetShape() });
    layer->GetOutputSlot(0).SetTensorInfo(outputInfo[0]);

    // register the input connection -> for constant inputs, we need to make a newDim constant layer
    if(m_TensorsInfo[inputs.first].isConstant()) {
        CreateConstantLayer(inputs.first, fmt::format("Add:constant_of_{}", node.input(0)));
    }
    if(m_TensorsInfo[inputs.second].isConstant()) {
        CreateConstantLayer(inputs.second, fmt::format("Add:constant_of_{}", node.input(1)));
    }
    RegisterInputSlots(layer, {inputs.first, inputs.second});

    // register the output connection
    RegisterOutputSlots(layer, {node.output(0)});
}

void OnnxParserImpl::ParseAveragePool(const onnx::NodeProto& node)
{
    Pooling2dDescriptor desc;
    desc.m_PoolType = PoolingAlgorithm::Average;

    uint32_t count_include_pad = 0;
    count_include_pad = ReadOptionalNodeUint32Attribute(node, "count_include_pad");
    if(count_include_pad) {
        desc.m_PaddingMethod = PaddingMethod::IgnoreValue;
    }
    AddPoolingLayer(node, desc);
}

void OnnxParserImpl::ParseBatchNormalization(const onnx::NodeProto& node)
{
    //IGNORE momentum parameter and spatial parameters

    CHECK_VALID_SIZE(static_cast<size_t>(node.input_size()), 5);
    CHECK_VALID_SIZE(static_cast<size_t>(node.output_size()), 1);

    VALID_INPUTS(node, STR_LIST(onnx::TensorProto::FLOAT));
    for(int ind = 1; ind < node.input_size(); ++ind)
    {
        auto tensor = node.input(ind);
        if(! m_TensorsInfo[tensor].isConstant())
        {
            throw ParseException(
                fmt::format("Input tensor '{}' should be constant in BatchNormalization node '{}' {}",
                            tensor,
                            node.name(),
                            CHECK_LOCATION().AsString()));
        }
    }

    float epsilon = ReadOptionalNodeFloatAttribute(node, "epsilon", 1e-5f);
    BatchNormalizationDescriptor desc;
    desc.m_Eps = epsilon;

    auto scaleTensor = CreateConstTensor(node.input(1));
    auto biasTensor = CreateConstTensor(node.input(2));
    auto meanTensor = CreateConstTensor(node.input(3));
    auto varTensor = CreateConstTensor(node.input(4));

    IConnectableLayer* layer = m_Network->AddBatchNormalizationLayer(desc,
                                                                     meanTensor.first,
                                                                     varTensor.first,
                                                                     biasTensor.first,
                                                                     scaleTensor.first,
                                                                     node.name().c_str());
    ARMNN_ASSERT(layer != nullptr);

    auto outputInfo = ComputeOutputInfo({node.output(0)}, layer, {m_TensorsInfo[node.input(0)].m_info->GetShape()});
    layer->GetOutputSlot(0).SetTensorInfo(outputInfo[0]);

    RegisterInputSlots(layer, {node.input(0)}); //don't register constant inputs

    // register the output connection
    RegisterOutputSlots(layer, {node.output(0)});
}

void OnnxParserImpl::ParseConcat(const onnx::NodeProto& node)
{
    CHECK_VALID_SIZE(static_cast<size_t>(node.output_size()), 1);

    uint32_t numConcatView = static_cast<uint32_t>(node.input_size());
    uint32_t inputRank = m_TensorsInfo[node.input(0)].m_info->GetNumDimensions();

    int axisInt = ReadMandatoryNodeIntAttribute(node, "axis");

    unsigned int concatDimInput = static_cast<unsigned int>(
        (static_cast<int>(inputRank) + axisInt) % static_cast<int>(inputRank));

    OriginsDescriptor concatDescriptor(numConcatView, inputRank);
    concatDescriptor.SetConcatAxis(concatDimInput);

    unsigned int mergeDimOrigin = 0;

    std::vector<TensorShape> inputShapes;
    std::vector<std::string> tensorIds;

    for (unsigned int viewIndex = 0; viewIndex < numConcatView; ++viewIndex)
    {
        std::string nodeName = node.input(static_cast<int>(viewIndex));
        auto inputTensorInfo = *m_TensorsInfo[nodeName].m_info;
        inputShapes.push_back(inputTensorInfo.GetShape());
        tensorIds.push_back(nodeName);

        // Set up concatDescriptor view origin
        armnnUtils::ProcessConcatInputTensorInfo(
            inputTensorInfo, concatDescriptor, concatDimInput, viewIndex, mergeDimOrigin);
    }

    IConnectableLayer* layer = m_Network->AddConcatLayer(concatDescriptor, node.name().c_str());
    ARMNN_ASSERT(layer != nullptr);

    auto outputInfo = ComputeOutputInfo({node.output(0)}, layer, inputShapes,
                                        m_TensorsInfo[node.input(0)].m_dtype);

    layer->GetOutputSlot(0).SetTensorInfo(outputInfo[0]);

    // register the input connection slots for the layer, connections are made after all layers have been created
    RegisterInputSlots(layer, tensorIds);

    // register the output connection slots for the layer, connections are made after all layers have been created
    RegisterOutputSlots(layer, { node.output(0) });
}

void OnnxParserImpl::ParseConstant(const onnx::NodeProto& node)
{
    CHECK_VALID_SIZE(static_cast<size_t>(node.attribute_size()), 1);
    if (!node.attribute(0).has_t())
    {
        throw ParseException(fmt::format("Value not found for Constant node '{}' {}",
                                         node.name(),
                                         CHECK_LOCATION().AsString()));
    }
    const onnx::TensorProto& onnxTensor = node.attribute(0).t();

    //Register this as a m_ConstParam so we know we can use it as a constant param in future layers.
    m_TensorsInfo[node.output(0)].m_tensor = std::make_unique<const onnx::TensorProto>(onnxTensor);
    m_TensorsInfo[node.output(0)].m_info = std::make_unique<TensorInfo>(ToTensorInfo(onnxTensor));
    m_TensorsInfo[node.output(0)].m_dtype = static_cast<onnx::TensorProto::DataType>(onnxTensor.data_type());

    if (m_TensorsInfo[node.output(0)].m_dtype == onnx::TensorProto_DataType_FLOAT)
    {
        CreateConstantLayer(node.output(0), node.name());
    }
    else if (m_TensorsInfo[node.output(0)].m_dtype == onnx::TensorProto_DataType_INT64)
    {
        CreateInt64ConstantLayer(node.output(0), node.name());
    }
    else
    {
        throw ParseException(fmt::format("Data type not support for Constant node '{}' {}",
                                         node.name(),
                                         CHECK_LOCATION().AsString()));
    }
}

void OnnxParserImpl::ParseConv(const onnx::NodeProto& node)
{
    CHECK_VALID_SIZE(static_cast<size_t>(node.input_size()), 2, 3); //input, weight, (bias)
    CHECK_VALID_SIZE(static_cast<size_t>(node.output_size()), 1);

    VALID_INPUTS(node, STR_LIST(onnx::TensorProto::FLOAT));

    if(m_TensorsInfo[node.input(0)].m_info->GetNumDimensions() != 4)
    {
        throw ParseException(
            fmt::format("ArmNN only supports 2D convolution and Conv layer '{}' input {} {}",
                        node.name(),
                        TensorInfoAsString(*m_TensorsInfo[node.input(0)].m_info, node.input(0),
                                           m_TensorsInfo[node.input(0)].m_dtype),
                        CHECK_LOCATION().AsString()));
    }

    if(!m_TensorsInfo[node.input(1)].isConstant())
    {
        throw ParseException(
            fmt::format("Weights '{}' should be constant in Conv layer '{}' {}",
                        node.input(1),
                        node.name(),
                        CHECK_LOCATION().AsString()));
    }

    auto inputInfo = *m_TensorsInfo[node.input(0)].m_info;

    Convolution2dDescriptor desc;
    desc.m_BiasEnabled = false;

    std::vector<uint32_t> strides = ReadOptionalNodeUint32ListAttribute(node, "strides");
    if(strides.empty())
    {
        desc.m_StrideX    = 1;
        desc.m_StrideY    = 1;
    }
    else
    {
        desc.m_StrideX    = strides[1];
        desc.m_StrideY    = strides[0];
    }

    std::vector<uint32_t> dilations = ReadOptionalNodeUint32ListAttribute(node, "dilations");
    if(!dilations.empty())
    {
        desc.m_DilationX = dilations[1];
        desc.m_DilationY = dilations[0];
    }

    std::vector<uint32_t> pads = ReadOptionalNodeUint32ListAttribute(node, "pads");
    //Check new padding version first
    if(pads.empty())
    {
        //Check deprecated version
        std::string paddingString = ReadOptionalNodeStringAttribute(node, "auto_pad");
        if(paddingString != "VALID" && paddingString != "" && paddingString != "NOTSET")
        {
            bool isUpper;
            if( paddingString == "SAME_LOWER")
            {
                isUpper = false;
            }
            else if (paddingString == "SAME_UPPER")
            {
                isUpper = true;
            }
            else
            {
                throw ParseException(
                    fmt::format("Invalid auto_pad attribute for node {}. Only SAME_UPPER, SAME_LOWER or VALID "
                                "supported and found {} {}",
                                node.name(),
                                paddingString,
                                CHECK_LOCATION().AsString()));
            }
            uint32_t inputHeight = inputInfo.GetShape()[2];
            uint32_t inputWidth  = inputInfo.GetShape()[3];

            uint32_t weightHeight;
            uint32_t weightWidth;
            std::vector<uint32_t> kernel_shape = ReadOptionalNodeUint32ListAttribute(node, "kernel_shape");
            if (kernel_shape.empty())
            {
                const TensorInfo weightTensorInfo = *m_TensorsInfo[node.input(1)].m_info;
                weightHeight = weightTensorInfo.GetShape()[2];
                weightWidth = weightTensorInfo.GetShape()[3];
            }
            else
            {
                weightHeight = kernel_shape[0];
                weightWidth = kernel_shape[1];
            }
            CalcPadding(inputHeight,
                        weightHeight,
                        desc.m_StrideY,
                        desc.m_DilationY,
                        &desc.m_PadTop,
                        &desc.m_PadBottom,
                        isUpper);
            CalcPadding(inputWidth,
                        weightWidth,
                        desc.m_StrideX,
                        desc.m_DilationX,
                        &desc.m_PadLeft,
                        &desc.m_PadRight,
                        isUpper);
        }
    }
    else
    {
        desc.m_PadTop     = pads[0];
        desc.m_PadLeft    = pads[1];
        desc.m_PadBottom  = pads[2];
        desc.m_PadRight   = pads[3];
    }

    uint32_t group = ReadOptionalNodeUint32Attribute(node, "group", 1);
    if(group > 1)
    {
        if (group > inputInfo.GetShape()[1])
        {
            throw ParseException(
                fmt::format("Error parsing Convolution node: {}. "
                            "The 'group'={} parameter cannot be larger than the "
                            "channel of the input shape={} (in NCHW format). {}",
                            node.name(),
                            group,
                            inputInfo.GetShape()[1],
                            CHECK_LOCATION().AsString()));
        }
        else if (group == inputInfo.GetShape()[1])
        {
            // we use a depthwise convolution here, because the number of groups equals to the
            // input channels
            AddConvLayerWithDepthwiseConv(node, desc);
            return;
        }
        else
        {
            // TODO: split the input by channels into channels/groups separate convolutions
            //  and concatenate the results afterwards
            throw ParseException(fmt::format("Error parsing Convolution node: {}. "
                                             "The 'group'={} parameter should be 1 or be equal to the "
                                             "channel of the input shape={} (in NCHW format). {}",
                                             node.name(),
                                             group,
                                             inputInfo.GetShape()[1],
                                             CHECK_LOCATION().AsString()));
        }
    }

    node.input_size() == 3 ? desc.m_BiasEnabled = true : desc.m_BiasEnabled = false;
    armnn::IConnectableLayer* layer = m_Network->AddConvolution2dLayer(desc, node.name().c_str());
    std::vector<std::string> tensorIndexes= {node.input(0), node.input(1)};

    auto weightTensor = CreateConstTensor(node.input(1));

    IConnectableLayer* weightsLayer = m_Network->AddConstantLayer(weightTensor.first);
    weightsLayer->GetOutputSlot(0).SetTensorInfo(weightTensor.first.GetInfo());
    weightsLayer->GetOutputSlot(0).Connect(layer->GetInputSlot(1u));

    if (node.input_size() == 3)
    {
        if(!m_TensorsInfo[node.input(2)].isConstant())
        {
            throw ParseException(fmt::format("Bias '{}' should be constant in Conv layer '{}' {}",
                                             node.input(2),
                                             node.name(),
                                             CHECK_LOCATION().AsString()));
        }
        desc.m_BiasEnabled = true;
        auto biasTensor = CreateConstTensor(node.input(2));

        IConnectableLayer* biasLayer = m_Network->AddConstantLayer(biasTensor.first);
        biasLayer->GetOutputSlot(0).SetTensorInfo(biasTensor.first.GetInfo());
        biasLayer->GetOutputSlot(0).Connect(layer->GetInputSlot(2u));

        tensorIndexes.emplace_back(node.input(2));
    }

    ARMNN_ASSERT(layer != nullptr);

    auto outputInfo = ComputeOutputInfo({ node.output(0) }, layer,
                                        { m_TensorsInfo[node.input(0)].m_info->GetShape(),
                                          m_TensorsInfo[node.input(1)].m_info->GetShape() });
    layer->GetOutputSlot(0).SetTensorInfo(outputInfo[0]);

    // register the input connection slots for the layer, connections are made after all layers have been created
    // only the tensors for the inputs are relevant, exclude the const tensors
    RegisterInputSlots(layer, tensorIndexes);

    // register the output connection slots for the layer, connections are made after all layers have been created
    RegisterOutputSlots(layer, {node.output(0)});
}

void OnnxParserImpl::ParseFlatten(const onnx::NodeProto& node)
{
    CHECK_VALID_SIZE(static_cast<size_t>(node.input_size()), 1);
    CHECK_VALID_SIZE(static_cast<size_t>(node.output_size()), 1);

    CHECK_VALID_DATATYPE(node.name(), node.input(0),
                         m_TensorsInfo[node.input(0)].m_dtype,
                         onnx::TensorProto::FLOAT);

    int64_t axis = ReadOptionalNodeInt64Attribute(node, "axis", 1);
    TensorShape inputShape = m_TensorsInfo[node.input(0)].m_info->GetShape();

    /// Negative axis conversion
    if (axis < 0)
    {
        axis += inputShape.GetNumDimensions();
    }

    /// Check Axis is within dimensions
    if (axis < 0 || axis >= inputShape.GetNumDimensions())
    {
        throw ParseException(fmt::format("Axis '{}' invalid. Tensor has '{}' dimensions in FlattenLayer '{}'",
                                         axis, inputShape.GetNumDimensions(), node.name()));
    }

    /// If axis chosen is 0 dimension1 will always be 1 in output , default dimension2 to 1 because 0 is invalid
    uint dimension1{1};
    uint dimension2{1};
    uint i{0};

    /// dimension1 = (d_0 * d_1 ... d_(axis-1))
    for (i = 0; i < axis; i++){
        dimension1 *= inputShape[i];
    }

    /// dimension2 = (d_axis * d_(axis+1) ... d_n)
    for (i = static_cast<uint>(axis); i < inputShape.GetNumDimensions(); i++){
        dimension2 *= inputShape[i];
    }

    TensorShape outputShape{dimension1, dimension2};

    auto outInfo = ComputeReshapeInfo(outputShape, inputShape, node.output(0));
    m_TensorsInfo[node.output(0)].m_info = std::make_unique<TensorInfo>(outInfo);
    CreateReshapeLayer(node.input(0), node.output(0), node.name());
}

void OnnxParserImpl::ParseGather(const onnx::NodeProto& node)
{
    CHECK_VALID_SIZE(static_cast<size_t>(node.input_size()), 2);
    CHECK_VALID_SIZE(static_cast<size_t>(node.output_size()), 1);

    armnn::GatherDescriptor gatherDescriptor;
    gatherDescriptor.m_Axis = static_cast<int>(ReadOptionalNodeInt64Attribute(node, "axis", 0));

    IConnectableLayer* layer = m_Network->AddGatherLayer(gatherDescriptor, node.name().c_str());
    ARMNN_ASSERT(layer != nullptr);

    const TensorShape& inputShape = m_TensorsInfo[node.input(0)].m_info->GetShape();
    const TensorShape& indicesShape = m_TensorsInfo[node.input(1)].m_info->GetShape();
    auto outputInfo = ComputeOutputInfo({node.output(0)}, layer, { inputShape, indicesShape },
                                        m_TensorsInfo[node.input(0)].m_dtype);
    layer->GetOutputSlot(0).SetTensorInfo(outputInfo[0]);

    // register the input connection slots for the layer, connections are made after all layers have been created
    RegisterInputSlots(layer, { node.input(0), node.input(1) });

    // register the output connection slots for the layer, connections are made after all layers have been created
    RegisterOutputSlots(layer, { node.output(0) });
}

void OnnxParserImpl::ParseGemm(const onnx::NodeProto& node)
{
    CHECK_VALID_SIZE(static_cast<size_t>(node.input_size()), 2, 3);
    CHECK_VALID_SIZE(static_cast<size_t>(node.output_size()), 1);

    int transA = static_cast<int>(ReadOptionalNodeUint32Attribute(node, "transA", 0));
    int transB = static_cast<int>(ReadOptionalNodeUint32Attribute(node, "transB", 0));
    float alpha = ReadOptionalNodeFloatAttribute(node, "alpha", 1.0);
    float beta = ReadOptionalNodeFloatAttribute(node, "beta", 1.0);
    bool biasEnabled = node.input_size() == 3;

    TensorShape input0Shape = m_TensorsInfo[node.input(0)].m_info->GetShape();
    TensorShape input1Shape = m_TensorsInfo[node.input(1)].m_info->GetShape();

    // if transB != 0, add transpose to the input1 (tanspose weight matrix in FullyConnected)
    armnn::FullyConnectedDescriptor fullyConnectedDescriptor;
    fullyConnectedDescriptor.m_BiasEnabled = biasEnabled;
    fullyConnectedDescriptor.m_TransposeWeightMatrix = transB;

    IConnectableLayer* layer = nullptr;

    // Just add a FullyConnected layer, weights and biases are handled as inputs now.
    layer = m_Network->AddFullyConnectedLayer(fullyConnectedDescriptor, node.name().c_str());
    ARMNN_ASSERT(layer != nullptr);

    // if transA != 0, add transpose to the input0
    if (transA != 0)
    {
        std::string transAName = "transpose_" + node.input(0);
        armnn::TransposeDescriptor transposeADescriptor;
        transposeADescriptor.m_DimMappings = { 1, 0 };
        IConnectableLayer* transALayer = m_Network->AddTransposeLayer(transposeADescriptor, transAName.c_str());
        ARMNN_ASSERT(transALayer != nullptr);
        auto transAInfo = ComputeOutputInfo({ transAName }, transALayer, { input0Shape });
        transALayer->GetOutputSlot(0).SetTensorInfo(transAInfo[0]);
        transALayer->GetOutputSlot(0).Connect(layer->GetInputSlot(0u));
        // register the input connection slots for the layer, connections are made after all layers have been created
        RegisterInputSlot(transALayer, node.input(0), 0);
        input0Shape = transAInfo[0].GetShape();
    }
    else
    {
        RegisterInputSlot(layer, node.input(0), 0);
    }

    // Add constant layer to store weights/biases and connect to FullyConnected layer.
    if(m_TensorsInfo[node.input(1)].isConstant())
    {
        IConnectableLayer* weightsLayer = m_Network->AddConstantLayer(CreateConstTensor(node.input(1)).first);
        TensorInfo weightInfo = *m_TensorsInfo[node.input(1)].m_info;
        weightInfo.SetConstant();
        weightsLayer->GetOutputSlot(0).SetTensorInfo(weightInfo);

        // if alpha != 1, multiply to the weight
        if (alpha != 1)
        {
            std::string activationName = "activation_" + node.input(1);
            armnn::ActivationDescriptor activationDescriptor;
            activationDescriptor.m_A = alpha;
            activationDescriptor.m_Function = ActivationFunction::Linear;
            IConnectableLayer* actLayer = m_Network->AddActivationLayer(activationDescriptor, activationName.c_str());
            ARMNN_ASSERT(actLayer != nullptr);

            auto actInfo = ComputeOutputInfo({ activationName }, actLayer, { weightInfo.GetShape() });
            actLayer->GetOutputSlot(0).SetTensorInfo(actInfo[0]);
            actLayer->GetOutputSlot(0).Connect(layer->GetInputSlot(1u));
            weightsLayer->GetOutputSlot(0).Connect(actLayer->GetInputSlot(0u));
            input1Shape = actInfo[0].GetShape();
        }
        else
        {
            weightsLayer->GetOutputSlot(0).Connect(layer->GetInputSlot(1u));
            input1Shape = weightInfo.GetShape();
        }
    }
    else
    {
        // if alpha != 1, multiply to the weight
        if (alpha != 1)
        {
            std::string activationName = "activation_" + node.input(1);
            armnn::ActivationDescriptor activationDescriptor;
            activationDescriptor.m_A = alpha;
            activationDescriptor.m_Function = ActivationFunction::Linear;
            IConnectableLayer* actLayer = m_Network->AddActivationLayer(activationDescriptor, activationName.c_str());
            ARMNN_ASSERT(actLayer != nullptr);

            auto actInfo = ComputeOutputInfo({ activationName }, actLayer, { input1Shape });
            actLayer->GetOutputSlot(0).SetTensorInfo(actInfo[0]);
            actLayer->GetOutputSlot(0).Connect(layer->GetInputSlot(1u));
            RegisterInputSlot(actLayer, node.input(1), 0);
            input1Shape = actInfo[0].GetShape();
        }
        else
        {
            RegisterInputSlot(layer, node.input(1), 1);
        }
    }

    if(biasEnabled && m_TensorsInfo[node.input(2)].isConstant())
    {
        To1DTensor(node.input(2), CHECK_LOCATION());
        IConnectableLayer* biasLayer = m_Network->AddConstantLayer(CreateConstTensor(node.input(2)).first);
        TensorInfo biasInfo = *m_TensorsInfo[node.input(2)].m_info;
        biasInfo.SetConstant();
        biasLayer->GetOutputSlot(0).SetTensorInfo(biasInfo);

        // if beta != 1, multiply to the bias
        if (beta != 1)
        {
            std::string activationName = "activation_" + node.input(2);
            armnn::ActivationDescriptor activationDescriptor;
            activationDescriptor.m_A = beta;
            activationDescriptor.m_Function = ActivationFunction::Linear;
            IConnectableLayer* actLayer = m_Network->AddActivationLayer(activationDescriptor, activationName.c_str());
            ARMNN_ASSERT(actLayer != nullptr);

            auto actInfo = ComputeOutputInfo({ activationName }, actLayer, { biasInfo.GetShape() });
            actLayer->GetOutputSlot(0).SetTensorInfo(actInfo[0]);
            actLayer->GetOutputSlot(0).Connect(layer->GetInputSlot(2u));
            biasLayer->GetOutputSlot(0).Connect(actLayer->GetInputSlot(0u));
        }
        else
        {
            biasLayer->GetOutputSlot(0).Connect(layer->GetInputSlot(2u));
        }
    }
    else if (biasEnabled)
    {
        // Currently we support non-constant tensor of input C (bias) of Gemm when the dimension is 1
        if (m_TensorsInfo[node.input(2)].m_info->GetNumDimensions() != 1)
        {
            throw ParseException(fmt::format("The parser supports constant or non-constant with 1 dimension for "
                                             "Input C of Gemm. Input '{}' in '{}' is not supported '{}'",
                                             node.input(2),
                                             node.name(),
                                             CHECK_LOCATION().AsString()));
        }
        // if beta != 1, multiply to the bias
        if (beta != 1)
        {
            std::string activationName = "activation_" + node.input(2);
            armnn::ActivationDescriptor activationDescriptor;
            activationDescriptor.m_A = beta;
            activationDescriptor.m_Function = ActivationFunction::Linear;
            IConnectableLayer* actLayer = m_Network->AddActivationLayer(activationDescriptor, activationName.c_str());
            ARMNN_ASSERT(actLayer != nullptr);

            auto actInfo = ComputeOutputInfo({ activationName },
                                             actLayer,
                                             { m_TensorsInfo[node.input(2)].m_info->GetShape() });
            actLayer->GetOutputSlot(0).SetTensorInfo(actInfo[0]);
            actLayer->GetOutputSlot(0).Connect(layer->GetInputSlot(2u));
            RegisterInputSlot(actLayer, node.input(2), 0);
        }
        else
        {
            RegisterInputSlot(layer, node.input(2), 2);
        }
    }

    // Set final output of the FullyConnected layer
    auto outputInfo = ComputeOutputInfo({ node.output(0) }, layer,
                                        { input0Shape, input1Shape });
    layer->GetOutputSlot(0).SetTensorInfo(outputInfo[0]);

    RegisterOutputSlots(layer, {node.output(0)});
}

void OnnxParserImpl::ParseGlobalAveragePool(const onnx::NodeProto& node)
{
    Pooling2dDescriptor desc = Pooling2dDescriptor();
    desc.m_PoolType = PoolingAlgorithm::Average;

    //kernel size is the same as input
    TensorShape inputShape = m_TensorsInfo[node.input(0)].m_info->GetShape();
    desc.m_PoolWidth  = inputShape[3];
    desc.m_PoolHeight = inputShape[2];

    IConnectableLayer* layer = m_Network->AddPooling2dLayer(desc, node.name().c_str());
    ARMNN_ASSERT(layer != nullptr);

    auto outputInfo = ComputeOutputInfo({node.output(0)}, layer, {inputShape});
    layer->GetOutputSlot(0).SetTensorInfo(outputInfo[0]);

    // register the input connection slots for the layer, connections are made after all layers have been created
    // only the tensors for the inputs are relevant, exclude the const tensors
    RegisterInputSlots(layer, {node.input(0)});

    // register the output connection slots for the layer, connections are made after all layers have been created
    RegisterOutputSlots(layer, {node.output(0)});
}

void OnnxParserImpl::ParseMaxPool(const onnx::NodeProto& node)
{
    Pooling2dDescriptor desc;
    desc.m_PoolType = PoolingAlgorithm::Max;
    desc.m_PaddingMethod = PaddingMethod::Exclude;
    AddPoolingLayer(node, desc);
}

void OnnxParserImpl::ParseShape(const onnx::NodeProto& node)
{
    CHECK_VALID_SIZE(static_cast<size_t>(node.input_size()), 1);
    CHECK_VALID_SIZE(static_cast<size_t>(node.output_size()), 1);

    IConnectableLayer* layer = m_Network->AddShapeLayer(node.name().c_str());
    ARMNN_ASSERT(layer != nullptr);

    TensorShape inputShape = m_TensorsInfo[node.input(0)].m_info->GetShape();
    auto outputInfo = ComputeOutputInfo({node.output(0)}, layer, {inputShape}, onnx::TensorProto::INT64);
    layer->GetOutputSlot(0).SetTensorInfo(outputInfo[0]);

    // register the input connection slots for the layer, connections are made after all layers have been created
    RegisterInputSlots(layer, {node.input(0)});

    // register the output connection slots for the layer, connections are made after all layers have been created
    RegisterOutputSlots(layer, {node.output(0)});
}

void OnnxParserImpl::ParseReshape(const onnx::NodeProto& node)
{
    CHECK_VALID_SIZE(static_cast<size_t>(node.input_size()), 2);
    CHECK_VALID_SIZE(static_cast<size_t>(node.output_size()), 1);

    CHECK_VALID_DATATYPE(node.name(), node.input(0),
                         m_TensorsInfo[node.input(0)].m_dtype,
                         onnx::TensorProto::FLOAT); //input
    CHECK_VALID_DATATYPE(node.name(), node.input(1),
                         m_TensorsInfo[node.input(1)].m_dtype,
                         onnx::TensorProto::INT64); //shape

    TensorShape inputShape = m_TensorsInfo[node.input(0)].m_info->GetShape();

    std::vector<unsigned int> targetShape;
    if(m_TensorsInfo[node.input(1)].isConstant())
    {
        unsigned int dims = static_cast<unsigned int>(m_TensorsInfo[node.input(1)].m_tensor->int64_data_size());
        targetShape.reserve(dims);

        for(uint i = 0; i < dims; i++)
        {
            int val = CHECKED_INT32(m_TensorsInfo[node.input(1)].m_tensor->int64_data(static_cast<int>(i)));
            targetShape[i]= static_cast<unsigned int>(val);
        }
    }
    else
    {
        // The parser only supports shape (batch, -1) or (-1) for non-constant shape input.
        unsigned int dims = m_TensorsInfo[node.input(1)].m_info->GetNumDimensions();
        TensorShape shapes = m_TensorsInfo[node.input(1)].m_info->GetShape();
        if (dims != 1 || shapes[0] > 2)
        {
            throw ParseException(fmt::format("Invalid input shape '{}' in Reshape layer '{}' {}",
                                             node.input(1),
                                             node.name(),
                                             CHECK_LOCATION().AsString()));
        }

        unsigned int numInputElements = m_TensorsInfo[node.input(0)].m_info->GetNumElements();
        if (shapes[0] == 1)
        {
            targetShape = { numInputElements };
        }
        else if (shapes[0] == 2)
        {
            targetShape = { inputShape[0] , numInputElements / inputShape[0] };
        }
    }

    if(m_TensorsInfo[node.input(0)].isConstant())
    {
        //make a new cst tensor -> move the data to the output tensor (the shape is already good in the output tensor)
        if(m_TensorsInfo.count(node.output(0)) == 0)
        {
            m_TensorsInfo[node.output(0)] = OnnxTensor();
        }
        m_TensorsInfo[node.output(0)].m_tensor =
            std::make_unique<onnx::TensorProto>(*m_TensorsInfo[node.input(0)].m_tensor);
    }
    else
    {
        if(m_TensorsInfo.count(node.output(0)) == 0 || m_TensorsInfo[node.output(0)].m_info == nullptr)
        {
            auto outInfo = ComputeReshapeInfo(
                TensorShape(static_cast<unsigned int>(targetShape.size()), targetShape.data()),
                inputShape, node.output(0));
            m_TensorsInfo[node.output(0)].m_info = std::make_unique<TensorInfo>(outInfo);
        }

        CreateReshapeLayer(node.input(0), node.output(0), node.name());
    }
}

void OnnxParserImpl::ParseUnsqueeze(const onnx::NodeProto& node)
{
    CHECK_VALID_SIZE(armnn::numeric_cast<size_t>(node.input_size()), 1, 2);
    CHECK_VALID_SIZE(armnn::numeric_cast<size_t>(node.output_size()), 1);

    TensorShape inputShape = m_TensorsInfo[node.input(0)].m_info->GetShape();
    std::vector<uint32_t> dims;
    if (node.input_size() == 1 && node.attribute_size() > 0)
    {
        dims = ReadMandatoryNodeUint32ListAttribute(node, "axes");
    }
    else
    {
        CHECK_VALID_DATATYPE(node.name(), node.input(1),
                             m_TensorsInfo[node.input(1)].m_dtype,
                             onnx::TensorProto::INT64); //axes

        auto int64Axes = m_TensorsInfo[node.input(1)].m_tensor->int64_data().data();
        uint numDim = armnn::numeric_cast<uint>(m_TensorsInfo[node.input(1)].m_tensor->int64_data_size());

        for(uint i = 0; i < numDim; i++)
        {
            uint32_t uint32Value = CHECKED_NON_NEGATIVE(CHECKED_INT32(int64Axes[i]));
            dims.push_back(uint32Value);
        }
    }

    // Ensure that the axes are sorted
    std::sort(dims.begin(), dims.end());

    std::vector<unsigned int> targetShape;

    if (inputShape.GetDimensionality() != Dimensionality::Scalar)
    {
        for(uint i = 0; i < inputShape.GetNumDimensions(); i++)
        {
            targetShape.push_back(inputShape[i]);
        }
    }

    for(uint i = 0; i < dims.size(); i++)
    {
        targetShape.insert(targetShape.begin() + armnn::numeric_cast<int>(dims[i]), 1);
    }

    auto outInfo = ComputeReshapeInfo(TensorShape(static_cast<unsigned int>(targetShape.size()), targetShape.data()),
                                      inputShape, node.output(0), m_TensorsInfo[node.input(0)].m_info->GetDataType());
    m_TensorsInfo[node.output(0)].m_info = std::make_unique<TensorInfo>(outInfo);
    m_TensorsInfo[node.output(0)].m_dtype = m_TensorsInfo[node.input(0)].m_dtype;

    CreateReshapeLayer(node.input(0), node.output(0), node.name());
}

void OnnxParserImpl::PrependForBroadcast(const std::string& outputName,
                                         const std::string& input0,
                                         const std::string& input1)
{
    //input0 should be reshaped to have same number of dim as input1
    TensorInfo outputTensorInfo = TensorInfo(*m_TensorsInfo[input0].m_info);

    TensorShape input0Shape = m_TensorsInfo[input0].m_info->GetShape();
    TensorShape input1Shape = m_TensorsInfo[input1].m_info->GetShape();

    uint32_t diff = input1Shape.GetNumDimensions() - input0Shape.GetNumDimensions();
    std::vector<uint32_t> newShape;
    while(diff > 0)
    {
        newShape.push_back(1);
        diff--;
    }
    for (uint dim = 0; dim < input0Shape.GetNumDimensions(); ++dim)
    {
        newShape.push_back(input0Shape[dim]);
    }
    outputTensorInfo.SetShape(TensorShape(static_cast<unsigned int>(newShape.size()), newShape.data()));

    //add the new tensor to m_TensorsInfo
    m_TensorsInfo[outputName] = OnnxTensor();
    m_TensorsInfo[outputName].m_info = std::make_unique<TensorInfo>(outputTensorInfo);

    //add reshape layer if the parent was not constant...
    if( ! m_TensorsInfo[input0].isConstant())
    {
        CreateReshapeLayer(input0, outputName, fmt::format("Add:reshapeOf{}", input0));
    }
    else //make it constant and it will be create in Add
    {
        m_TensorsInfo[outputName].m_tensor = std::make_unique<onnx::TensorProto>(*m_TensorsInfo[input0].m_tensor);

    }
}

void OnnxParserImpl::SetupInputLayers()
{
    //Find user input and add their layers
    for(int inputIndex = 0; inputIndex < m_Graph->input_size(); ++inputIndex)
    {
        auto input = m_Graph->input(inputIndex);
        if (!m_TensorsInfo[input.name()].isConstant())
        {
            IConnectableLayer* layer =
                m_Network->AddInputLayer(static_cast<armnn::LayerBindingId>(inputIndex), input.name().c_str());
            TensorInfo tensorInfo = *m_TensorsInfo[input.name()].m_info;
            if (tensorInfo.GetShape().GetDimensionality() == Dimensionality::NotSpecified)
            {
                if (m_InputShapes.find(input.name()) == m_InputShapes.end())
                {
                    throw ParseException(fmt::format("The parser does not support dynamic tensor, "
                                                     "please specify input shape for {}. {}",
                                                     input.name(),
                                                     CHECK_LOCATION().AsString()));
                }
                else
                {
                    tensorInfo.SetShape(m_InputShapes[input.name()]);
                    m_TensorsInfo[input.name()].m_info = std::make_unique<TensorInfo>(tensorInfo);
                }

            }
            layer->GetOutputSlot(0).SetTensorInfo(tensorInfo);

            m_InputInfos[input.name()] = tensorInfo;

            RegisterOutputSlots(layer,{ input.name() });
        }
    }
}

void OnnxParserImpl::SetupOutputLayers()
{
    if(m_Graph->output_size() == 0)
    {
        throw ParseException(fmt::format("The given model does not have any outputs {}", CHECK_LOCATION().AsString()));
    }

    for(int outputIndex = 0; outputIndex < m_Graph->output_size(); ++outputIndex)
    {
        IConnectableLayer* layer =
            m_Network->AddOutputLayer(static_cast<armnn::LayerBindingId>(outputIndex),
                m_Graph->output(outputIndex).name().c_str());

        RegisterInputSlots(layer, { m_Graph->output(outputIndex).name() });
    }
}

void OnnxParserImpl::RegisterInputSlot(IConnectableLayer* layer,
                                       const std::string& tensorId,
                                       unsigned int slotIndex)
{
    armnn::IInputSlot* slot = &(layer->GetInputSlot(slotIndex));

    auto it = m_TensorConnections.find(tensorId);

    if (it == m_TensorConnections.end())
    {
        //First time seeing this tensor, we need to map it
        m_TensorConnections[tensorId] = TensorSlots();
    }
    m_TensorConnections[tensorId].inputSlots.push_back(slot);
}

void OnnxParserImpl::RegisterInputSlots(IConnectableLayer* layer, const std::vector<std::string>& tensorIds)
{
    ARMNN_ASSERT(layer != nullptr);
    if (tensorIds.size() != layer->GetNumInputSlots())
    {
        throw ParseException(
            fmt::format("The number of tensor inputs ({}) does not match the number expected ({}) {}",
                        tensorIds.size(),
                        layer->GetNumInputSlots(),
                        CHECK_LOCATION().AsString()));
    }

    for (unsigned int slotIndex = 0; slotIndex < layer->GetNumInputSlots(); ++slotIndex)
    {
        std::string tensorId = tensorIds[slotIndex];
        armnn::IInputSlot* slot = &(layer->GetInputSlot(slotIndex));

        auto it = m_TensorConnections.find(tensorId);

        if (it == m_TensorConnections.end())
        {
            // First time seing this tensor, we need to map it
            m_TensorConnections[tensorId] = TensorSlots();
        }
        m_TensorConnections[tensorId].inputSlots.push_back(slot);
    }
}

void OnnxParserImpl::RegisterOutputSlots(IConnectableLayer* layer, const std::vector<std::string>& tensorIds)
{
    ARMNN_ASSERT(layer != nullptr);
    if (tensorIds.size() != layer->GetNumOutputSlots())
    {
        throw ParseException(
            fmt::format("The number of tensor outputs ({}) does not match the number expected ({}) {} ",
                        tensorIds.size(),
                        layer->GetNumOutputSlots(),
                        CHECK_LOCATION().AsString()));
    }

    for (unsigned int slotIndex = 0; slotIndex < layer->GetNumOutputSlots(); ++slotIndex)
    {
        std::string tensorId = tensorIds[slotIndex];
        armnn::IOutputSlot* slot = &(layer->GetOutputSlot(slotIndex));

        auto it = m_TensorConnections.find(tensorId);

        if (it == m_TensorConnections.end())
        {
            //First time seing this tensor, we need to map it
            m_TensorConnections[tensorId] = TensorSlots();
        }

        TensorSlots& tensorSlots = m_TensorConnections[tensorId];

        // assuming there is only one producer for that tensor
        if (tensorSlots.outputSlot != nullptr)
        {
            throw ParseException(fmt::format("Another layer has already registered itself as the producer of "
                                             "tensor:{} {}",
                                             tensorId,
                                             CHECK_LOCATION().AsString()));
        }
        tensorSlots.outputSlot = slot;
    }

}

BindingPointInfo OnnxParserImpl::GetNetworkInputBindingInfo(const std::string& name) const
{
    for(int i = 0; i < m_Graph->input_size(); ++i)
    {
        auto input = m_Graph->input(i);
        if(input.name() == name)
        {
            auto it = m_InputInfos.find(name);

            if (it != m_InputInfos.end())
            {
                return std::make_pair(static_cast<armnn::LayerBindingId>(i), it->second);
            }
        }
    }
    throw InvalidArgumentException(fmt::format("The input layer '{}' does not exist {}",
                                               name, CHECK_LOCATION().AsString()));
}

BindingPointInfo OnnxParserImpl::GetNetworkOutputBindingInfo(const std::string& name) const
{
    for(int i = 0; i < m_Graph->output_size(); ++i)
    {
        auto output = m_Graph->output(i);
        if(output.name() == name)
        {
            auto it = m_OutputInfos.find(name);

            if (it != m_OutputInfos.end())
            {
                return std::make_pair(static_cast<armnn::LayerBindingId>(i), it->second);
            }
        }
    }
    throw InvalidArgumentException(fmt::format("The output layer '{}' does not exist {}",
                                               name, CHECK_LOCATION().AsString()));
}

std::vector<std::string> OnnxParserImpl::GetInputs(ModelPtr& model)
{
    if(model == nullptr) {
        throw InvalidArgumentException(fmt::format("The given model cannot be null {}",
                                                   CHECK_LOCATION().AsString()));
    }

    std::vector<std::string> inputNames;
    std::map<std::string, bool> isConstant;
    for(auto tensor : model->graph().initializer())
    {
        isConstant[tensor.name()] = true;
    }
    for(auto input : model->graph().input())
    {
        auto it = isConstant.find(input.name());
        if(it == isConstant.end())
        {
            inputNames.push_back(input.name());
        }
    }
    return inputNames;
}

std::vector<std::string> OnnxParserImpl::GetOutputs(ModelPtr& model)
{
    if(model == nullptr) {
        throw InvalidArgumentException(fmt::format("The given model cannot be null {}",
                                                   CHECK_LOCATION().AsString()));
    }

    std::vector<std::string> outputNames;
    for(auto output : model->graph().output())
    {
        outputNames.push_back(output.name());
    }
    return outputNames;
}

const std::string OnnxParserImpl::GetVersion()
{
    return ONNX_PARSER_VERSION;
}

} // namespace armnnOnnxParser
