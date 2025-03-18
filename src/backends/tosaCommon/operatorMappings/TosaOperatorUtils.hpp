//
// Copyright © 2022-2025 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//
//
// Copyright © 2020 The TensorFlow Authors. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <Layer.hpp>
#include <armnn/Tensor.hpp>
#include <armnn/Types.hpp>

#include "common/include/ProfilingGuid.hpp"

#include <tosa_serialization_handler.h>

using namespace armnn;
using namespace tosa;

const std::string mainName = "main";

// Function to return Tosa datatype from input ArmNN datatype.
inline DType ArmNNToDType(const DataType& type)
{
    switch (type)
    {
        case DataType::Float16:
            return DType_FP16;
        case DataType::BFloat16:
            return DType_BF16;
        case DataType::Float32:
            return DType_FP32;
        case DataType::QAsymmU8:
            return DType_UINT8;
        case DataType::QSymmS8:
        case DataType::QAsymmS8:
            return DType_INT8;
        case DataType::QSymmS16:
            return DType_INT16;
        case DataType::Signed32:
            return DType_INT32;
        case DataType::Signed64:
            // No signed 64, only DType_INT48.
            return DType_UNKNOWN;
        case DataType::Boolean:
            return DType_BOOL;
        default:
            return DType_UNKNOWN;
    }
}

// Function to return ArmNN datatype from input Tosa datatype.
inline DataType DtypeToArmNN(const DType type)
{
    switch (type)
    {
        case DType_FP16:
            return DataType::Float16;
        case DType_BF16:
            return DataType::BFloat16;
        case DType_FP32:
            return DataType::Float32;
        case DType_UINT8:
            return DataType::QAsymmU8;
        case DType_INT8:
            return DataType::QSymmS8;
        case DType_INT16:
            return DataType::QSymmS16;
        case DType_INT32:
            return DataType::Signed32;
        case DType_BOOL:
            return DataType::Boolean;
        default:
            throw armnn::Exception("DtypeToArmNN: Unsupported tosa::DType in ArmNN.");
            return DataType::Boolean;
    }
}

// Function to return Tosa tensor shape from input ArmNN tensor shape.
inline std::vector<int32_t> GetTosaTensorShape(const TensorShape& shape)
{
    std::vector<int32_t> returnShape;
    for (u_int32_t i = 0; i < shape.GetNumDimensions(); i++)
    {
        returnShape.push_back(static_cast<int32_t>(shape[i]));
    }
    return returnShape;
}

// Function that generates unique name using the layer type, input slot and layer guid.
static std::string GenerateUniqueName(const Layer& layer, uint32_t layerSlot)
{
    std::string guid        = std::to_string(layer.GetGuid());
    std::string slotAndGuid = std::to_string(layerSlot) + "_" + guid;

    switch (layer.GetType())
    {
        case LayerType::Input:
            return "input_" + guid;
        case LayerType::Output:
            return "output" + slotAndGuid;
        case LayerType::Constant:
            return "constant_" + guid;
        default:
            return "intermediate" + slotAndGuid;
    }
}

// Function that generates unique name for the parent layer from the child layer input slot.
inline std::string GenerateUniqueInputName(const armnn::InputSlot& slot)
{
    // Get the layers connected to the input slots and determine unique tensor names.
    Layer& connectedLayer = slot.GetConnectedOutputSlot()->GetOwningLayer();
    // For layer input, we want to ensure we get the correct output slot of the parent layer.
    // For example, if parent layer is split, the parent output slot could be 0 or 1 index.
    uint32_t connectedInputSlotIdx = slot.GetConnectedOutputSlot()->CalculateIndexOnOwner();
    return GenerateUniqueName(connectedLayer, connectedInputSlotIdx);
}

// Function to determine if inputs are from different layers.
inline bool WeightFromDifferentLayer(const Layer& layer)
{
    bool multipleParents = false;
    if (layer.GetNumInputSlots()> 1)
    {
        multipleParents = layer.GetInputSlots()[0].GetConnectedOutputSlot()->GetOwningLayerGuid() !=
                         layer.GetInputSlots()[1].GetConnectedOutputSlot()->GetOwningLayerGuid();
    }

    return multipleParents;
}

// Function that generates unique output name using the layer type, input slot and layer guid.
inline std::string GenerateUniqueOutputName(const Layer& layer, uint32_t layerSlot = 0)
{
    Layer& connectedLayer = layer.GetOutputSlot().GetConnection(0)->GetOwningLayer();

    // Get the layer connected to the output slot, if output use that layer and id,
    // otherwise use current layer and id.
    if(connectedLayer.GetType() == LayerType::Output)
    {
        return GenerateUniqueName(connectedLayer, layerSlot);
    }
    else
    {
        return GenerateUniqueName(layer, layerSlot);
    }
}

// Function to return unique int as a string to ensure uniqueness between all input, output and block names.
inline int uniqueTosaMappingID = 0;
inline std::string GetUniqueTosaMappingID()
{
    return std::to_string(++uniqueTosaMappingID);
}

// Function to return Tosa DType as string.
inline std::string TosaDTypeToString(DType tosaDType)
{
    switch (tosaDType)
    {
        case DType_UNKNOWN:
            return "DType_UNKNOWN";
        case DType_BOOL:
            return "DType_BOOL";
        case DType_UINT8:
            return "DType_UINT8";
        case DType_INT4:
            return "DType_INT4";
        case DType_INT8:
            return "DType_INT8";
        case DType_INT16:
            return "DType_INT16";
        case DType_INT32:
            return "DType_INT32";
        case DType_INT48:
            return "DType_INT48";
        case DType_FP32:
            return "DType_FP32";
        case DType_UINT16:
            return "DType_UINT16";
        case DType_FP16:
            return "DType_FP16";
        case DType_BF16:
            return "DType_BF16";
        case DType_SHAPE:
            return "DType_SHAPE";
    }
    return "";
}

// Function to return Tosa Op as string.
inline std::string TosaOpToString(Op tosaOp)
{
    switch (tosaOp)
    {
        case Op_ADD:
            return "Op_ADD";
        case Op_AVG_POOL2D:
            return "Op_AVG_POOL2D";
        case Op_MAX_POOL2D:
            return "Op_MAX_POOL2D";
        case Op_PAD:
            return "Op_PAD";
        case Op_UNKNOWN:
            return "Op_UNKNOWN";
        case Op_ARGMAX:
            return "Op_ARGMAX";
        case Op_CONV2D:
            return "Op_CONV2D";
        case Op_CONV3D:
            return "Op_CONV3D";
        case Op_DEPTHWISE_CONV2D:
            return "Op_DEPTHWISE_CONV2D";
        case Op_FULLY_CONNECTED:
            return "Op_FULLY_CONNECTED";
        case Op_MATMUL:
            return "Op_MATMUL";
        case Op_TRANSPOSE_CONV2D:
            return "Op_TRANSPOSE_CONV2D";
        case Op_CLAMP:
            return "Op_CLAMP";
        case Op_RESERVED:
            return "Op_RESERVED";
        case Op_SIGMOID:
            return "Op_SIGMOID";
        case Op_TANH:
            return "Op_TANH";
        case Op_ARITHMETIC_RIGHT_SHIFT:
            return "Op_ARITHMETIC_RIGHT_SHIFT";
        case Op_BITWISE_AND:
            return "Op_BITWISE_AND";
        case Op_BITWISE_OR:
            return "Op_BITWISE_OR";
        case Op_BITWISE_XOR:
            return "Op_BITWISE_XOR";
        case Op_INTDIV:
            return "Op_INTDIV";
        case Op_LOGICAL_AND:
            return "Op_LOGICAL_AND";
        case Op_LOGICAL_LEFT_SHIFT:
            return "Op_LOGICAL_LEFT_SHIFT";
        case Op_LOGICAL_RIGHT_SHIFT:
            return "Op_LOGICAL_RIGHT_SHIFT";
        case Op_LOGICAL_OR:
            return "Op_LOGICAL_OR";
        case Op_LOGICAL_XOR:
            return "Op_LOGICAL_XOR";
        case Op_MAXIMUM:
            return "Op_MAXIMUM";
        case Op_MINIMUM:
            return "Op_MINIMUM";
        case Op_MUL:
            return "Op_MUL";
        case Op_POW:
            return "Op_POW";
        case Op_SUB:
            return "Op_SUB";
        case Op_TABLE:
            return "Op_TABLE";
        case Op_ABS:
            return "Op_ABS";
        case Op_BITWISE_NOT:
            return "Op_BITWISE_NOT";
        case Op_CEIL:
            return "Op_CEIL";
        case Op_CLZ:
            return "Op_CLZ";
        case Op_EXP:
            return "Op_EXP";
        case Op_FLOOR:
            return "Op_FLOOR";
        case Op_LOG:
            return "Op_LOG";
        case Op_LOGICAL_NOT:
            return "Op_LOGICAL_NOT";
        case Op_NEGATE:
            return "Op_NEGATE";
        case Op_RECIPROCAL:
            return "Op_RECIPROCAL";
        case Op_RSQRT:
            return "Op_RSQRT";
        case Op_SELECT:
            return "Op_SELECT";
        case Op_EQUAL:
            return "Op_EQUAL";
        case Op_GREATER:
            return "Op_GREATER";
        case Op_GREATER_EQUAL:
            return "Op_GREATER_EQUAL";
        case Op_REDUCE_ANY:
            return "Op_REDUCE_ANY";
        case Op_REDUCE_ALL:
            return "Op_REDUCE_ALL";
        case Op_REDUCE_MAX:
            return "Op_REDUCE_MAX";
        case Op_REDUCE_MIN:
            return "Op_REDUCE_MIN";
        case Op_REDUCE_PRODUCT:
            return "Op_REDUCE_PRODUCT";
        case Op_REDUCE_SUM:
            return "Op_REDUCE_SUM";
        case Op_CONCAT:
            return "Op_CONCAT";
        case Op_RESHAPE:
            return "Op_RESHAPE";
        case Op_REVERSE:
            return "Op_REVERSE";
        case Op_SLICE:
            return "Op_SLICE";
        case Op_TILE:
            return "Op_TILE";
        case Op_TRANSPOSE:
            return "Op_TRANSPOSE";
        case Op_GATHER:
            return "Op_GATHER";
        case Op_SCATTER:
            return "Op_SCATTER";
        case Op_RESIZE:
            return "Op_RESIZE";
        case Op_CAST:
            return "Op_CAST";
        case Op_RESCALE:
            return "Op_RESCALE";
        case Op_CONST:
            return "Op_CONST";
        case Op_IDENTITY:
            return "Op_IDENTITY";
        case Op_CUSTOM:
            return "Op_CUSTOM";
        case Op_COND_IF:
            return "Op_COND_IF";
        case Op_WHILE_LOOP:
            return "Op_WHILE_LOOP";
        case Op_FFT2D:
            return "Op_FFT2D";
        case Op_RFFT2D:
            return "Op_RFFT2D";
        case Op_ERF:
            return "Op_ERF";
        case Op_DIM: // = Op_MAX
            return "Op_DIM";
    }
    return "";
}

inline std::vector<uint8_t> ConvertConstantTensorDataToBuffer(const std::shared_ptr<ConstTensorHandle>& tensorHandle)
{
    tosa_err_t error = tosa_err_t::TOSA_OK;
    std::vector<uint8_t> uint8Data;
    auto tensorInfo = tensorHandle->GetTensorInfo();

    switch (tensorInfo.GetDataType())
    {
        case DataType::Float32:
        {
            std::vector<float> data(tensorInfo.GetNumElements());
            memcpy(data.data(), tensorHandle->Map(true), tensorInfo.GetNumBytes());

            error = TosaSerializationHandler::ConvertF32toU8(data, uint8Data);
            break;
        }
        case DataType::Float16:
        {
            std::vector<float> data(tensorInfo.GetNumElements());
            memcpy(data.data(), tensorHandle->Map(true), tensorInfo.GetNumBytes());

            error = TosaSerializationHandler::ConvertF16toU8(data, uint8Data);
            break;
        }
        case DataType::QSymmS8:
        case DataType::QAsymmS8:
        {
            std::vector<int8_t> data(tensorInfo.GetNumElements());
            memcpy(data.data(), tensorHandle->Map(true), tensorInfo.GetNumBytes());

            error = TosaSerializationHandler::ConvertI8toU8(data, uint8Data);
            break;
        }
        case DataType::QAsymmU8:
        {
            memcpy(uint8Data.data(), tensorHandle->Map(true), tensorInfo.GetNumBytes());
            break;
        }
        case DataType::QSymmS16:
        {
            std::vector<int16_t> data(tensorInfo.GetNumElements());
            memcpy(data.data(), tensorHandle->Map(true), tensorInfo.GetNumBytes());

            error = TosaSerializationHandler::ConvertI16toU8(data, uint8Data);
            break;
        }
        case DataType::Signed32:
        {
            std::vector<int32_t> data(tensorInfo.GetNumElements());
            memcpy(data.data(), tensorHandle->Map(true), tensorInfo.GetNumBytes());

            error = TosaSerializationHandler::ConvertI32toU8(data, uint8Data);
            break;
        }
        default:
        {
            throw armnn::Exception("SetConstantTensorData: An unsupported data type was encountered.");
        }
    }

    if(error != tosa_err_t::TOSA_OK)
    {
        throw armnn::Exception("SetConstantTensorData: An error occurred when converting constant data");
    }

    tensorHandle->Unmap();
    return uint8Data;
}

inline std::vector<uint8_t> CreateConstTosaData(const void* value,
                                                DType dtype,
                                                const std::vector<int32_t>& shape)
{
    std::vector<uint8_t> uint8Data;
    tosa_err_t error = tosa_err_t::TOSA_OK;

    unsigned int numElements = 1;
    for (auto s : shape)
    {
        if (s < 0)
        {
            throw armnn::Exception("CreateConstTosaData: negative shape elements unhandled.");
        }
        numElements = numElements * static_cast<unsigned int>(s);
    }

    switch (dtype)
    {
        case DType::DType_FP32:
        {
            std::vector<float> data(numElements, *static_cast<const float*>(value));
            error = TosaSerializationHandler::ConvertF32toU8(data, uint8Data);
            break;
        }
        case DType::DType_FP16:
        {
            std::vector<float> data(numElements, *static_cast<const float*>(value));
            error = TosaSerializationHandler::ConvertF16toU8(data, uint8Data);
            break;
        }
        case DType::DType_INT48:
        {
            std::vector<int64_t> data(numElements, *static_cast<const int64_t*>(value));
            error = TosaSerializationHandler::ConvertI48toU8(data, uint8Data);
            break;
        }
        case DType::DType_INT32:
        {
            std::vector<int32_t> data(numElements, *static_cast<const int32_t*>(value));
            error = TosaSerializationHandler::ConvertI32toU8(data, uint8Data);
            break;
        }
        case DType::DType_INT16:
        {
            std::vector<int16_t> data(numElements, *static_cast<const int16_t*>(value));
            error = TosaSerializationHandler::ConvertI16toU8(data, uint8Data);
            break;
        }
        case DType::DType_INT8:
        {
            std::vector<int8_t> data(numElements, *static_cast<const int8_t*>(value));
            error = TosaSerializationHandler::ConvertI8toU8(data, uint8Data);
            break;
        }
        case DType::DType_UINT8:
        {
            const int8_t* copy_data = static_cast<const int8_t*>(value);
            uint8Data.assign(copy_data, copy_data + numElements);
            break;
        }
        case DType::DType_INT4:
        {
            std::vector<int8_t> data(numElements, *static_cast<const int8_t*>(value));
            error = TosaSerializationHandler::ConvertI4toU8(data, uint8Data);
            break;
        }
        case DType::DType_BOOL:
        {
            std::vector<bool> data(numElements, *static_cast<const bool*>(value));
            error = TosaSerializationHandler::ConvertBooltoU8(data, uint8Data);
            break;
        }
        default:
        {
            throw armnn::Exception("CreateConstTosaData: An unsupported data type was encountered.");
        }
    }

    if(error != tosa_err_t::TOSA_OK)
    {
        throw armnn::Exception("CreateConstTosaData: An error occurred when converting constant data");
    }

    return uint8Data;
}

template<typename T>
inline void CreateConstTosaOperator(const std::string& outputName,
                                    const T value,
                                    DType dtype,
                                    const std::vector<int32_t>& shape,
                                    TosaSerializationOperator*& op,
                                    TosaSerializationTensor*& tensor)
{
    if (outputName.find("constant") == std::string::npos)
    {
        throw armnn::Exception(std::string("CreateConstTosaOperator: outputName must contain the string 'constant'"));
    }

    std::vector<uint8_t> uint8Data = CreateConstTosaData(static_cast<const void *>(&value), dtype, shape);

    op = new TosaSerializationOperator(Op_CONST, Attribute_NONE, nullptr, {}, {outputName});
    ARMNN_THROW_MSG_IF_FALSE(op, armnn::Exception, "CreateConstTosaOperator: failed to created operator");

    tensor = new TosaSerializationTensor(outputName, shape, dtype, uint8Data);
    ARMNN_THROW_MSG_IF_FALSE(tensor, armnn::Exception, "CreateConstTosaOperator: failed to created tensor");
}

inline bool IsUnsignedDataType(DType type)
{
    bool type_unsigned = false;
    switch(type)
    {
        case DType_UINT8:
        case DType_UINT16:
            type_unsigned = true;
            break;
        default:
            type_unsigned = false;
            break;
    }
    return type_unsigned;
}

inline void FlipSignage(DType& type)
{
    switch(type)
    {
        case DType_UINT8:
            type = DType_INT8;
            break;
        case DType_UINT16:
            type = DType_INT16;
            break;
        case DType_INT8:
            type = DType_UINT8;
            break;
        case DType_INT16:
            type = DType_UINT16;
            break;
        default:
            throw armnn::Exception("Unknown type to change signage");
    }
}

// This function is paraphrased from:
// tensorflow/core/util/tensor_format.h from function GetTensorSpatialDimIndex
inline int GetTensorSpatialDimIndex(DataLayout format, int spatialDim)
{
  switch (format)
  {
    case DataLayout::NHWC:
        return spatialDim + 1;
    case DataLayout::NCHW:
    case DataLayout::NDHWC:
        return spatialDim + 2;
    case DataLayout::NCDHW:
        return spatialDim + 3;
    default:
      throw Exception("GetTensorSpatialDimIndex Unknown format");
  }
}

// This function is paraphrased from:
// tensorflow/core/util/tensor_format.h from function GetTensorSpatialDims
inline int GetTensorSpatialDims(int numDims, DataLayout format)
{
  switch (format)
  {
    case DataLayout::NHWC:
    case DataLayout::NCHW:
        return numDims - 2; // Exclude N,C.
    case DataLayout::NDHWC:
    case DataLayout::NCDHW:
      return numDims - 3;  // Exclude N,C,D.
    default:
      throw Exception("GetTensorFeatureDimIndex Unknown format");
  }
}

// This function is paraphrased from:
// tensorflow/compiler/mlir/tosa/transforms/legalize_utils.cc from function getInputSlicedToItsUsedSize
inline std::string GetInputSlicedToItsUsedSize(const std::vector<int32_t>& inputShape,
                                               const std::string& inputName,
                                               const DataLayout layout,
                                               const DType datatype,
                                               const std::vector<int32_t>& kernel,
                                               const std::vector<int32_t>& pad,
                                               const std::vector<int32_t>& stride,
                                               const std::vector<int32_t>& dilations,
                                               std::vector<TosaSerializationTensor*>& tensors,
                                               std::vector<TosaSerializationOperator*>& operators,
                                               const bool isPoolingOp = false)
{
    const int32_t spatialDims = GetTensorSpatialDims(static_cast<int>(inputShape.size()), layout);

    std::vector<int32_t> outputSizeRemainder;
    for (int spatialDim = 0; spatialDim < spatialDims; spatialDim++)
    {
        const size_t spatialDimSize_t = static_cast<size_t>(spatialDim);
        const size_t spatialDimIndexSize_t = static_cast<size_t>(GetTensorSpatialDimIndex(layout, spatialDim));
        const int32_t kernelVal = isPoolingOp ? kernel[spatialDimSize_t] : kernel[spatialDimIndexSize_t];

        const int32_t inSize = inputShape[spatialDimIndexSize_t];
        const int32_t fullPad = pad[2 * spatialDimSize_t + 0] + pad[2 * spatialDimSize_t + 1];
        const int32_t fullSize = inSize - 1 + fullPad - (kernelVal - 1) * dilations[spatialDimSize_t];
        outputSizeRemainder.push_back(fullSize % stride[spatialDimSize_t]);
    }

    const bool needSlicing = std::any_of(
        outputSizeRemainder.begin(), outputSizeRemainder.end(), [](int64_t v) { return v > 0; });
    const bool zeroPads = std::all_of(pad.begin(), pad.end(), [](int v) { return v == 0; });

    std::string sliceOutputName = inputName;
    if (needSlicing && zeroPads)
    {
        sliceOutputName = std::string("layer_intermediate1_") + GetUniqueTosaMappingID();
        std::vector<int32_t> start(inputShape.size(), 0);
        std::vector<int32_t> size = inputShape;
        for (int spatialDim = 0; spatialDim < spatialDims; spatialDim++)
        {
            const int index = GetTensorSpatialDimIndex(layout, spatialDim);
            size[static_cast<size_t>(index)] -= outputSizeRemainder[static_cast<size_t>(spatialDim)];
        }

        TosaSliceAttribute attribute(start, size);

        operators.push_back(new TosaSerializationOperator(Op_SLICE,
                                                          Attribute_SliceAttribute,
                                                          &attribute,
                                                          {inputName},
                                                          {sliceOutputName}));
        tensors.push_back(new TosaSerializationTensor(sliceOutputName, size, datatype, {}));
    }
    return sliceOutputName;
}
