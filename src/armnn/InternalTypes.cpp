//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "InternalTypes.hpp"

#include <boost/assert.hpp>

namespace armnn
{

char const* GetLayerTypeAsCString(LayerType type)
{
    switch (type)
    {
        case LayerType::Activation: return "Activation";
        case LayerType::Addition: return "Addition";
        case LayerType::BatchNormalization: return "BatchNormalization";
        case LayerType::BatchToSpaceNd: return "BatchToSpaceNd";
        case LayerType::Concat: return "Concat";
        case LayerType::Constant: return "Constant";
        case LayerType::ConvertFp16ToFp32: return "ConvertFp16ToFp32";
        case LayerType::ConvertFp32ToFp16: return "ConvertFp32ToFp16";
        case LayerType::Convolution2d: return "Convolution2d";
        case LayerType::Debug: return "Debug";
        case LayerType::DepthwiseConvolution2d: return "DepthwiseConvolution2d";
        case LayerType::Dequantize: return "Dequantize";
        case LayerType::DetectionPostProcess: return "DetectionPostProcess";
        case LayerType::Division: return "Division";
        case LayerType::Equal: return "Equal";
        case LayerType::FakeQuantization: return "FakeQuantization";
        case LayerType::Floor: return "Floor";
        case LayerType::FullyConnected: return "FullyConnected";
        case LayerType::Gather: return "Gather";
        case LayerType::Greater: return "Greater";
        case LayerType::Input: return "Input";
        case LayerType::L2Normalization: return "L2Normalization";
        case LayerType::Lstm: return "Lstm";
        case LayerType::Maximum: return "Maximum";
        case LayerType::Mean: return "Mean";
        case LayerType::MemCopy: return "MemCopy";
        case LayerType::MemImport: return "MemImport";
        case LayerType::Merge: return "Merge";
        case LayerType::Minimum: return "Minimum";
        case LayerType::Multiplication: return "Multiplication";
        case LayerType::Normalization: return "Normalization";
        case LayerType::Output: return "Output";
        case LayerType::Pad: return "Pad";
        case LayerType::Permute: return "Permute";
        case LayerType::Pooling2d: return "Pooling2d";
        case LayerType::PreCompiled: return "PreCompiled";
        case LayerType::Prelu: return "Prelu";
        case LayerType::Quantize: return "Quantize";
        case LayerType::QuantizedLstm: return "QuantizedLstm";
        case LayerType::Reshape: return "Reshape";
        case LayerType::Resize: return "Resize";
        case LayerType::Rsqrt: return "Rsqrt";
        case LayerType::Softmax: return "Softmax";
        case LayerType::SpaceToBatchNd: return "SpaceToBatchNd";
        case LayerType::SpaceToDepth: return "SpaceToDepth";
        case LayerType::Splitter: return "Splitter";
        case LayerType::Stack: return "Stack";
        case LayerType::StridedSlice: return "StridedSlice";
        case LayerType::Subtraction: return "Subtraction";
        case LayerType::Switch: return "Switch";
        case LayerType::TransposeConvolution2d: return "TransposeConvolution2d";
        default:
            BOOST_ASSERT_MSG(false, "Unknown layer type");
            return "Unknown";
    }
}

}
