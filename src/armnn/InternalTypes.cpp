//
// Copyright © 2017 Arm Ltd. All rights reserved.
// See LICENSE file in the project root for full license information.
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
        case LayerType::Constant: return "Constant";
        case LayerType::ConvertFp16ToFp32: return "ConvertFp16ToFp32";
        case LayerType::ConvertFp32ToFp16: return "ConvertFp32ToFp16";
        case LayerType::Convolution2d: return "Convolution2d";
        case LayerType::DepthwiseConvolution2d: return "DepthwiseConvolution2d";
        case LayerType::FakeQuantization: return "FakeQuantization";
        case LayerType::Floor: return "Floor";
        case LayerType::FullyConnected: return "FullyConnected";
        case LayerType::Input: return "Input";
        case LayerType::L2Normalization: return "L2Normalization";
        case LayerType::Lstm: return "Lstm";
        case LayerType::MemCopy: return "MemCopy";
        case LayerType::Merger: return "Merger";
        case LayerType::Multiplication: return "Multiplication";
        case LayerType::Normalization: return "Normalization";
        case LayerType::Output: return "Output";
        case LayerType::Permute: return "Permute";
        case LayerType::Pooling2d: return "Pooling2d";
        case LayerType::Reshape: return "Reshape";
        case LayerType::ResizeBilinear: return "ResizeBilinear";
        case LayerType::Softmax: return "Softmax";
        case LayerType::Splitter: return "Splitter";
        default:
            BOOST_ASSERT_MSG(false, "Unknown layer type");
            return "Unknown";
    }
}

}
