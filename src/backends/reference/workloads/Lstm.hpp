//
// Copyright © 2021 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <armnn/TypesUtils.hpp>
#include <armnn/backends/WorkloadData.hpp>

#include "Encoders.hpp"
#include "Decoders.hpp"

namespace armnn
{

void LstmImpl(const LstmDescriptor& descriptor,
              const TensorInfo& inputInfo,
              const TensorInfo& outputInfo,
              const TensorShape& inputToOutputWeightsShape,
              const TensorShape& recurrentToOutputWeightsShape,
              std::unique_ptr<Decoder<float>>& inputData,
              std::unique_ptr<Decoder<float>>& outputStateIn,
              std::unique_ptr<Decoder<float>>& cellStateIn,
              std::unique_ptr<Encoder<float>>& outputStateOut,
              std::unique_ptr<Encoder<float>>& cellStateOut,
              std::unique_ptr<Encoder<float>>& output,
              std::unique_ptr<Decoder<float>>& cellStateOutDecoder,
              std::unique_ptr<Decoder<float>>& outputDecoder,
              std::unique_ptr<Decoder<float>>& inputToInputWeightsTensor,
              std::unique_ptr<Decoder<float>>& inputToForgetWeightsTensor,
              std::unique_ptr<Decoder<float>>& inputToCellWeightsTensor,
              std::unique_ptr<Decoder<float>>& inputToOutputWeightsTensor,
              std::unique_ptr<Decoder<float>>& recurrentToInputWeightsTensor,
              std::unique_ptr<Decoder<float>>& recurrentToForgetWeightsTensor,
              std::unique_ptr<Decoder<float>>& recurrentToCellWeightsTensor,
              std::unique_ptr<Decoder<float>>& recurrentToOutputWeightsTensor,
              std::unique_ptr<Decoder<float>>& cellToInputWeightsTensor,
              std::unique_ptr<Decoder<float>>& cellToForgetWeightsTensor,
              std::unique_ptr<Decoder<float>>& cellToOutputWeightsTensor,
              std::unique_ptr<Decoder<float>>& inputGateBiasTensor,
              std::unique_ptr<Decoder<float>>& forgetGateBiasTensor,
              std::unique_ptr<Decoder<float>>& cellBiasTensor,
              std::unique_ptr<Decoder<float>>& outputGateBiasTensor,
              std::unique_ptr<Decoder<float>>& projectionWeightsTensor,
              std::unique_ptr<Decoder<float>>& projectionBiasTensor,
              std::unique_ptr<Decoder<float>>& inputLayerNormWeights,
              std::unique_ptr<Decoder<float>>& forgetLayerNormWeights,
              std::unique_ptr<Decoder<float>>& cellLayerNormWeights,
              std::unique_ptr<Decoder<float>>& outputLayerNormWeights,
              std::unique_ptr<Encoder<float>>& inputGateScratch,
              std::unique_ptr<Encoder<float>>& cellScratch,
              std::unique_ptr<Encoder<float>>& forgetGateScratch,
              std::unique_ptr<Encoder<float>>& outputGateScratch,
              std::unique_ptr<Decoder<float>>& inputGateScratchDecoder,
              std::unique_ptr<Decoder<float>>& cellScratchDecoder,
              std::unique_ptr<Decoder<float>>& forgetGateScratchDecoder,
              std::unique_ptr<Decoder<float>>& outputGateScratchDecoder,
              float layerNormEpsilon);

} //namespace armnn
