//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "InferOutputTests.hpp"

#include <test/UnitTests.hpp>

#include <boost/test/unit_test.hpp>

BOOST_AUTO_TEST_SUITE(LayerValidateOutput)

// BatchToSpace
ARMNN_SIMPLE_TEST_CASE(BatchToSpaceInferOutputShape, BatchToSpaceInferOutputShapeTest)

// SpaceToDepth
ARMNN_SIMPLE_TEST_CASE(SpaceToDepthInferOutputShape, SpaceToDepthInferOutputShapeTest)

// PReLU
ARMNN_SIMPLE_TEST_CASE(PreluInferOutputShapeSameDims,              PreluInferOutputShapeSameDimsTest)
ARMNN_SIMPLE_TEST_CASE(PreluInferOutputShapeInputBigger,           PreluInferOutputShapeInputBiggerTest)
ARMNN_SIMPLE_TEST_CASE(PreluInferOutputShapeAlphaBigger,           PreluInferOutputShapeAlphaBiggerTest)
ARMNN_SIMPLE_TEST_CASE(PreluInferOutputShapeNoMatch,               PreluInferOutputShapeNoMatchTest)
ARMNN_SIMPLE_TEST_CASE(PreluValidateTensorShapesFromInputsMatch,   PreluValidateTensorShapesFromInputsMatchTest)
ARMNN_SIMPLE_TEST_CASE(PreluValidateTensorShapesFromInputsNoMatch, PreluValidateTensorShapesFromInputsNoMatchTest)

// Stack
ARMNN_SIMPLE_TEST_CASE(StackInferOutputShapeFromInputsMatch,       StackInferOutputShapeFromInputsMatchTest)
ARMNN_SIMPLE_TEST_CASE(StackInferOutputShapeFromInputsNoMatch,     StackInferOutputShapeFromInputsNoMatchTest)
ARMNN_SIMPLE_TEST_CASE(StackValidateTensorShapesFromInputsMatch,   StackValidateTensorShapesFromInputsMatchTest)
ARMNN_SIMPLE_TEST_CASE(StackValidateTensorShapesFromInputsNoMatch, StackValidateTensorShapesFromInputsNoMatchTest)

// Convolution2D
ARMNN_SIMPLE_TEST_CASE(Convolution2dInferOutputShape, Convolution2dInferOutputShapeTest)

// DepthwiseConvolution2D
ARMNN_SIMPLE_TEST_CASE(DepthwiseConvolution2dInferOutputShape, DepthwiseConvolution2dInferOutputShapeTest)

// TransposeConvolution2D
ARMNN_SIMPLE_TEST_CASE(TransposeConvolution2dInferOutputShape, TransposeConvolution2dInferOutputShapeTest)

// QuantizedLstm
ARMNN_SIMPLE_TEST_CASE(QuantizedLstmInferOutputShape, QuantizedLstmInferOutputShapeTest)

BOOST_AUTO_TEST_SUITE_END()
