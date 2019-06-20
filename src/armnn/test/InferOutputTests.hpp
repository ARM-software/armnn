//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include "TestUtils.hpp"

#include <armnn/ArmNN.hpp>

#include <Graph.hpp>
#include <layers/BatchToSpaceNdLayer.hpp>
#include <layers/SpaceToDepthLayer.hpp>
#include <layers/PreluLayer.hpp>

#include <boost/algorithm/string.hpp>
#include <boost/test/unit_test.hpp>

void BatchToSpaceInferOutputShapeTest()
{
    armnn::Graph graph;

    armnn::BatchToSpaceNdDescriptor descriptor;
    descriptor.m_BlockShape = {2, 2};
    descriptor.m_Crops = {{0, 0}, {2, 0}};
    descriptor.m_DataLayout = armnn::DataLayout::NHWC;

    armnn::BatchToSpaceNdLayer* const batchToSpaceLayer =
        graph.AddLayer<armnn::BatchToSpaceNdLayer>(descriptor, "batchToSpace");

    std::vector<armnn::TensorShape> shapes;
    const std::vector<unsigned int> theDimSizes = {8, 1, 3, 1};
    armnn::TensorShape shape(4, theDimSizes.data());
    shapes.push_back(shape);

    const std::vector<unsigned int> expectedDimSizes = {2, 2, 4, 1};
    armnn::TensorShape expectedShape(4, expectedDimSizes.data());

    BOOST_CHECK(expectedShape == batchToSpaceLayer->InferOutputShapes(shapes).at(0));
}

void SpaceToDepthInferOutputShapeTest()
{
    armnn::Graph graph;

    armnn::SpaceToDepthDescriptor descriptor;
    descriptor.m_BlockSize  = 2;
    descriptor.m_DataLayout = armnn::DataLayout::NHWC;

    armnn::SpaceToDepthLayer* const spaceToDepthLayer =
        graph.AddLayer<armnn::SpaceToDepthLayer>(descriptor, "spaceToDepth");

    std::vector<armnn::TensorShape> shapes;
    const std::vector<unsigned int> dimSizes{ 1, 16, 8, 3 };
    armnn::TensorShape shape(4, dimSizes.data());
    shapes.push_back(shape);

    const std::vector<unsigned int> expectedDimSizes{ 1, 8, 4, 12 };
    armnn::TensorShape expectedShape(4, expectedDimSizes.data());

    BOOST_CHECK(expectedShape == spaceToDepthLayer->InferOutputShapes(shapes).at(0));
}

void PreluInferOutputShapeImpl(const std::vector<armnn::TensorShape>& inputShapes,
                               std::vector<armnn::TensorShape>&       outputShapes)
{
    armnn::Graph graph;
    armnn::PreluLayer* const preluLayer = graph.AddLayer<armnn::PreluLayer>("prelu");
    outputShapes = preluLayer->InferOutputShapes(inputShapes);
}

void PreluInferOutputShapeSameDimsTest()
{
    const std::vector<armnn::TensorShape> inputShapes
    {
        { 5, 1, 1, 7 }, // Input shape
        { 5, 4, 3, 1 }  // Alpha shape
    };

    const std::vector<armnn::TensorShape> expectedOutputShapes
    {
        { 5, 4, 3, 7 }  // Output shape
    };

    std::vector<armnn::TensorShape> outputShapes;
    BOOST_CHECK_NO_THROW(PreluInferOutputShapeImpl(inputShapes, outputShapes));

    BOOST_CHECK(outputShapes.size() == 1);
    BOOST_CHECK(outputShapes[0] == expectedOutputShapes[0]);
}

void PreluInferOutputShapeInputBiggerTest()
{
    const std::vector<armnn::TensorShape> inputShapes
    {
        { 4, 1, 4, 8 }, // Input shape
        { 5, 4, 1 }     // Alpha shape
    };

    const std::vector<armnn::TensorShape> expectedOutputShapes
    {
        { 4, 5, 4, 8 } // Output shape
    };

    std::vector<armnn::TensorShape> outputShapes;
    BOOST_CHECK_NO_THROW(PreluInferOutputShapeImpl(inputShapes, outputShapes));

    BOOST_CHECK(outputShapes.size() == 1);
    BOOST_CHECK(outputShapes[0] == expectedOutputShapes[0]);
}

void PreluInferOutputShapeAlphaBiggerTest()
{
    const std::vector<armnn::TensorShape> inputShapes
    {
        { 4, 1, 2 },   // Input shape
        { 5, 4, 3, 1 } // Alpha shape
    };

    const std::vector<armnn::TensorShape> expectedOutputShapes
    {
        { 5, 4, 3, 2 } // Output shape
    };

    std::vector<armnn::TensorShape> outputShapes;
    BOOST_CHECK_NO_THROW(PreluInferOutputShapeImpl(inputShapes, outputShapes));

    BOOST_CHECK(outputShapes.size() == 1);
    BOOST_CHECK(outputShapes[0] == expectedOutputShapes[0]);
}

void PreluInferOutputShapeNoMatchTest()
{
    const std::vector<armnn::TensorShape> inputShapes
    {
        { 4, 1, 2 },   // Input shape
        { 5, 4, 3, 1 } // Alpha shape
    };

    const std::vector<armnn::TensorShape> expectedOutputShapes
    {
        { 5, 7, 3, 2 } // Output shape
    };

    std::vector<armnn::TensorShape> outputShapes;
    BOOST_CHECK_NO_THROW(PreluInferOutputShapeImpl(inputShapes, outputShapes));

    BOOST_CHECK(outputShapes.size() == 1);
    BOOST_CHECK(outputShapes[0] != expectedOutputShapes[0]);
}

void CreatePreluLayerHelper(armnn::Graph& graph,
                            const armnn::TensorShape& inputShape,
                            const armnn::TensorShape& alphaShape,
                            const armnn::TensorShape& outputShape)
{
    // Creates the PReLU layer
    armnn::Layer* const preluLayer = graph.AddLayer<armnn::PreluLayer>("prelu");

    // Creates extra layers
    armnn::Layer* const input  = graph.AddLayer<armnn::InputLayer> (0, "input");
    armnn::Layer* const alpha  = graph.AddLayer<armnn::InputLayer> (1, "alpha");
    armnn::Layer* const output = graph.AddLayer<armnn::OutputLayer>(0, "output");

    // Connects up
    armnn::TensorInfo inputTensorInfo (inputShape,  armnn::DataType::Float32);
    armnn::TensorInfo alphaTensorInfo (alphaShape,  armnn::DataType::Float32);
    armnn::TensorInfo outputTensorInfo(outputShape, armnn::DataType::Float32);
    Connect(input, preluLayer,  inputTensorInfo,  0, 0);
    Connect(alpha, preluLayer,  alphaTensorInfo,  0, 1);
    Connect(preluLayer, output, outputTensorInfo, 0, 0);
}

void PreluValidateTensorShapesFromInputsMatchTest()
{
    armnn::Graph graph;

    // Creates the PReLU layer
    CreatePreluLayerHelper(graph, { 1, 4, 1, 2 }, { 5, 4, 3, 1 }, { 5, 4, 3, 2 });

    // Graph::InferTensorInfos calls Layer::ValidateTensorShapesFromInputs
    BOOST_CHECK_NO_THROW(graph.InferTensorInfos());
}

void PreluValidateTensorShapesFromInputsNoMatchTest()
{
    armnn::Graph graph;

    // Creates the PReLU layer
    CreatePreluLayerHelper(graph, { 1, 4, 1, 2 }, { 5, 4, 3, 1 }, { 5, 7, 3, 2 });

    // Graph::InferTensorInfos calls Layer::ValidateTensorShapesFromInputs
    BOOST_CHECK_THROW(graph.InferTensorInfos(), armnn::LayerValidationException);
}
