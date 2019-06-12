//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//
#include <armnn/ArmNN.hpp>

#include <Graph.hpp>
#include <layers/BatchToSpaceNdLayer.hpp>
#include <layers/SpaceToDepthLayer.hpp>

#include <boost/algorithm/string.hpp>
#include <boost/test/unit_test.hpp>

BOOST_AUTO_TEST_SUITE(LayerValidateOutput)

BOOST_AUTO_TEST_CASE(TestBatchToSpaceInferOutputShape)
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

BOOST_AUTO_TEST_CASE(TestSpaceToDepthInferOutputShape)
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

BOOST_AUTO_TEST_CASE(TestPreluInferOutputShape)
{
    armnn::Graph graph;

    armnn::PreluLayer* const preluLayer = graph.AddLayer<armnn::PreluLayer>("prelu");

    std::vector<armnn::TensorShape> inputShapes
    {
        { 4, 1, 2 },  // Input shape
        { 5, 4, 3, 1} // Alpha shape
    };

    const std::vector<armnn::TensorShape> expectedOutputShapes
    {
        { 5, 4, 3, 2 } // Output shape
    };

    const std::vector<armnn::TensorShape> outputShapes = preluLayer->InferOutputShapes(inputShapes);

    BOOST_CHECK(outputShapes.size() == 1);
    BOOST_CHECK(outputShapes[0] == expectedOutputShapes[0]);
}

BOOST_AUTO_TEST_SUITE_END()
