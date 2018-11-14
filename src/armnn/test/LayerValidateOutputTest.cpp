//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//
#include <armnn/ArmNN.hpp>

#include <boost/algorithm/string.hpp>
#include <boost/test/unit_test.hpp>
#include <layers/BatchToSpaceNdLayer.hpp>
#include <Graph.hpp>


BOOST_AUTO_TEST_SUITE(LayerValidateOutput)

BOOST_AUTO_TEST_CASE(TestBatchToSpaceInferOutputShape)
{
    armnn::Graph graph;

    armnn::BatchToSpaceNdDescriptor descriptor;
    std::vector<unsigned int> theBlockShape = {2, 2};
    descriptor.m_BlockShape = theBlockShape;
    descriptor.m_DataLayout = armnn::DataLayout::NHWC;

    armnn::BatchToSpaceNdLayer* const batchToSpaceLayer =
        graph.AddLayer<armnn::BatchToSpaceNdLayer>(descriptor, "batchToSpace");

    std::vector<armnn::TensorShape> shapes;
    const std::vector<unsigned int> theDimSizes = {4, 2, 2, 1};
    armnn::TensorShape shape(4, theDimSizes.data());
    shapes.push_back(shape);

    const std::vector<unsigned int> expectedDimSizes = {1, 4, 4, 1};
    armnn::TensorShape expectedShape(4, expectedDimSizes.data());

    BOOST_CHECK(expectedShape == batchToSpaceLayer->InferOutputShapes(shapes).at(0));
}

BOOST_AUTO_TEST_SUITE_END()