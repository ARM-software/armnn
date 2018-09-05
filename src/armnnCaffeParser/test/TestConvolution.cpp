//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//
#include <boost/test/unit_test.hpp>
#include "armnnCaffeParser/ICaffeParser.hpp"
#include "ParserPrototxtFixture.hpp"
#include <sstream>
#include <initializer_list>

namespace
{

template <typename T>
std::string TaggedSequence(const std::string & tag, const std::initializer_list<T> & data)
{
    bool first = true;
    std::stringstream ss;
    for (auto && d : data)
    {
        if (!first)
        {
            ss << " , ";
        }
        else
        {
            first = false;
        }
        ss << " " << tag << " : " << d << " ";
    }
    return ss.str();
}

template <typename T>
std::string TaggedSequence(const std::string & tag, T data, unsigned int n)
{
    std::stringstream ss;
    for (unsigned int i=0; i<n; ++i)
    {
        if (i>0)
        {
            ss << " , ";
        }
        ss << " " << tag << " : " << data << " ";
    }
    return ss.str();
}

} // namespace <anonymous>

BOOST_AUTO_TEST_SUITE(CaffeParser)

struct ConvolutionFixture : public armnnUtils::ParserPrototxtFixture<armnnCaffeParser::ICaffeParser>
{
    ConvolutionFixture(const std::initializer_list<unsigned int> & inputDims,
                       const std::initializer_list<float> & filterData,
                       unsigned int kernelSize,
                       unsigned int numOutput=1,
                       unsigned int group=1)
    {
        m_Prototext = R"(
            name: "ConvolutionTest"
            layer {
                name: "input1"
                type: "Input"
                top: "input1"
                input_param { shape: { )" + TaggedSequence("dim", inputDims) + R"( } }
            }
            layer {
                name: "conv1"
                type: "Convolution"
                bottom: "input1"
                top: "conv1"
                blobs: { )" + TaggedSequence("data", filterData) + R"( }
                blobs: { )" + TaggedSequence("data", 0, numOutput) + R"( }
                convolution_param {
                    num_output: )" + std::to_string(numOutput) + R"(
                    pad: 0
                    kernel_size: )" +  std::to_string(kernelSize) + R"(
                    stride: 1
                    group: )" +  std::to_string(group) + R"(
                }
            }
        )";
        SetupSingleInputSingleOutput("input1", "conv1");
    }
};

struct SimpleConvolutionFixture : public ConvolutionFixture
{
    SimpleConvolutionFixture()
    : ConvolutionFixture( {1, 1, 2, 2}, {1.0f, 1.0f, 1.0f, 1.0f}, 2)
    {
    }
};

BOOST_FIXTURE_TEST_CASE(SimpleConvolution, SimpleConvolutionFixture)
{
    RunTest<4>({ 1, 3, 5, 7 }, { 16 });
}

struct GroupConvolutionFixture : public ConvolutionFixture
{
    GroupConvolutionFixture()
    : ConvolutionFixture(
        {1, 2, 2, 2},
        {
            1.0f, 1.0f, 1.0f, 1.0f, // filter for channel #0
            2.0f, 2.0f, 2.0f, 2.0f  // filter for channel #1
        },
        2, // kernel size is 2x2
        2, // number of output channels is 2
        2) // number of groups (separate filters)
    {
    }
};

BOOST_FIXTURE_TEST_CASE(GroupConvolution, GroupConvolutionFixture)
{
    RunTest<4>(
        {
            1, 2, 3, 4, // input channel #0
            5, 6, 7, 8, // input channel #1
        },
        {
            10, // convolution result for channel #0 applying filter #0
            52  // same for channel #1 and filter #1
        }
    );
}


BOOST_AUTO_TEST_SUITE_END()