//
// Copyright © 2020 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "armnnTfParser/ITfParser.hpp"
#include "ParserPrototxtFixture.hpp"

#include <boost/test/unit_test.hpp>
#include <PrototxtConversions.hpp>

BOOST_AUTO_TEST_SUITE(TensorflowParser)

namespace
{
    std::string ConvertInt32VectorToOctalString(const std::vector<unsigned int>& data)
    {
        std::stringstream ss;
        ss << "\"";
        std::for_each(data.begin(), data.end(), [&ss](unsigned int d) {
            ss << armnnUtils::ConvertInt32ToOctalString(static_cast<int>(d));
        });
        ss << "\"";
        return ss.str();
    }
} // namespace

struct TransposeFixture : public armnnUtils::ParserPrototxtFixture<armnnTfParser::ITfParser>
{
    TransposeFixture(const armnn::TensorShape&        inputShape,
                     const std::vector<unsigned int>& permuteVectorData)
    {
        using armnnUtils::ConvertTensorShapeToString;
        armnn::TensorShape permuteVectorShape({static_cast<unsigned int>(permuteVectorData.size())});

        m_Prototext = "node {\n"
"  name: \"input\"\n"
"  op: \"Placeholder\"\n"
"  attr {\n"
"    key: \"dtype\"\n"
"    value {\n"
"      type: DT_FLOAT\n"
"    }\n"
"  }\n"
"  attr {\n"
"    key: \"shape\"\n"
"    value {\n"
"      shape {\n";
        m_Prototext.append(ConvertTensorShapeToString(inputShape));
        m_Prototext.append(
"      }\n"
"    }\n"
"  }\n"
"}\n"
"node {\n"
"  name: \"transpose/perm\"\n"
"  op: \"Const\"\n"
"  attr {\n"
"    key: \"dtype\"\n"
"    value {\n"
"      type: DT_INT32\n"
"    }\n"
"  }\n"
"  attr {\n"
"    key: \"value\"\n"
"    value {\n"
"      tensor {\n"
"        dtype: DT_INT32\n"
"        tensor_shape {\n"
        );
        m_Prototext.append(ConvertTensorShapeToString(permuteVectorShape));
        m_Prototext.append(
"        }\n"
"        tensor_content: "
        );
        m_Prototext.append(ConvertInt32VectorToOctalString(permuteVectorData) + "\n");
        m_Prototext.append(
"      }\n"
"    }\n"
"  }\n"
"}\n"
        );
        m_Prototext.append(
"node {\n"
"  name: \"output\"\n"
"  op: \"Transpose\"\n"
"  input: \"input\"\n"
"  input: \"transpose/perm\"\n"
"  attr {\n"
"    key: \"T\"\n"
"    value {\n"
"      type: DT_FLOAT\n"
"    }\n"
"  }\n"
"  attr {\n"
"    key: \"Tperm\"\n"
"    value {\n"
"      type: DT_INT32\n"
"    }\n"
"  }\n"
"}\n"
        );
        Setup({{"input", inputShape}}, {"output"});
    }
};

struct TransposeFixtureWithPermuteData : TransposeFixture
{
    TransposeFixtureWithPermuteData()
        : TransposeFixture({2, 2, 3, 4},
                           std::vector<unsigned int>({1, 3, 2, 0})) {}
};

BOOST_FIXTURE_TEST_CASE(TransposeWithPermuteData, TransposeFixtureWithPermuteData)
{
    RunTest<4>(
        {{"input", {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
        16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
        32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47}}},
        {{"output", {0, 24, 4, 28, 8, 32, 1, 25, 5, 29, 9, 33, 2, 26, 6,
        30, 10, 34, 3, 27, 7, 31, 11, 35, 12, 36, 16, 40, 20, 44, 13, 37,
        17, 41, 21, 45, 14, 38, 18, 42, 22, 46, 15, 39, 19, 43, 23, 47}}});

    BOOST_TEST((m_Parser->GetNetworkOutputBindingInfo("output").second.GetShape()
                == armnn::TensorShape({2, 4, 3, 2})));
}

struct TransposeFixtureWithoutPermuteData : TransposeFixture
{
    // In case permute data is not given, it assumes (n-1,...,0) is given
    // where n is the rank of input tensor.
    TransposeFixtureWithoutPermuteData()
        : TransposeFixture({2, 2, 3, 4},
                           std::vector<unsigned int>({3, 2, 1, 0})) {}
};

BOOST_FIXTURE_TEST_CASE(TransposeWithoutPermuteData, TransposeFixtureWithoutPermuteData)
{
    RunTest<4>(
        {{"input", {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
        16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
        32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47}}},
        {{"output", {0, 24, 12, 36, 4, 28, 16, 40, 8, 32, 20, 44, 1, 25,
        13, 37, 5, 29, 17, 41, 9, 33, 21, 45, 2, 26, 14, 38, 6, 30, 18,
        42,10, 34, 22, 46, 3, 27, 15, 39, 7, 31, 19, 43, 11, 35, 23, 47}}});

    BOOST_TEST((m_Parser->GetNetworkOutputBindingInfo("output").second.GetShape()
                == armnn::TensorShape({4, 3, 2, 2})));
}

BOOST_AUTO_TEST_SUITE_END()
