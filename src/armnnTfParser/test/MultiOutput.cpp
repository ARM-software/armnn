//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include <boost/test/unit_test.hpp>
#include "armnnTfParser/ITfParser.hpp"
#include "ParserPrototxtFixture.hpp"

BOOST_AUTO_TEST_SUITE(TensorflowParser)

struct MultiOutMatchFixture : public armnnUtils::ParserPrototxtFixture<armnnTfParser::ITfParser>
{
    MultiOutMatchFixture()
    {
        m_Prototext = R"(
node {
    name: "input"
    op: "Placeholder"
    attr {
        key: "dtype"
        value {
            type: DT_FLOAT
        }
    }
    attr {
        key: "shape"
        value {
            shape {
            }
        }
    }
}
node {
    name: "softmax1"
    op: "Softmax"
    input: "input:0"
    attr {
        key: "T"
        value {
            type: DT_FLOAT
        }
    }
}
        )";
        SetupSingleInputSingleOutput({ 1, 7 }, "input", "softmax1");
    }
};

BOOST_FIXTURE_TEST_CASE(MultiOutMatch, MultiOutMatchFixture)
{
    // Note that the point of this test is to verify the parsing went well.
    // Here we make sure the softmax has really connected to the input layer.
    RunTest<2>({ 0, 0, 10000, 0, 0, 0, 0 }, { 0, 0, 1, 0, 0, 0, 0 });
}

struct MultiOutFailFixture : public armnnUtils::ParserPrototxtFixture<armnnTfParser::ITfParser>
{
    MultiOutFailFixture()
    {
        m_Prototext = R"(
node {
    name: "input"
    op: "Placeholder"
    attr {
        key: "dtype"
        value {
            type: DT_FLOAT
        }
    }
    attr {
        key: "shape"
        value {
            shape {
            }
        }
    }
}
node {
    name: "softmax1"
    op: "Softmax"
    input: "input:1"
    attr {
        key: "T"
        value {
            type: DT_FLOAT
        }
    }
}
        )";
        BOOST_CHECK_THROW(SetupSingleInputSingleOutput({ 1, 7 }, "input", "softmax1"), armnn::ParseException);
    }
};

BOOST_FIXTURE_TEST_CASE(MultiOutFail, MultiOutFailFixture)
{
    // Not running the graph because this is expected to throw an exception during parsing.
}

struct MultiOutInvalidFixture : public armnnUtils::ParserPrototxtFixture<armnnTfParser::ITfParser>
{
    MultiOutInvalidFixture()
    {
        m_Prototext = R"(
node {
    name: "input"
    op: "Placeholder"
    attr {
        key: "dtype"
        value {
            type: DT_FLOAT
        }
    }
    attr {
        key: "shape"
        value {
            shape {
            }
        }
    }
}
node {
    name: "softmax1"
    op: "Softmax"
    input: "input:-1"
    attr {
        key: "T"
        value {
            type: DT_FLOAT
        }
    }
}
        )";
        BOOST_CHECK_THROW(SetupSingleInputSingleOutput({ 1, 7 }, "input", "softmax1"), armnn::ParseException);
    }
};

BOOST_FIXTURE_TEST_CASE(MultiOutInvalid, MultiOutInvalidFixture)
{
    // Not running the graph because this is expected to throw an exception during parsing.
}


BOOST_AUTO_TEST_SUITE_END()