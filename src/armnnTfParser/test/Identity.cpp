//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include <boost/test/unit_test.hpp>
#include "armnnTfParser/ITfParser.hpp"
#include "ParserPrototxtFixture.hpp"

BOOST_AUTO_TEST_SUITE(TensorflowParser)

struct IdentitySimpleFixture : public armnnUtils::ParserPrototxtFixture<armnnTfParser::ITfParser>
{
    IdentitySimpleFixture()
    {
        m_Prototext = "node{ "
            "  name: \"Placeholder\""
            "  op: \"Placeholder\""
            "  attr {"
            "    key: \"dtype\""
            "    value {"
            "      type: DT_FLOAT"
            "    }"
            "  }"
            "  attr {"
            "    key: \"shape\""
            "    value {"
            "      shape {"
            "        unknown_rank: true"
            "      }"
            "    }"
            "  }"
            "}"
            "node {"
            "  name: \"Identity\""
            "  op: \"Identity\""
            "  input: \"Placeholder\""
            "  attr {"
            "    key: \"T\""
            "    value {"
            "      type: DT_FLOAT"
            "    }"
            "  }"
            "}";
        SetupSingleInputSingleOutput({ 4 }, "Placeholder", "Identity");
    }
};

BOOST_FIXTURE_TEST_CASE(IdentitySimple, IdentitySimpleFixture)
{
    RunTest<1>({ 1.0f, 2.0f, 3.0f, 4.0f }, { 1.0f, 2.0f, 3.0f, 4.0f });
}

struct IdentityFixture : public armnnUtils::ParserPrototxtFixture<armnnTfParser::ITfParser>
{
    IdentityFixture()
    {
        m_Prototext = "node{ "
            "  name: \"Placeholder\""
            "  op: \"Placeholder\""
            "  attr {"
            "    key: \"dtype\""
            "    value {"
            "      type: DT_FLOAT"
            "    }"
            "  }"
            "  attr {"
            "    key: \"shape\""
            "    value {"
            "      shape {"
            "        unknown_rank: true"
            "      }"
            "    }"
            "  }"
            "}"
            "node {"
            "  name: \"Identity\""
            "  op: \"Identity\""
            "  input: \"Placeholder\""
            "  attr {"
            "    key: \"T\""
            "    value {"
            "      type: DT_FLOAT"
            "    }"
            "  }"
            "}"
            "node {"
            "  name: \"Add\""
            "  op: \"Add\""
            "  input: \"Identity\""
            "  input: \"Identity\""
            "  attr {"
            "    key: \"T\""
            "    value {"
            "      type: DT_FLOAT"
            "    }"
            "  }"
            "}";
        SetupSingleInputSingleOutput({ 4 }, "Placeholder", "Add");
    }
};

BOOST_FIXTURE_TEST_CASE(ParseIdentity, IdentityFixture)
{
    RunTest<1>({ 1.0f, 2.0f, 3.0f, 4.0f }, { 2.0f, 4.0f, 6.0f, 8.0f });
}

struct IdentityChainFixture : public armnnUtils::ParserPrototxtFixture<armnnTfParser::ITfParser>
{
    IdentityChainFixture()
    {
        m_Prototext = "node{ "
            "  name: \"Placeholder\""
            "  op: \"Placeholder\""
            "  attr {"
            "    key: \"dtype\""
            "    value {"
            "      type: DT_FLOAT"
            "    }"
            "  }"
            "  attr {"
            "    key: \"shape\""
            "    value {"
            "      shape {"
            "        unknown_rank: true"
            "      }"
            "    }"
            "  }"
            "}"
            "node {"
            "  name: \"Identity\""
            "  op: \"Identity\""
            "  input: \"Placeholder\""
            "  attr {"
            "    key: \"T\""
            "    value {"
            "      type: DT_FLOAT"
            "    }"
            "  }"
            "}"
            "node {"
            "  name: \"Identity2\""
            "  op: \"Identity\""
            "  input: \"Identity\""
            "  attr {"
            "    key: \"T\""
            "    value {"
            "      type: DT_FLOAT"
            "    }"
            "  }"
            "}";
        SetupSingleInputSingleOutput({ 4 }, "Placeholder", "Identity2");
    }
};

BOOST_FIXTURE_TEST_CASE(IdentityChain, IdentityChainFixture)
{
    RunTest<1>({ 1.0f, 2.0f, 3.0f, 4.0f }, { 1.0f, 2.0f, 3.0f, 4.0f });
}

BOOST_AUTO_TEST_SUITE_END()
