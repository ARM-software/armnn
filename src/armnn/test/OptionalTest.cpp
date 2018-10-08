//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//
#include <boost/test/unit_test.hpp>

#include <armnn/Optional.hpp>
#include <boost/optional.hpp>
#include <string>

namespace
{

void PassStringRef(armnn::Optional<std::string&> value)
{
}

void PassStringRefWithDefault(armnn::Optional<std::string&> value = armnn::EmptyOptional())
{
}

void BoostCompatibilityTester(const armnn::Optional<std::string>& optionalString,
                              bool hasValue,
                              const std::string& expectedValue)
{
    BOOST_TEST(optionalString.has_value() == hasValue);
    if (hasValue)
    {
        BOOST_TEST(optionalString.value() == expectedValue);
    }
}

}


BOOST_AUTO_TEST_SUITE(OptionalTests)

BOOST_AUTO_TEST_CASE(SimpleStringTests)
{
    armnn::Optional<std::string> optionalString;
    BOOST_TEST(optionalString == false);
    BOOST_TEST(optionalString.has_value() == false);

    optionalString = std::string("Hello World");
    BOOST_TEST(optionalString == true);
    BOOST_TEST(optionalString.has_value() == true);
    BOOST_TEST(optionalString.value() == "Hello World");

    armnn::Optional<std::string> otherString;
    otherString = optionalString;
    BOOST_TEST(otherString == true);
    BOOST_TEST(otherString.value() == "Hello World");

    optionalString.reset();
    BOOST_TEST(optionalString == false);
    BOOST_TEST(optionalString.has_value() == false);

    const std::string stringValue("Hello World");
    armnn::Optional<std::string> optionalString2(stringValue);
    BOOST_TEST(optionalString2 == true);
    BOOST_TEST(optionalString2.has_value() == true);
    BOOST_TEST(optionalString2.value() == "Hello World");

    armnn::Optional<std::string> optionalString3(std::move(optionalString2));
    BOOST_TEST(optionalString3 == true);
    BOOST_TEST(optionalString3.has_value() == true);
    BOOST_TEST(optionalString3.value() == "Hello World");
}


BOOST_AUTO_TEST_CASE(StringRefTests)
{
    armnn::Optional<std::string&> optionalStringRef{armnn::EmptyOptional()};
    BOOST_TEST(optionalStringRef.has_value() == false);

    PassStringRef(optionalStringRef);
    PassStringRefWithDefault();

    armnn::Optional<std::string&> optionalStringRef2 = optionalStringRef;

    std::string helloWorld("Hello World");

    std::string& helloWorldRef = helloWorld;
    armnn::Optional<std::string&> optionalHelloRef = helloWorldRef;
    BOOST_TEST(optionalHelloRef.has_value() == true);
    BOOST_TEST(optionalHelloRef.value() == "Hello World");

    armnn::Optional<std::string&> optionalHelloRef2 = helloWorld;
    BOOST_TEST(optionalHelloRef2.has_value() == true);
    BOOST_TEST(optionalHelloRef2.value() == "Hello World");

    armnn::Optional<std::string&> optionalHelloRef3{helloWorldRef};
    BOOST_TEST(optionalHelloRef3.has_value() == true);
    BOOST_TEST(optionalHelloRef3.value() == "Hello World");

    armnn::Optional<std::string&> optionalHelloRef4{helloWorld};
    BOOST_TEST(optionalHelloRef4.has_value() == true);
    BOOST_TEST(optionalHelloRef4.value() == "Hello World");

    // modify through the optional reference
    optionalHelloRef4.value().assign("Long Other String");
    BOOST_TEST(helloWorld == "Long Other String");
    BOOST_TEST(optionalHelloRef.value() == "Long Other String");
    BOOST_TEST(optionalHelloRef2.value() == "Long Other String");
    BOOST_TEST(optionalHelloRef3.value() == "Long Other String");
}

BOOST_AUTO_TEST_CASE(BoostCompatibilityTests)
{
    // sanity checks
    BoostCompatibilityTester(armnn::Optional<std::string>(), false, "");
    BoostCompatibilityTester(armnn::Optional<std::string>("Hello World"), true, "Hello World");

    // verify boost signature selector
    BOOST_TEST(armnn::CheckBoostOptionalSignature<boost::optional<std::string>>::Result() == true);
    BOOST_TEST(armnn::CheckBoostOptionalSignature<armnn::Optional<std::string>>::Result() == false);

    // the real thing is to see that we can pass a boost::optional in place
    // of an ArmNN Optional
    boost::optional<std::string> empty;
    boost::optional<std::string> helloWorld("Hello World");

    BoostCompatibilityTester(empty, false, "");
    BoostCompatibilityTester(helloWorld, true, "Hello World");

    BoostCompatibilityTester(boost::optional<std::string>(), false, "");
    BoostCompatibilityTester(boost::optional<std::string>("Hello World"), true, "Hello World");
}

BOOST_AUTO_TEST_CASE(SimpleIntTests)
{
    const int intValue = 123;

    armnn::Optional<int> optionalInt;
    BOOST_TEST(optionalInt == false);
    BOOST_TEST(optionalInt.has_value() == false);

    optionalInt = intValue;
    BOOST_TEST(optionalInt == true);
    BOOST_TEST(optionalInt.has_value() == true);
    BOOST_TEST(optionalInt.value() == intValue);

    armnn::Optional<int> otherOptionalInt;
    otherOptionalInt = optionalInt;
    BOOST_TEST(otherOptionalInt == true);
    BOOST_TEST(otherOptionalInt.value() == intValue);
}

BOOST_AUTO_TEST_SUITE_END()
