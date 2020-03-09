//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//
#include <boost/test/unit_test.hpp>

#include <armnn/Optional.hpp>
#include <string>

#include <armnn/utility/IgnoreUnused.hpp>

namespace
{

void PassStringRef(armnn::Optional<std::string&> value)
{
    armnn::IgnoreUnused(value);
}

void PassStringRefWithDefault(armnn::Optional<std::string&> value = armnn::EmptyOptional())
{
    armnn::IgnoreUnused(value);
}

} // namespace <anonymous>

BOOST_AUTO_TEST_SUITE(OptionalTests)

BOOST_AUTO_TEST_CASE(SimpleStringTests)
{
    armnn::Optional<std::string> optionalString;
    BOOST_TEST(static_cast<bool>(optionalString) == false);
    BOOST_TEST(optionalString.has_value() == false);
    BOOST_TEST((optionalString == armnn::Optional<std::string>()));

    optionalString = std::string("Hello World");
    BOOST_TEST(static_cast<bool>(optionalString) == true);
    BOOST_TEST(optionalString.has_value() == true);
    BOOST_TEST(optionalString.value() == "Hello World");
    BOOST_TEST((optionalString == armnn::Optional<std::string>("Hello World")));

    armnn::Optional<std::string> otherString;
    otherString = optionalString;
    BOOST_TEST(static_cast<bool>(otherString) == true);
    BOOST_TEST(otherString.value() == "Hello World");

    optionalString.reset();
    BOOST_TEST(static_cast<bool>(optionalString) == false);
    BOOST_TEST(optionalString.has_value() == false);

    const std::string stringValue("Hello World");
    armnn::Optional<std::string> optionalString2(stringValue);
    BOOST_TEST(static_cast<bool>(optionalString2) == true);
    BOOST_TEST(optionalString2.has_value() == true);
    BOOST_TEST(optionalString2.value() == "Hello World");

    armnn::Optional<std::string> optionalString3(std::move(optionalString2));
    BOOST_TEST(static_cast<bool>(optionalString3) == true);
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

BOOST_AUTO_TEST_CASE(SimpleIntTests)
{
    const int intValue = 123;

    armnn::Optional<int> optionalInt;
    BOOST_TEST(static_cast<bool>(optionalInt) == false);
    BOOST_TEST(optionalInt.has_value() == false);
    BOOST_TEST((optionalInt == armnn::Optional<int>()));

    optionalInt = intValue;
    BOOST_TEST(static_cast<bool>(optionalInt) == true);
    BOOST_TEST(optionalInt.has_value() == true);
    BOOST_TEST(optionalInt.value() == intValue);
    BOOST_TEST((optionalInt == armnn::Optional<int>(intValue)));

    armnn::Optional<int> otherOptionalInt;
    otherOptionalInt = optionalInt;
    BOOST_TEST(static_cast<bool>(otherOptionalInt) == true);
    BOOST_TEST(otherOptionalInt.value() == intValue);
}

BOOST_AUTO_TEST_CASE(ObjectConstructedInPlaceTests)
{
    struct SimpleObject
    {
        public:
            SimpleObject(const std::string& name, int value)
                : m_Name(name)
                , m_Value(value)
            {}

            bool operator ==(const SimpleObject& other)
            {
                return m_Name  == other.m_Name &&
                       m_Value == other.m_Value;
            }

        private:
            std::string m_Name;
            int         m_Value;
    };

    std::string objectName("SimpleObject");
    int objectValue = 1;
    SimpleObject referenceObject(objectName, objectValue);

    // Use MakeOptional
    armnn::Optional<SimpleObject> optionalObject1 = armnn::MakeOptional<SimpleObject>(objectName, objectValue);
    BOOST_CHECK(static_cast<bool>(optionalObject1) == true);
    BOOST_CHECK(optionalObject1.has_value() == true);
    BOOST_CHECK(optionalObject1.value() == referenceObject);

    // Call in-place constructor directly
    armnn::Optional<SimpleObject> optionalObject2(CONSTRUCT_IN_PLACE, objectName, objectValue);
    BOOST_CHECK(static_cast<bool>(optionalObject1) == true);
    BOOST_CHECK(optionalObject1.has_value() == true);
    BOOST_CHECK(optionalObject1.value() == referenceObject);
}

BOOST_AUTO_TEST_SUITE_END()
