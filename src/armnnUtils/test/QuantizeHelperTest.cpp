//
// Copyright © 2019 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include <QuantizeHelper.hpp>
#include <armnn/utility/IgnoreUnused.hpp>

#include <boost/test/unit_test.hpp>

#include <vector>

BOOST_AUTO_TEST_SUITE(QuantizeHelper)

namespace
{

template<typename T>
bool IsFloatIterFunc(T iter)
{
    armnn::IgnoreUnused(iter);
    return armnnUtils::IsFloatingPointIterator<T>::value;
}

} // anonymous namespace

BOOST_AUTO_TEST_CASE(IsFloatIterFuncTest)
{
    std::vector<float> fArray;
    BOOST_TEST(IsFloatIterFunc(fArray.begin()) == true);
    BOOST_TEST(IsFloatIterFunc(fArray.cbegin()) == true);

    std::vector<double> dArray;
    BOOST_TEST(IsFloatIterFunc(dArray.begin()) == true);

    std::vector<int> iArray;
    BOOST_TEST(IsFloatIterFunc(iArray.begin()) == false);

    float floats[5];
    BOOST_TEST(IsFloatIterFunc(&floats[0]) == true);

    int ints[5];
    BOOST_TEST(IsFloatIterFunc(&ints[0]) == false);
}

BOOST_AUTO_TEST_SUITE_END()
