//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include <PrototxtConversions.hpp>

#include <boost/test/unit_test.hpp>

BOOST_AUTO_TEST_SUITE(PrototxtConversions)

BOOST_AUTO_TEST_CASE(ConvertInt32ToOctalStringTest)
{
    using armnnUtils::ConvertInt32ToOctalString;

    std::string octalString = ConvertInt32ToOctalString(1);
    BOOST_ASSERT(octalString.compare("\\\\001\\\\000\\\\000\\\\000"));

    octalString = ConvertInt32ToOctalString(256);
    BOOST_ASSERT(octalString.compare("\\\\000\\\\100\\\\000\\\\000"));

    octalString = ConvertInt32ToOctalString(65536);
    BOOST_ASSERT(octalString.compare("\\\\000\\\\000\\\\100\\\\000"));

    octalString = ConvertInt32ToOctalString(16777216);
    BOOST_ASSERT(octalString.compare("\\\\000\\\\000\\\\000\\\\100"));

    octalString = ConvertInt32ToOctalString(-1);
    BOOST_ASSERT(octalString.compare("\\\\377\\\\377\\\\377\\\\377"));

    octalString = ConvertInt32ToOctalString(-256);
    BOOST_ASSERT(octalString.compare("\\\\000\\\\377\\\\377\\\\377"));

    octalString = ConvertInt32ToOctalString(-65536);
    BOOST_ASSERT(octalString.compare("\\\\000\\\\000\\\\377\\\\377"));

    octalString = ConvertInt32ToOctalString(-16777216);
    BOOST_ASSERT(octalString.compare("\\\\000\\\\000\\\\000\\\\377"));
}

BOOST_AUTO_TEST_SUITE_END()
