//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <cstdint>

// No name mangling
extern "C"
{
int TestFunction1(int i);
}

// C++ name mangling
extern int TestFunction2(int i);

// No external linkage
int TestFunction3(int i);
