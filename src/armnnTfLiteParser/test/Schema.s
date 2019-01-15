//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

.section .rodata

.global tflite_schema_start
.global tflite_schema_end

tflite_schema_start:
.incbin ARMNN_TF_LITE_SCHEMA_PATH
tflite_schema_end:
