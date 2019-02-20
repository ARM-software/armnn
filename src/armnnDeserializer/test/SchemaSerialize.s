//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

.section .rodata

.global deserialize_schema_start
.global deserialize_schema_end

deserialize_schema_start:
.incbin ARMNN_SERIALIZER_SCHEMA_PATH
deserialize_schema_end:
