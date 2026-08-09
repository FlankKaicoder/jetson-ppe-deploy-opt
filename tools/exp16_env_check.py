#!/usr/bin/env python3
import json

import numpy
import onnx
import onnx_graphsurgeon
import google.protobuf


print(json.dumps({
    "status": "PASS",
    "numpy": numpy.__version__,
    "onnx": onnx.__version__,
    "onnx_graphsurgeon": onnx_graphsurgeon.__version__,
    "protobuf": google.protobuf.__version__,
}, indent=2))
