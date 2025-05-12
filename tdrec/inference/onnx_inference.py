# -*- coding: utf-8 -*-

import os
import json
from typing import Dict, Optional, List
from fastapi import FastAPI
import onnxruntime as ort

app_name = os.getenv("app_name")
model_file = os.getenv("model_file")
print(f"app_name:{app_name}, model_file:{model_file}")

app = FastAPI(title=app_name)
model = ort.InferenceSession(model_file)


@app.post("/inference")
async def inference(inputs: Dict[str, Optional[int, float, List[int], List[float]]]):
    outputs = model.run("outputs", inputs)
    return json.dumps({"outputs": outputs})
