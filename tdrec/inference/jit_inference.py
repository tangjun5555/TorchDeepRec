# -*- coding: utf-8 -*-

import os
import json
from typing import Dict, Optional, List
import torch
from fastapi import FastAPI

app_name = os.getenv("app_name")
model_file = os.getenv("model_file")

app = FastAPI(title=app_name)
model = torch.jit.load(model_file, map_location=torch.device('cpu'))
model.eval()


@app.post("/inference")
async def inference(inputs: Dict[str, Optional[int, float, List[int], List[float]]]):
    outputs = model(inputs)
    return json.dumps({"outputs": outputs})
