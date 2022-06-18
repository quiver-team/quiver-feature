import torch
import qvf

pipe_param = qvf.PipeParam(1, 1, 1, 1)
print(f"ParamVec: {pipe_param.get_param_vec()}")

pipe_param2 = qvf.PipeParam()
pipe_param2.set_param_vec(pipe_param.get_param_vec())
print(f"ParamVec: {pipe_param2.get_param_vec()}")

