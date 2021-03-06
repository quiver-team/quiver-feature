from multiprocessing.reduction import ForkingPickler
import qvf
from ..local_tensor_pgas import LocalTensorPGAS

def rebuild_qvf_pipeparam(ipc_handle):

    pipe_param = qvf.PipeParam()
    pipe_param.set_param_vec(ipc_handle)
    return pipe_param

def reduce_qvf_pipeparam(pipe_param):
    param_vec = pipe_param.get_param_vec()
    return(rebuild_qvf_pipeparam, (param_vec, ))


def rebuild_qvf_comendpoint(ipc_handle):

    com_endpoint = qvf.ComEndPoint(ipc_handle[0], ipc_handle[1], ipc_handle[2])
    return com_endpoint

def reduce_qvf_comendpoint(com_endpoint):
    param_vec = (com_endpoint.rank(), com_endpoint.address(), com_endpoint.port())
    return (rebuild_qvf_comendpoint, (param_vec, ))

def rebuild_localtensorpgas(ipc_handle):

    feature = LocalTensorPGAS.lazy_from_ipc_handle(ipc_handle)
    return feature


def reduce_localtensorpgas(feature):

    ipc_handle = feature.share_ipc()
    return (rebuild_localtensorpgas, (ipc_handle, ))

def init_reductions():
    ForkingPickler.register(qvf.PipeParam, reduce_qvf_pipeparam)
    ForkingPickler.register(qvf.ComEndPoint, reduce_qvf_comendpoint)
    ForkingPickler.register(LocalTensorPGAS, reduce_localtensorpgas)