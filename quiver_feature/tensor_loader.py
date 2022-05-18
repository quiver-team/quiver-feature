import torch
import qvf
import torch.serialization as se
from torch.serialization import *


class _open_zipfile_reader(torch.serialization._opener):
    def __init__(self, name_or_buffer) -> None:
        super(_open_zipfile_reader, self).__init__(qvf.SharedTensorLoader(name_or_buffer))


def shared_load(f, map_location=None, pickle_module=pickle, **pickle_load_args):
    se._check_dill_version(pickle_module)

    if 'encoding' not in pickle_load_args.keys():
        pickle_load_args['encoding'] = 'utf-8'

    with se._open_file_like(f, 'rb') as opened_file:
        if se._is_zipfile(opened_file):
            # The zipfile reader is going to advance the current file position.
            # If we want to actually tail call to torch.jit.load, we need to
            # reset back to the original position.
            orig_position = opened_file.tell()
            with _open_zipfile_reader(opened_file) as opened_zipfile:
                if se._is_torchscript_zip(opened_zipfile):
                    warnings.warn("'torch.load' received a zip file that looks like a TorchScript archive"
                                  " dispatching to 'torch.jit.load' (call 'torch.jit.load' directly to"
                                  " silence this warning)", UserWarning)
                    opened_file.seek(orig_position)
                    return torch.jit.load(opened_file)
                return se._load(opened_zipfile, map_location, pickle_module, **pickle_load_args)
        return se._legacy_load(opened_file, map_location, pickle_module, **pickle_load_args)
