import torch
import time

import os
import quiver_feature
from quiver_feature import Range
from quiver_feature import DistHelper

MASTER_ADDR = '155.198.152.17'
MASTER_PORT = 5678

MY_SERVER_RANK = 1
SERVER_WORLD_SIZE = 2


dist_helper = DistHelper(MASTER_ADDR, MASTER_PORT, SERVER_WORLD_SIZE, MY_SERVER_RANK)
LOCAL_RANGE = Range(MY_SERVER_RANK * 100, MY_SERVER_RANK * 200)
tensor_endpoints = dist_helper.exchange_tensor_endpoints_info(LOCAL_RANGE)

print(f"Check TensorEndPoint ", tensor_endpoints)
time.sleep(MY_SERVER_RANK * 5 + 1)
print(f"Rank {MY_SERVER_RANK} Finished, Begin To Sync")
dist_helper.sync_all()
print(f"Rank {MY_SERVER_RANK} Finished, Bye Bye")
