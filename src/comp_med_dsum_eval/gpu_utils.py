# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import subprocess


def get_free_gpus():
    gpu_stats = subprocess.check_output(
        ["nvidia-smi", "--format=csv,noheader", "--query-gpu=memory.used"], encoding='UTF-8')
    used = list(filter(lambda x: len(x) > 0, gpu_stats.split('\n')))
    return [idx for idx, x in enumerate(used) if int(x.strip().rstrip(' [MiB]')) <= 500]
