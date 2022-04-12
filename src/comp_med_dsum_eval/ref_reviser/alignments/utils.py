# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

def keep_sent(improvement):
    """
    Whether or not to keep the context sentence for particular reference sentence based on how much it
    improves the coverage
    """
    if type(improvement) == int and improvement == -1:
        return True
    return improvement[0] >= 0.01 or improvement[1] >= 0.05