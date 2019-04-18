"""
Tools for encoding/decoding data for compression

Copyright (c) 21012 Johns Hopkins Unversity. All Rights Reserved.
"""

import numpy as np


class DeltaCompress:

    """
    Simple encode/decode based on delta compression for 1D arrays
    """

    @staticmethod
    def encode(data):
        """Compress 1D data array into a delta matrix
        """
        data = np.asarray(data)
        init = [[1, data[0]], [0, data[1]-data[0]]]
        engram = np.array(init, dtype=data.dtype)
        new_row = np.zeros(2, dtype=data.dtype)

        d = np.diff(data)
        for i in range(d.size):
            if d[i] != engram[-1,1]:
                engram = np.vstack((engram, new_row))
                engram[-1,1] = d[i]
            engram[-1,0] += 1

        return engram

    @staticmethod
    def decode(engram):
        """Extract original 1D data array from a delta matrix
        """
        N = engram[:,0].sum()
        data = np.empty((N,), dtype=engram.dtype)
        data[0] = engram[0,1]

        i = 1
        for d_i in range(1, engram.shape[0]):
            for repeat in range(engram[d_i,0]):
                data[i] = data[i-1] + engram[d_i,1]
                i += 1

        return data
