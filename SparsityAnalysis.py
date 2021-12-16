import numpy as np

def extract_patterns(weight):
    """
    return a list of patterns in this weight
    :param weight:
    :return:
    """
    patterns = []
    kernel_dist = np.zeros((weight.shape[0], weight.shape[1]))
    for oc in range(weight.shape[0]):
        for ic in range(weight.shape[1]):
            kernel = weight[oc, ic].reshape(-1)
            if kernel.sum() != 0:
                kernel_dist[oc, ic] += 1
                kernel[kernel != 0] = 1
                curr_pattern = kernel.astype(np.int8).tolist()
                if curr_pattern not in patterns:
                    patterns.append(curr_pattern)
    return patterns

class SparseConvArrays(object):
    """tensor object in FKW format."""

    def __init__(self, filter, patterns):
        """Construct a FKW format matrix.

        Parameters
        ----------
        filter : numpy.ndarray
            The filter in format OIHW

        patterns: numpy.ndarray
            The patterns

        ctx: tvmContext
            The corresponding context.

        shape : tuple of int
            The shape of the array
        """
        # Get weight pattern types from pre-configured pattern inventories
        def parse_pattern(kernel, patterns):
            input_pattern = kernel.reshape(-1).copy()
            input_pattern[np.where(input_pattern != 0)] = 1
            for i, pattern in enumerate(patterns):
                if np.array_equal(pattern.astype('int32').reshape(-1), input_pattern.astype('int32')):
                    return i + 1
            raise RuntimeError(
                "filter contains unsupported pattern types {}, please check pattern inventory".format(
                    kernel.reshape(-1))
            )

        def convert_pattern_set_to_indexes(pattern_set):
            dict = {
                0: [0, 0],
                1: [0, 1],
                2: [0, 2],
                3: [1, 0],
                4: [1, 1],
                5: [1, 2],
                6: [2, 0],
                7: [2, 1],
                8: [2, 2],
            }
            patterns = []
            for pattern_list in pattern_set:
                assert len(pattern_list) == 9
                assert sum(pattern_list) == 4
                pattern = []
                for i in range(len(pattern_list)):
                    if pattern_list[i] == 1:
                        pattern.append(dict[i])
                patterns.append(pattern)
            return patterns


        self.offset = None
        self.reorder = None
        self.index = None
        self.stride = None
        self.weight = None


        # for te.compute output tensor order
        self.order = None
        self.pattern_set_size = patterns.shape[0]
        self.nonzero_per_kernel = 4

        if isinstance(filter, np.ndarray):
            # start filter and weight reorder
            source_array = filter.copy()
            oc, ic, h, w = source_array.shape
            # orig filters IR
            filters_ir = np.zeros((oc, ic), dtype='int32')
            for i in range(oc):
                for j in range(ic):
                    num_nonzero = np.count_nonzero(source_array[i, j])
                    if num_nonzero:
                        # parse pattern type
                        filters_ir[i, j] = parse_pattern(source_array[i, j], patterns)
            # print(filters_ir)
            num_per_filter = np.zeros((oc,), dtype='int32')
            # orig weight index per filter
            idx_per_filter = np.zeros((oc, ic), dtype='int32')
            # filter IR after reordered
            filters_ir_reorder = np.zeros_like(filters_ir)
            # 1. get filter IR
            for i in range(oc):
                nzero_index = np.nonzero(filters_ir[i])[0]
                # weight reorder
                sorted_ind = sorted(nzero_index, key=lambda k: filters_ir[i][k])
                num_per_filter[i] = len(nzero_index)
                idx_per_filter[i, :len(nzero_index)] = sorted_ind
            # 2. filter reorder (according to weight num per filter)
            self.reorder = np.array(sorted(range(oc), key=lambda k: num_per_filter[k]), dtype='int32')
            # print('reorder: ', self.reorder)
            # get correct order (for te.compute usage)
            self.order = np.argsort(self.reorder).astype('int32')
            self.offset = np.append(0, np.cumsum(num_per_filter[self.reorder])).astype('int32')
            # print('offset: ', self.offset)
            self.index = np.empty(0, dtype='int32')
            for i in range(oc):
                idx = self.reorder[i]
                self.index = np.append(self.index, idx_per_filter[idx, :num_per_filter[idx]])
                filters_ir_reorder[i, :num_per_filter[idx]] = \
                    filters_ir[idx, idx_per_filter[idx, :num_per_filter[idx]]]
            # print('index: ', self.index)
            self.stride = np.zeros((self.pattern_set_size + 1) * oc).astype('int32')
            stride_index = 0
            for i in range(oc):
                tmp_pattern_num = np.zeros(self.pattern_set_size)
                for j in range(ic):
                    pattern_type = filters_ir_reorder[i, j]
                    if pattern_type > 0:
                        tmp_pattern_num[pattern_type - 1] += 1
                self.stride[stride_index] = 0
                stride_index += 1
                for j in range(len(tmp_pattern_num)):
                    self.stride[stride_index] = tmp_pattern_num[j] + self.stride[stride_index - 1]
                    stride_index += 1
            # print('stride: ', self.stride)
            # print('filter: ', filters_ir)
            # print('reorder: ', self.reorder)
            # print('filter reorder: ', filters_ir_reorder)
            # fill self.weight
            self.weight = np.zeros(self.nonzero_per_kernel * self.index.shape[0]).astype('float32')

            # iterate self.index array
            index_cnt = 0
            for i in range(oc):
                filter_idx = self.reorder[i]
                for j in range(self.offset[i + 1] - self.offset[i]):
                    weight_idx = np.nonzero(source_array[filter_idx, self.index[index_cnt]])
                    self.weight[(index_cnt * self.nonzero_per_kernel):
                                (index_cnt * self.nonzero_per_kernel + self.nonzero_per_kernel)] = \
                        source_array[filter_idx, self.index[index_cnt]][weight_idx]
                    index_cnt += 1
            # print('weight: ', self.weight)


        self.ptset = np.zeros((self.pattern_set_size, self.nonzero_per_kernel, 2)).astype('int32')
        pattern_format = convert_pattern_set_to_indexes(patterns)
        for pi, pattern in enumerate(pattern_format):
            for v in range(self.nonzero_per_kernel):
                self.ptset[pi, v, 0] = pattern[v][0]
                self.ptset[pi, v, 1] = pattern[v][1]