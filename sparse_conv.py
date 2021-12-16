import numpy as np
import time
from scipy import signal
import pycuda.autoinit
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
from SparsityAnalysis import extract_patterns, SparseConvArrays
import torch.nn as nn
import torch
from conv_naive import Convolution
from tqdm import tqdm


class SparseConvolution:
    def __init__(self):
        """
        Attributes for instance of EncoderDecoder module
        """
        self.mod = self.getSourceModule()
        pass

    def getSourceModule(self):
        # kernel code wrapper
        kernelwrapper = """

            #define BLOCK_WIDTH 4
            #define CONSTANT_SIZE 16384

            __constant__ float Weight[CONSTANT_SIZE];


            __global__ void sparse_conv_naive(float *data, int *offset, int *reorder, int *index, int *stride, float *weight, float *ptset, float *out,
            int bn, int IC, int IH, int IW, int OC, int OH, int OW, int ptset_size, int non_zero, int mask_width, int step) {

                int filter_idx = blockIdx.x * blockDim.x + threadIdx.x;
                int row = blockIdx.y * blockDim.y + threadIdx.y;
                int col = blockIdx.z * blockDim.z + threadIdx.z;
                int output_index = bn * OC * OH * OW + reorder[filter_idx] * OH * OW + row * OW + col;

                //int tx = threadIdx.x;
                //int ty = threadIdx.y;
                //int tz = threadIdx.z;

                if(filter_idx < OC){

                    int topleft_col = col * step - ((mask_width - 1) / 2);
                    int topleft_row = row * step - ((mask_width - 1) / 2);

                    int oc_stride_start = filter_idx * (ptset_size + 1);
                    float output = 0;

                    for(int pi = 0; pi < ptset_size; pi++){

                        int stride_index = oc_stride_start + pi;
                        int nkernels = stride[stride_index + 1] - stride[stride_index];
                        int kernel_index_start = offset[filter_idx] + stride[stride_index];

                        for(int k = 0; k < nkernels; k++){

                            int topleft_data_index = bn * IC * IH * IW + index[kernel_index_start + k] * IH * IW + topleft_row * IW + topleft_col;


                            for(int v = 0; v < non_zero; v++){

                                int cur_row = topleft_row + ptset[pi * non_zero * 2 + v * 2 + 0];
                                int cur_col = topleft_col + ptset[pi * non_zero * 2 + v * 2 + 1];
                                int cur_index = topleft_data_index + ptset[pi * non_zero * 2 + v * 2 + 0] * IW + ptset[pi * non_zero * 2 + v * 2 + 1];

                                //out[output_index] = ptset[pi * non_zero * 2 + v * 2 + 0];

                                if(cur_row >= 0 && cur_row < IH && cur_col >=0 && cur_col < IW){
                                    output += data[cur_index] * weight[non_zero * (kernel_index_start + k) + v];

                                }
                            }
                        }

                    }
                    out[output_index] = output;
                }
			}



			 __global__ void sparse_conv_shared(float *data, int *offset, int *reorder, int *index, int *stride, float *weight, float *ptset, float *out,
            int bn, int IC, int IH, int IW, int OC, int OH, int OW, int ptset_size, int non_zero, int mask_width, int step) {

                int filter_idx = blockIdx.x * blockDim.x + threadIdx.x;
                int row = blockIdx.y * blockDim.y + threadIdx.y;
                int col =  blockIdx.z * blockDim.z + threadIdx.z;
                int output_index = bn * OC * OH * OW + reorder[filter_idx] * OH * OW + row * OW + col;

                int tx = threadIdx.x;
                int ty = threadIdx.y;
                int tz = threadIdx.z;

                __shared__ float output[BLOCK_WIDTH][BLOCK_WIDTH][BLOCK_WIDTH];

                output[tz][ty][tx] = 0;

                if(filter_idx < OC){

                    int topleft_col = col * step - ((mask_width - 1) / 2);
                    int topleft_row = row * step - ((mask_width - 1) / 2);

                    int oc_stride_start = filter_idx * (ptset_size + 1);

                    for(int pi = 0; pi < ptset_size; pi++){

                        int stride_index = oc_stride_start + pi;
                        int nkernels = stride[stride_index + 1] - stride[stride_index];
                        int kernel_index_start = offset[filter_idx] + stride[stride_index];

                        for(int k = 0; k < nkernels; k++){

                            int topleft_data_index = bn * IC * IH * IW + index[kernel_index_start + k] * IH * IW + topleft_row * IW + topleft_col;


                            for(int v = 0; v < non_zero; v++){

                                int cur_row = topleft_row + ptset[pi * non_zero * 2 + v * 2 + 0];
                                int cur_col = topleft_col + ptset[pi * non_zero * 2 + v * 2 + 1];
                                int cur_index = topleft_data_index + ptset[pi * non_zero * 2 + v * 2 + 0] * IW + ptset[pi * non_zero * 2 + v * 2 + 1];

                                //out[output_index] = ptset[pi * non_zero * 2 + v * 2 + 0];

                                if(cur_row >= 0 && cur_row < IH && cur_col >=0 && cur_col < IW){
                                                    output[tz][ty][tx] += data[cur_index] * weight[non_zero * (kernel_index_start + k) + v];

                                }
                            }
                        }

                    }
                    out[output_index] = output[tz][ty][tx];

                }
			}


			__global__ void sparse_conv_shared_constant(float *data, int *offset, int *reorder, int *index, int *stride, float *ptset, float *out,
            int bn, int IC, int IH, int IW, int OC, int OH, int OW, int ptset_size, int non_zero, int mask_width, int step) {

                int filter_idx = blockIdx.x * blockDim.x + threadIdx.x;
                int row = blockIdx.y * blockDim.y + threadIdx.y;
                int col =  blockIdx.z * blockDim.z + threadIdx.z;
                int output_index = bn * OC * OH * OW + reorder[filter_idx] * OH * OW + row * OW + col;

                int tx = threadIdx.x;
                int ty = threadIdx.y;
                int tz = threadIdx.z;

                __shared__ float output[BLOCK_WIDTH][BLOCK_WIDTH][BLOCK_WIDTH];

                output[tz][ty][tx] = 0;

                if(filter_idx < OC){

                    int topleft_col = col * step - ((mask_width - 1) / 2);
                    int topleft_row = row * step - ((mask_width - 1) / 2);

                    int oc_stride_start = filter_idx * (ptset_size + 1);

                    for(int pi = 0; pi < ptset_size; pi++){

                        int stride_index = oc_stride_start + pi;
                        int nkernels = stride[stride_index + 1] - stride[stride_index];
                        int kernel_index_start = offset[filter_idx] + stride[stride_index];

                        for(int k = 0; k < nkernels; k++){

                            int topleft_data_index = bn * IC * IH * IW + index[kernel_index_start + k] * IH * IW + topleft_row * IW + topleft_col;


                            for(int v = 0; v < non_zero; v++){

                                int cur_row = topleft_row + ptset[pi * non_zero * 2 + v * 2 + 0];
                                int cur_col = topleft_col + ptset[pi * non_zero * 2 + v * 2 + 1];
                                int cur_index = topleft_data_index + ptset[pi * non_zero * 2 + v * 2 + 0] * IW + ptset[pi * non_zero * 2 + v * 2 + 1];

                                //out[output_index] = ptset[pi * non_zero * 2 + v * 2 + 0];

                                if(cur_row >= 0 && cur_row < IH && cur_col >=0 && cur_col < IW){
                                                    output[tz][ty][tx] += data[cur_index] * Weight[non_zero * (kernel_index_start + k) + v];

                                }
                            }
                        }

                    }
                    out[output_index] = output[tz][ty][tx];
                }
			}
            """
        # you can either use a string or save the kernel in kernel.cu file and reference it here.
        # Compile the kernel code when an instance
        # of this class is made.
        return SourceModule(kernelwrapper)

    def conv_sparse_naive(self, data, offset, reorder, index, stride, weight, ptset, step, out):
        BN, IC, IH, IW = data.shape
        _, OC, OH, OW = out.shape
        mask_width = 3

        pattern_set_size = ptset.shape[0]
        nonzero_per_kernel = ptset.shape[1]

        block_width = 4
        block_dim = (block_width, block_width, block_width)
        grid_dim = ((OC - 1) // block_width + 1, (OH - 1) // block_width + 1, (OW - 1) // block_width + 1)

        time_computation_start = cuda.Event()
        time_computation_end = cuda.Event()
        time_mem_transfer_start = cuda.Event()
        time_mem_transfer_end = cuda.Event()

        time_mem_transfer_start.record()

        data_d = cuda.mem_alloc(data.nbytes)
        offset_d = cuda.mem_alloc(offset.nbytes)
        reorder_d = cuda.mem_alloc(reorder.nbytes)
        index_d = cuda.mem_alloc(index.nbytes)
        stride_d = cuda.mem_alloc(stride.nbytes)
        weight_d = cuda.mem_alloc(weight.nbytes)
        ptset_d = cuda.mem_alloc(ptset.nbytes)
        out_d = cuda.mem_alloc(out.nbytes)

        do_conv_sparse = self.mod.get_function("sparse_conv_naive")

        cuda.memcpy_htod(data_d, data)
        cuda.memcpy_htod(offset_d, offset)
        cuda.memcpy_htod(reorder_d, reorder)
        cuda.memcpy_htod(index_d, index)
        cuda.memcpy_htod(stride_d, stride)
        cuda.memcpy_htod(weight_d, weight)
        cuda.memcpy_htod(ptset_d, ptset)

        time_computation_start.record()

        for bn in range(BN):
            do_conv_sparse(data_d, offset_d, reorder_d, index_d, stride_d, weight_d, ptset_d, out_d,
                           np.int32(bn), np.int32(IC), np.int32(IH), np.int32(IW), np.int32(OC), np.int32(OH),
                           np.int32(OW),
                           np.int32(pattern_set_size), np.int32(nonzero_per_kernel), np.int32(mask_width),
                           np.int32(step), block=block_dim,
                           grid=grid_dim)

        time_computation_end.record()
        time_computation_end.synchronize()
        # time_computation_end = time.time()

        cuda.memcpy_dtoh(out, out_d)

        time_mem_transfer_end.record()
        time_mem_transfer_end.synchronize()
        # time_mem_transfer_end = time.time()

        time_without_mem = time_computation_start.time_till(time_computation_end)*1e-3
        time_include_mem = time_mem_transfer_start.time_till(time_mem_transfer_end)*1e-3

        # time_without_mem = time_computation_end - time_computation_start
        # time_include_mem = time_mem_transfer_end - time_mem_transfer_start

        return out, time_without_mem, time_include_mem

    def conv_sparse_shared_mem(self, data, offset, reorder, index, stride, weight, ptset, step, out):
        BN, IC, IH, IW = data.shape
        _, OC, OH, OW = out.shape
        mask_width = 3

        pattern_set_size = ptset.shape[0]
        nonzero_per_kernel = ptset.shape[1]

        block_width = 4
        block_dim = (block_width, block_width, block_width)
        grid_dim = ((OC - 1) // block_width + 1, (OH - 1) // block_width + 1, (OW - 1) // block_width + 1)

        time_computation_start = cuda.Event()
        time_computation_end = cuda.Event()
        time_mem_transfer_start = cuda.Event()
        time_mem_transfer_end = cuda.Event()

        time_mem_transfer_start.record()

        data_d = cuda.mem_alloc(data.nbytes)
        offset_d = cuda.mem_alloc(offset.nbytes)
        reorder_d = cuda.mem_alloc(reorder.nbytes)
        index_d = cuda.mem_alloc(index.nbytes)
        stride_d = cuda.mem_alloc(stride.nbytes)
        weight_d = cuda.mem_alloc(weight.nbytes)
        ptset_d = cuda.mem_alloc(ptset.nbytes)
        out_d = cuda.mem_alloc(out.nbytes)

        do_conv_sparse_shared = self.mod.get_function("sparse_conv_shared")

        cuda.memcpy_htod(data_d, data)
        cuda.memcpy_htod(offset_d, offset)
        cuda.memcpy_htod(reorder_d, reorder)
        cuda.memcpy_htod(index_d, index)
        cuda.memcpy_htod(stride_d, stride)
        cuda.memcpy_htod(weight_d, weight)
        cuda.memcpy_htod(ptset_d, ptset)

        time_computation_start.record()

        for bn in range(BN):
            do_conv_sparse_shared(data_d, offset_d, reorder_d, index_d, stride_d, weight_d, ptset_d, out_d,
                                  np.int32(bn), np.int32(IC), np.int32(IH), np.int32(IW), np.int32(OC), np.int32(OH),
                                  np.int32(OW),
                                  np.int32(pattern_set_size), np.int32(nonzero_per_kernel), np.int32(mask_width),
                                  np.int32(step), block=block_dim,
                                  grid=grid_dim)

        time_computation_end.record()
        time_computation_end.synchronize()
        # time_computation_end = time.time()

        cuda.memcpy_dtoh(out, out_d)

        time_mem_transfer_end.record()
        time_mem_transfer_end.synchronize()
        # time_mem_transfer_end = time.time()


        time_without_mem = time_computation_start.time_till(time_computation_end)*1e-3
        time_include_mem = time_mem_transfer_start.time_till(time_mem_transfer_end)*1e-3
        # time_without_mem = time_computation_end - time_computation_start
        # time_include_mem = time_mem_transfer_end - time_mem_transfer_start

        return out, time_without_mem, time_include_mem

    def conv_sparse_shared_constant_mem(self, data, offset, reorder, index, stride, weight, ptset, step, out):
        BN, IC, IH, IW = data.shape
        _, OC, OH, OW = out.shape
        mask_width = 3

        pattern_set_size = ptset.shape[0]
        nonzero_per_kernel = ptset.shape[1]

        block_width = 4
        block_dim = (block_width, block_width, block_width)
        grid_dim = ((OC - 1) // block_width + 1, (OH - 1) // block_width + 1, (OW - 1) // block_width + 1)

        time_computation_start = cuda.Event()
        time_computation_end = cuda.Event()
        time_mem_transfer_start = cuda.Event()
        time_mem_transfer_end = cuda.Event()

        time_mem_transfer_start.record()

        data_d = cuda.mem_alloc(data.nbytes)
        offset_d = cuda.mem_alloc(offset.nbytes)
        reorder_d = cuda.mem_alloc(reorder.nbytes)
        index_d = cuda.mem_alloc(index.nbytes)
        stride_d = cuda.mem_alloc(stride.nbytes)

        weight_d, _ = self.mod.get_global('Weight')

        ptset_d = cuda.mem_alloc(ptset.nbytes)
        out_d = cuda.mem_alloc(out.nbytes)

        do_conv_sparse_shared_constant = self.mod.get_function("sparse_conv_shared_constant")

        cuda.memcpy_htod(data_d, data)
        cuda.memcpy_htod(offset_d, offset)
        cuda.memcpy_htod(reorder_d, reorder)
        cuda.memcpy_htod(index_d, index)
        cuda.memcpy_htod(stride_d, stride)
        cuda.memcpy_htod(weight_d, weight)
        cuda.memcpy_htod(ptset_d, ptset)

        time_computation_start.record()

        for bn in range(BN):
            do_conv_sparse_shared_constant(data_d, offset_d, reorder_d, index_d, stride_d, ptset_d, out_d,
                                           np.int32(bn), np.int32(IC), np.int32(IH), np.int32(IW), np.int32(OC),
                                           np.int32(OH), np.int32(OW),
                                           np.int32(pattern_set_size), np.int32(nonzero_per_kernel),
                                           np.int32(mask_width), np.int32(step), block=block_dim,
                                           grid=grid_dim)

        time_computation_end.record()
        time_computation_end.synchronize()

        cuda.memcpy_dtoh(out, out_d)

        time_mem_transfer_end.record()
        time_mem_transfer_end.synchronize()

        time_without_mem = time_computation_start.time_till(time_computation_end)*1e-3
        time_include_mem = time_mem_transfer_start.time_till(time_mem_transfer_end)*1e-3

        return out, time_without_mem, time_include_mem

def conv_nnpack():
    conv = Convolution()
    cuda0 = torch.device('cuda:0')
    cpu = torch.device('cpu')
    total_time = 0

    for idx in tqdm(range(len(residual_convs[:]))):
        input_data = np.ones(data_shapes[idx]).astype(np.float32)
        conv_mask = residual_convs[idx].astype(np.float32)
        input_data_g = torch.tensor(input_data, device=cuda0)
        conv_mask_g = torch.tensor(conv_mask, device=cuda0)
        start = time.time()
        output_gt = nn.functional.conv2d(input_data_g, conv_mask_g, padding=1)
        # output_gt = nn.functional.conv2d(torch.tensor(input_data), torch.tensor(conv_mask),padding=1)
        end = time.time()
        total_time += end - start
    print("nnpack")
    print(f'{round(total_time, 3)}s')

def conv_naive():
    conv = Convolution()
    cuda0 = torch.device('cuda:0')
    total_time = 0

    for idx in tqdm(range(len(residual_convs[:]))):
        input_data = np.ones(data_shapes[idx]).astype(np.float32)
        conv_mask = residual_convs[idx].astype(np.float32)
        output_1, time_ = conv.conv_multiple_filters(input_data, conv_mask)
        total_time += time_

    #print(f'{round(total_time,3)}s')
    print("naive")
    print(f'{round(total_time,3)}s')



# if __name__ == "__main__":
#     conv = SparseConvolution()
#
#     path = 'resnet34_6_pattern_connectivity_pruning.pt'
#     state_dict = torch.load(path, map_location=torch.device('cpu'))
#
#     residual_convs = [v.cpu().numpy() for (k, v) in state_dict.items() if "layer" in k and "conv" in k]
#     data_shapes = [
#         [1, 64, 32, 32], [1, 64, 32, 32], [1, 64, 32, 32], [1, 64, 32, 32], [1, 64, 32, 32], [1, 64, 32, 32],
#         [1, 64, 32, 32], [1, 128, 16, 16], [1, 128, 16, 16], [1, 128, 16, 16], [1, 128, 16, 16], [1, 128, 16, 16],
#         [1, 128, 16, 16], [1, 128, 16, 16], [1, 128, 16, 16], [1, 256, 8, 8], [1, 256, 8, 8], [1, 256, 8, 8],
#         [1, 256, 8, 8], [1, 256, 8, 8], [1, 256, 8, 8], [1, 256, 8, 8], [1, 256, 8, 8], [1, 256, 8, 8],
#         [1, 256, 8, 8], [1, 256, 8, 8], [1, 256, 8, 8], [1, 512, 4, 4], [1, 512, 4, 4], [1, 512, 4, 4],
#         [1, 512, 4, 4], [1, 512, 4, 4],
#     ]
#
#     time_without_mem_list_naive = []
#     time_include_mem_list_naive = []
#     time_without_mem_list_shared = []
#     time_include_mem_list_shared = []
#     time_without_mem_list_constant = []
#     time_include_mem_list_constant = []
#     time_list_pytorch = []
#
#     for i in range(len(residual_convs)):
#         print("layer" + str(i))
#         input_data = np.float32(np.ones(data_shapes[i]))
#
#         output_data =  np.float32(np.zeros(data_shapes[i + 1]))
#
#         conv_layer_weight = residual_convs[i].astype(np.float32)
#         patterns = np.array(extract_patterns(conv_layer_weight))
#         sparse_conv_arrays = SparseConvArrays(conv_layer_weight, patterns)
#         offset = sparse_conv_arrays.offset
#         reorder = sparse_conv_arrays.reorder
#         index = sparse_conv_arrays.index
#         stride = sparse_conv_arrays.stride
#         sparse_weight = sparse_conv_arrays.weight
#         ptset = np.float32(sparse_conv_arrays.ptset)
#
#         # step 卷积步长
#         step = int(data_shapes[i][2] / data_shapes[i + 1][2])
#         output_naive, time_without_mem_naive, time_include_mem_naive = conv.conv_sparse_naive(input_data, offset, reorder, index, stride, sparse_weight, ptset, step, output_data)
#         output_shared, time_without_mem_shared, time_include_mem_shared = conv.conv_sparse_shared_mem(input_data, offset, reorder, index, stride, sparse_weight, ptset, step, output_data)
#
#         time_without_mem_list_naive.append(time_without_mem_naive)
#         time_include_mem_list_naive.append(time_include_mem_naive)
#         time_without_mem_list_shared.append(time_without_mem_shared)
#         time_include_mem_list_shared.append(time_include_mem_shared)
#
#         #constant memory limit
#         if sparse_weight.shape[0] <= 16384:
#             output_constant, time_without_mem_constant,  time_include_mem_constant = conv.conv_sparse_shared_constant_mem(input_data, offset, reorder, index, stride, sparse_weight, ptset, step, output_data)
#             time_without_mem_list_constant.append(time_without_mem_constant)
#             time_include_mem_list_constant.append(time_include_mem_constant)
#
#         pytorch_start = time.time()
#         output_gt = nn.functional.conv2d(torch.tensor(input_data), torch.tensor(conv_layer_weight), padding=1, stride=step)
#         pytorch_end = time.time()
#
#         pytorch_time = pytorch_end - pytorch_start
#         time_list_pytorch.append(pytorch_time)
#
#         output_gt = output_gt.cpu().numpy()
#         # print(np.allclose(output_naive, output_gt))
#         # print(np.allclose(output_shared, output_gt))
#
#         #constant memory limit
#         # if sparse_weight.shape[0] <= 16384:
#         #     print(np.allclose(output_constant, output_gt))
#
#         #break when it comes to last layer
#         if i == len(residual_convs) - 2:
#             break
#
# print("Pytorch Conv2d time")
# print(np.sum(time_list_pytorch))
# print("Sparse Naive without memory transfer")
# print(np.sum(time_without_mem_list_naive))
# print("parse Naive include memory transfer")
# print(np.sum(time_include_mem_list_naive))
# print("Sparse Shared memory without memory transfer")
# print(np.sum(time_without_mem_list_shared))
# print("Sparse Shared memory include memory transfer")
# print(np.sum(time_include_mem_list_shared))
# print("Sparse Shared and Constant memory without memory transfer")
# print(time_without_mem_list_constant)
# print("Sparse Shared and Constant memory include memory transfer")
# print(time_include_mem_list_constant)
#
# conv_naive()
# conv_nnpack()


