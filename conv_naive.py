import numpy as np
import time
from scipy import signal
import pycuda.autoinit
import pycuda.driver as cuda
from pycuda.compiler import SourceModule


class Convolution:
    def __init__(self):
        """
        Attributes for instance of EncoderDecoder module
        """
        self.mod = self.getSourceModule()
        pass

    def getSourceModule(self):
        # kernel code wrapper
        kernelwrapper = """
            #define TILE_WIDTH 8
            #define MASK_WIDTH 3
            #define CHANNEL 3
            //__constant__ float M[MASK_WIDTH][MASK_WIDTH];

            __global__ void Conv_naive(float *data, float *mask, float *res, const int IC, 
                                       const int IH, const int IW, const int OC, 
                                       const int MH, const int MW) {
                
                int tx = threadIdx.x;
                int oc = blockIdx.x * TILE_WIDTH + tx;
                
                int ty = threadIdx.y;
                int tz = threadIdx.z;
                int row_o = blockIdx.z * TILE_WIDTH + tz;
                int col_o = blockIdx.y * TILE_WIDTH + ty;
                int row_i = row_o - (MH - 1) / 2;
                int col_i = col_o - (MH - 1) / 2;

                if (oc < OC){
                    float output = 0.0f;
                
                    for(int i = 0; i < IC; i++){
                        for(int j = 0; j < MW; j++){
                            for(int k = 0; k < MW; k++){
                                if ((row_i + j >= 0) && (row_i + j < IH) && (col_i + k >= 0) && (col_i + k < IW)){
                                    output = output + data[i * IW * IH + (row_i + j) * IW + col_i + k] * mask[oc * IC * MH * MW + i * MH * MW + j * MW + k];
                                    }
                                else{
                                    output = output + 0;
                                }               
                            }
                        }
                    }

                    __syncthreads();

                    if(row_o < IH && col_o < IW){
                        res[oc * IH * IW + row_o * IW + col_o] = output;
                    }

                    
                }
            } """
        # you can either use a string or save the kernel in kernel.cu file and reference it here.
        # Compile the kernel code when an instance
        # of this class is made.
        return SourceModule(kernelwrapper)

    def conv_multiple_filters(self, input_data, conv_mask):

        # A: kernel
        # B: input feature map
        if len(input_data.shape) == 4:
            input_data = input_data[0]

        IC,IH,IW = np.int32(input_data.shape)
        OC,_,MH,MW = np.int32(conv_mask.shape)


        output = np.zeros((OC,IH,IW)).astype(np.float32)
        tile_width = 8
        block_width = int(tile_width + (MW - 1))
        block_dim = (block_width, block_width, block_width)
        grid_dim = (int((OC - 1) // tile_width + 1), int((IH - 1) // tile_width + 1), int((IW - 1) // tile_width + 1))

        mask_d = cuda.mem_alloc(conv_mask.nbytes)
        data_d = cuda.mem_alloc(input_data.nbytes)
        output_d = cuda.mem_alloc(output.nbytes)
        do_conv_naive = self.mod.get_function("Conv_naive")
        cuda.memcpy_htod(mask_d, conv_mask)
        cuda.memcpy_htod(data_d, input_data)
        
        time_start = cuda.Event()
        time_end = cuda.Event()
        #start = time.time()
        time_start.record()
        do_conv_naive(data_d, mask_d, output_d, IC, IH, IW, OC, MH, MW, block=block_dim,grid=grid_dim)
        #end = time.time()
        time_end.record()
        time_end.synchronize()

        #time_ = end - start
        time_= time_start.time_till(time_end)*1e-3

        cuda.memcpy_dtoh(output, output_d)
       
        return output, time_





if __name__ == "__main__":
    conv = Convolution()

    #kernel size(OC, IC, KH, KW) = (4, 2, 3, 3)
    conv_mask = np.float32(np.ones((3,3,3,3)))

    # input_feature_map size (1, 2, 16, 16)
    input_feature_map = np.float32(np.ones((3, 4, 4)))
    # depth = input_feature_map.shape[0]
    # width = input_feature_map.shape[1]
    # height = input_feature_map.shape[2]
    # print(width)
    # print(height)
    # print(input_feature_map)
    # print(mask)
    # print(conv.conv_gpu_shared_and_constant_mem(mask, input_feature_map, depth, width, height, mask_width))
    output = conv.conv_multiple_filters(input_feature_map, conv_mask)
    print(output)





