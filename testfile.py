import numpy as np
import cupy as cp
import pygame
import sys

# from cupyx.profiler import benchmark
# print(benchmark(squared_diff, (x, y), n_repeat=1000))

pygame.init()
screen = pygame.display.set_mode((1120, 800))
loaded_from_source = r'''
extern "C"
{
__global__ void test_sum(int* y ) //, const unsigned int arrxsize)
    {
    unsigned int i = (blockIdx.x * blockDim.x) + threadIdx.x;
    unsigned int j = (blockIdx.y * blockDim.y) + threadIdx.y;
    unsigned int ylim = gridDim.y * blockDim.y;
    unsigned int tid = j + (i * ylim);
    unsigned int xlim = gridDim.x * blockDim.x;
    unsigned int py = j;
    unsigned int px = i;
        if (tid < xlim * ylim)
        {
            if(px > 300)
            {
                *(y + (tid * 3)) = 0;
                *(y + (tid * 3) + 1) = 255;
                *(y + (tid * 3) + 2) = 0;
            }
            else
            {
                *(y + (tid * 3)) = 255;
                *(y + (tid * 3) + 1) = 0;
                *(y + (tid * 3) + 2) = 0;
            }
        }
    }
}
'''
module = cp.RawModule(code=loaded_from_source)
disp = module.get_function('test_sum')
scrn = cp.zeros((1120, 800, 3), dtype=cp.uint32)
disp((70, 50), (16, 16), scrn)
while True:
    screen.fill("Black")
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            print("User quit")
            sys.exit()

    pygame.surfarray.blit_array(screen, scrn.get())
    pygame.display.flip()
