import numpy as np
import cupy as cp
import pygame
import sys

# from cupyx.profiler import benchmark
# print(benchmark(squared_diff, (x, y), n_repeat=1000))

pygame.init()
screen = pygame.display.set_mode((1280, 720))
loaded_from_source = r'''
extern "C"
{
__global__ void test_sum(int* y)
    {
    unsigned int tid = (blockDim.x * blockIdx.x) + threadIdx.x;
    unsigned int py = tid % blockDim.x;
    unsigned int px = tid / blockDim.x;
        if (tid < 1280 * 720)
        {
            if(px < 240)
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
scrn = cp.zeros((1280, 720, 3), dtype=cp.uint32)
disp((1280,), (720,), scrn)
while True:
    screen.fill("Black")
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            print("User quit")
            sys.exit()

    pygame.surfarray.blit_array(screen, scrn.get())
    pygame.display.flip()
