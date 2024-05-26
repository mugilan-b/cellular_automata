import numpy as np
import cupy as cp
import pygame
import sys


pygame.init()
resX = int(1280)
resY = int(720)
screen = pygame.display.set_mode((resX, resY))
loaded_from_source = r'''
extern "C"
{
__global__ void test_sum(int* y, int* z)
    {
    unsigned int i = (blockIdx.x * blockDim.x) + threadIdx.x;
    unsigned int j = (blockIdx.y * blockDim.y) + threadIdx.y;
    unsigned int ylim = gridDim.y * blockDim.y;
    unsigned int xlim = gridDim.x * blockDim.x;
    unsigned int tid = j + (i * ylim);
        if (tid < xlim * ylim)
        {
            unsigned int tmp = *(z + tid);
            if(true)
            {
                *(y + (tid * 3)) = tmp;
                *(y + (tid * 3) + 1) = 0;
                *(y + (tid * 3) + 2) = 0;
            }
            else
            {
                *(y + (tid * 3)) = 0;
                *(y + (tid * 3) + 1) = 0;
                *(y + (tid * 3) + 2) = 0;
            }
        }
    }
}
'''
module = cp.RawModule(code=loaded_from_source)
disp = module.get_function('test_sum')
arr = cp.zeros((resX, resY), dtype=np.uint32)
for i in range(resX):
    arr[i][400] = 255
    arr[i][401] = 255
    arr[i][402] = 255
scrn = cp.zeros((resX, resY, 3), dtype=cp.uint32)
disp((int(resX / 32), int(resY / 16)), (32, 16), (scrn, arr))
while True:
    screen.fill("Black")
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            print("User quit")
            print(cp.max(arr))
            sys.exit()

    pygame.surfarray.blit_array(screen, scrn.get())
    pygame.display.flip()
