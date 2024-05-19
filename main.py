import numpy as np
import cupy as cp
import time
import math
import pygame
import sys
import cv2


# Sim settings
scrw = 1280  # screen width
scrh = 720  # screen height
bw = int(1e+2)  # board dims
cellsize = 40  # default cell size in pixels
base_cps = 100  # base mvmt speed
cellclr = {0: (0, 0, 0),
           1: (0, 0, 255),
           2: (0, 255, 0),
           3: (255, 255, 255)
           }  # cell colors
scrsplit = 0.7  # screen split - board | stats, eff 896px
maxfps = int(60)  # max FPS limit
margin = 2  # extra cells to render
zoomsens = 1.008  # zoom sensitivity, multiplicative
textcolor = "White"
textbg = "Black"

# Initialize pygame & game setup:
pygame.init()
screen = pygame.display.set_mode((scrw, scrh))
pygame.display.set_caption("Cellular Automation Simulator")
font = pygame.font.Font('freesansbold.ttf', 32)
clock = pygame.time.Clock()

currfps = font.render('Current FPS: ',
                      True,
                      textcolor,
                      textbg
                      )
currfpsRect = currfps.get_rect()
currfpsRect.left = (scrw * scrsplit * 1.03)
currfpsRect.top = (scrh * 0.05)
fps = 0
fpstext = font.render(str(fps),
                      True,
                      textcolor,
                      textbg)
fpsRect = fpstext.get_rect()
fpsRect.left = currfpsRect.right * 1.01
fpsRect.top = currfpsRect.top
cst = font.render('Scale: ',
                  True,
                  textcolor,
                  textbg)
cstRect = cst.get_rect()
cstRect.left = (scrw * scrsplit * 1.03)
cstRect.top = currfpsRect.bottom * 1.1
csnt = font.render(str(cellsize),
                   True,
                   textcolor,
                   textbg)
csntRect = csnt.get_rect()
csntRect.left = cstRect.right * 1.01
csntRect.top = cstRect.top

running = True
dt = 0  # time elapsed (in ms)
bh = int(bw)
board = np.zeros((bw, bh), dtype=np.uint8)
cam_pos = [bw / 2, bh / 2]
cps = base_cps * math.pow(cellsize, -0.6)

# Initial state
for i in range(bw):
    for j in range(bh):
        board[i][j] = 1

for i in range(bw):
    board[i][0] = 3
    board[i][bh - 1] = 3
for i in range(bh):
    board[0][i] = 3
    board[bw - 1][i] = 3
board[50][50] = 2
i = 0

cppsource = r'''
extern "C"
{
__global__ void render(const float* inps, const unsigned int* board, int* scrn)
    {
        unsigned int indx = blockDim.x * blockIdx.x + threadIdx.x;
        if (indx < int(*(inps + 6)))
        {
            float bx = 0;
            float by = 0;
            int px = 0;
            int py = 0;
            //int escrw = 0;
            float cs = 1;
            float cpx = 0;
            float cpy = 0;
            int bw = 100;
            int bh = 100;
            
            //escrw = *(inps + 0);
            cpx = *(inps + 1);
            cpy = *(inps + 2);
            cs = *(inps + 3);
            bw = *(inps + 4);
            bh = *(inps + 5);
            
            px = indx / blockDim.x;
            py = indx % blockDim.x;
            //bx = cpx + (px / cs);
            //by = cpy + (py / cs);
            bx = (px / cs) + cpx;
            by = (py / cs) + cpy;
            //printf("%lf\n", bx);
            if(bx >= 0 && by >= 0 && int(bx) <= bw && int(by) <= bh)
            {
                if(*(board + int(bx) + (bw * int(by))) == 1)
                {
                    *(scrn + (3 * indx)) = 30;
                    *(scrn + (3 * indx) + 1) = 30;
                    *(scrn + (3 * indx) + 2) = 30;
                } else if (*(board + int(bx) + (bw * int(by))) == 2)
                {
                    *(scrn + (3 * indx)) = 255;
                    *(scrn + (3 * indx) + 1) = 255;
                    *(scrn + (3 * indx) + 2) = 255;  
                } else
                {
                    *(scrn + (3 * indx)) = 0;
                    *(scrn + (3 * indx) + 1) = 0;
                    *(scrn + (3 * indx) + 2) = 0;
                }
            }
            else
            {
                *(scrn + (3 * indx)) = 0;
                *(scrn + (3 * indx) + 1) = 0;
                *(scrn + (3 * indx) + 2) = 0;
            }
        }
    }
}
'''
gpgpumodule = cp.RawModule(code=cppsource)
rendergpu = gpgpumodule.get_function('render')


def renderer():
    bd = cp.asarray(board, dtype=cp.uint32)
    screenarr = cp.zeros((int(scrw * scrsplit), scrh, 3), dtype=cp.uint32)

    args = cp.array([scrw * scrsplit,
                     cam_pos[0],
                     cam_pos[1],
                     cellsize,
                     bw,
                     bh,
                     scrw * scrsplit * scrh,
                     ], dtype=cp.float32)
    rendergpu((int(scrw * scrsplit),), (scrh,),
              (args,
               bd,
               screenarr))
    return screenarr.get()


# Main game loop:
while running:
    i += 1
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            print("User quit")
            sys.exit()

    scrn = np.zeros((scrw, scrh, 3), dtype=np.uint8)
    fps = clock.get_fps()
    fpstext = font.render(str(f'{fps:.2f}'), True, textcolor, textbg)
    csnt = font.render(str(f'{cellsize:.2f}'), True, textcolor, textbg)

    scrn[0:int(scrw * scrsplit)][:] = renderer()

    pygame.surfarray.blit_array(screen, scrn)
    pygame.draw.line(screen,
                     textcolor,
                     (int(scrw * scrsplit), 0),
                     (int(scrw * scrsplit), scrh),
                     3
                     )
    pygame.draw.rect(screen,
                     textbg,
                     pygame.Rect(scrw * scrsplit,
                                 0,
                                 scrw * (1 - scrsplit),
                                 scrh
                                 )
                     )
    screen.blit(currfps, currfpsRect)
    screen.blit(fpstext, fpsRect)
    screen.blit(csnt, csntRect)
    screen.blit(cst, cstRect)

    # Input handling:
    keys = pygame.key.get_pressed()
    if keys[pygame.K_w]:
        cam_pos[1] -= cps * (dt / 1000)
    if keys[pygame.K_s]:
        cam_pos[1] += cps * (dt / 1000)
    if keys[pygame.K_a]:
        cam_pos[0] -= cps * (dt / 1000)
    if keys[pygame.K_d]:
        cam_pos[0] += cps * (dt / 1000)
    if keys[pygame.K_k]:
        if cellsize > 1:
            cellsize /= zoomsens
    if keys[pygame.K_j]:
        cellsize *= zoomsens

    cps = base_cps * math.pow(cellsize, -0.6)
    # Render
    pygame.display.flip()
    dt = clock.tick(maxfps)

print("Process finished")
