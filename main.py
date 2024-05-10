import numpy as np
# import matplotlib.pyplot as plt
import time
import math
import pygame

# Sim settings
scrw = 1280  # screen width
scrh = 720  # screen height
bw = int(1e+2)  # board dims
cellsize = 50  # default cell size in pixels
cps = 5  # mvmt speed in cells per second
cellclr = {0: "Black",
           1: "Blue",
           2: "Green",
           3: "White"
           }  # cell colors
scrsplit = 0.7  # screen split - board | stats, eff 896px
maxfps = int(60)  # max FPS limit
margin = 2  # extra cells to render
zoomsens = 1.008  # zoom sensitivity, multiplicative

# Initialize pygame & game setup:
pygame.init()
screen = pygame.display.set_mode((scrw, scrh))
pygame.display.set_caption("Cellular Automation Simulator")
font = pygame.font.Font('freesansbold.ttf', 32)
clock = pygame.time.Clock()
currfps = font.render('Current FPS: ',
                      True,
                      "White",
                      "Black"
                      )
currfpsRect = currfps.get_rect()
currfpsRect.left = (scrw * scrsplit * 1.03)
currfpsRect.top = (scrh * 0.05)
fps = 0
fpstext = font.render(str(fps), True, "White")
fpsrect = fpstext.get_rect()
fpsrect.left = currfpsRect.right * 1.03
fpsrect.top = currfpsRect.top
running = True
dt = 0  # time elapsed (in ms)
i = 0

bh = int(bw)
board = np.zeros((bw, bh), dtype=np.int8)
cells_h = (scrh / cellsize) + margin  # number of cells to render, w and h dims. float
cells_w = ((scrw * scrsplit) / cellsize) + margin
btl = [(bw / 2) - ((cells_w - margin) / 2) + 0.5,
       (bh / 2) - ((cells_h - margin) / 2) + 0.5
       ]  # board top left, in board coords
scrtl = [int(btl[0] * cellsize),
         int(btl[1] * cellsize)
         ]  # screen top left, in pixels.

# Initial state
for i in range(bw):
    for j in range(bh):
        board[i][j] = 1

for i in range(bw):
    board[i][0] = 4
    board[i][bh - 1] = 4
for i in range(bh):
    board[0][i] = 4
    board[bw - 1][i] = 4
board[50][50] = 2

# Main game loop:
while running:
    i += 1
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            break

    screen.fill("gray25")

    cell_left = math.floor(btl[0])
    cell_right = cell_left + int(cells_w)
    cell_top = math.floor(btl[1])
    cell_bottom = cell_top + int(cells_h)

    fps = int(clock.get_fps())
    fpstext = font.render(str(fps), True, "White")

    for wc in range(cell_left, cell_right, 1):
        for hc in range(cell_top, cell_bottom, 1):
            if board[wc][hc] != 0:
                pygame.draw.rect(screen,
                                 cellclr[int(board[wc][hc])],
                                 pygame.Rect(int((wc * cellsize) - scrtl[0]),
                                             int((hc * cellsize) - scrtl[1]),
                                             int(cellsize),
                                             int(cellsize)
                                             )
                                 )
                pygame.draw.rect(screen,
                                 "Black",
                                 pygame.Rect(int((wc * cellsize) - scrtl[0]),
                                             int((hc * cellsize) - scrtl[1]),
                                             int(cellsize),
                                             int(cellsize)
                                             ),
                                 math.ceil(cellsize / 40)
                                 )

    pygame.draw.line(screen,
                     "White",
                     (int(scrw * scrsplit), 0),
                     (int(scrw * scrsplit), scrh),
                     3
                     )
    pygame.draw.rect(screen,
                     "Black",
                     pygame.Rect(scrw * scrsplit,
                                 0,
                                 scrw * (1 - scrsplit),
                                 scrh
                                 )
                     )
    screen.blit(currfps, currfpsRect)
    screen.blit(fpstext, fpsrect)

    # Input handling:
    keys = pygame.key.get_pressed()
    if keys[pygame.K_w]:
        if btl[1] > cells_h / 2:
            btl[1] -= cps * (dt / 1000)
    if keys[pygame.K_s]:
        if btl[1] < bh - (cells_h / 2):
            btl[1] += cps * (dt / 1000)
    if keys[pygame.K_a]:
        if btl[0] > cells_w / 2:
            btl[0] -= cps * (dt / 1000)
    if keys[pygame.K_d]:
        if btl[0] < bw - (cells_w / 2):
            btl[0] += cps * (dt / 1000)
    if keys[pygame.K_k]:
        if cellsize > 20:
            cellsize /= zoomsens
    if keys[pygame.K_j]:
        if cellsize < 200:
            cellsize *= zoomsens

    scrtl = [int(btl[0] * cellsize), int(btl[1] * cellsize)]
    cells_h = (scrh / cellsize) + margin
    cells_w = ((scrw * scrsplit) / cellsize) + margin

    # Render
    pygame.display.flip()
    dt = clock.tick(maxfps)

print("Process finished")
