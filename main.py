import numpy as np
# import matplotlib.pyplot as plt
import time
import math
import pygame
import sys

# Sim settings
scrw = 1280  # screen width
scrh = 720  # screen height
bw = int(1e+2)  # board dims
cellsize = 50  # default cell size in pixels
cps = 9  # base mvmt speed in cells per second
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
fpstext = font.render(str(fps),
                      True,
                      "White",
                      "Black")
fpsRect = fpstext.get_rect()
fpsRect.left = currfpsRect.right * 1.01
fpsRect.top = currfpsRect.top
cst = font.render('Scale: ',
                  True,
                  "White",
                  "Black")
cstRect = cst.get_rect()
cstRect.left = (scrw * scrsplit * 1.03)
cstRect.top = currfpsRect.bottom * 1.1
csnt = font.render(str(cellsize),
                   True,
                   "White",
                   "Black")
csntRect = csnt.get_rect()
csntRect.left = cstRect.right * 1.01
csntRect.top = cstRect.top

running = True
dt = 0  # time elapsed (in ms)
bh = int(bw)
board = np.zeros((bw, bh), dtype=np.int8)
cells_h = (scrh / cellsize) + margin  # number of cells to render, w and h dims. float
cells_w = ((scrw * scrsplit) / cellsize) + margin
cam_pos = [(bw / 2) - ((cells_w - margin) / 2) + 0.5,
           (bh / 2) - ((cells_h - margin) / 2) + 0.5
           ]  # cam position, denotes top left in board coordinates

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

# Main game loop:
while running:
    i += 1
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            print("User quit")
            sys.exit()

    screen.fill("gray25")

    cell_left = max(math.floor(cam_pos[0]), 0)
    cell_right = cell_left + int(cells_w)
    cell_top = max(math.floor(cam_pos[1]), 0)
    cell_bottom = cell_top + int(cells_h)

    fps = clock.get_fps()
    fpstext = font.render(str(f'{fps:.2f}'), True, "White", "Black")
    csnt = font.render(str(f'{cellsize:.2f}'), True, "White", "Black")

    for wc in range(cell_left, min(cell_right, bw), 1):
        for hc in range(cell_top, min(cell_bottom, bh), 1):
            if board[wc][hc] != 0:
                pygame.draw.rect(screen,
                                 cellclr[int(board[wc][hc])],
                                 pygame.Rect(int((wc - cam_pos[0]) * cellsize),
                                             int((hc - cam_pos[1]) * cellsize),
                                             int(cellsize),
                                             int(cellsize)
                                             )
                                 )
                pygame.draw.rect(screen,
                                 "Black",
                                 pygame.Rect(int((wc - cam_pos[0]) * cellsize),
                                             int((hc - cam_pos[1]) * cellsize),
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
            cps *= zoomsens
    if keys[pygame.K_j]:
        cellsize *= zoomsens
        cps /= zoomsens

    cells_h = (scrh / cellsize) + margin
    cells_w = ((scrw * scrsplit) / cellsize) + margin

    # Render
    pygame.display.flip()
    dt = clock.tick(maxfps)

print("Process finished")
