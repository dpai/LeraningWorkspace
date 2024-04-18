import pygame as pg
import sys
import time
from pygame.locals import *
import threading
import pandas as pd
import numpy as np

## Required variables
solved = None
width = 500
height = 500
cellwidth = float(width/9)
cellheight = float(height/9)
white = (255, 255, 255)
line_color = (0, 0, 0)

## Initialize Pygame
pg.init()
fps = 30
CLOCK = pg.time.Clock()
screen = pg.display.set_mode((width, height+100), 0, 32)
pg.display.set_caption("My Sudoko Solver")

## Load images
one_img = pg.image.load("1.png")
two_img = pg.image.load("2.png")
three_img = pg.image.load("3.png")
four_img = pg.image.load("4.png")
five_img = pg.image.load("5.png")
six_img = pg.image.load("6.png")
seven_img = pg.image.load("7.png")
eight_img = pg.image.load("8.png")
nine_img = pg.image.load("9.png")

## Scale the imaages to grid size
one_img = pg.transform.scale(one_img, (cellheight/2,cellwidth/2))
two_img = pg.transform.scale(two_img, (cellheight/2,cellwidth/2))
three_img = pg.transform.scale(three_img, (cellheight/2,cellwidth/2))
four_img = pg.transform.scale(four_img, (cellheight/2,cellwidth/2))
five_img = pg.transform.scale(five_img, (cellheight/2,cellwidth/2))
six_img = pg.transform.scale(six_img, (cellheight/2,cellwidth/2))
seven_img = pg.transform.scale(seven_img, (cellheight/2,cellwidth/2))
eight_img = pg.transform.scale(eight_img, (cellheight/2,cellwidth/2))
nine_img = pg.transform.scale(nine_img, (cellheight/2,cellwidth/2))

## Map from number to image
mapnumtoimg = {
    1 : one_img,
    2 : two_img,
    3 : three_img,
    4 : four_img,
    5 : five_img,
    6 : six_img,
    7 : seven_img,
    8 : eight_img,
    9 : nine_img
}

## Init the Game Window
def game_initiating_window():
    global probs_numpy
    screen.fill(white)

    for i in range(1, 9):
        ## Veritcal lines
        pg.draw.line(screen, line_color, (cellwidth*i,0), (cellwidth*i,height), 7)

        ## Horizontal lines
        pg.draw.line(screen, line_color, (0, cellheight*i), (width, cellheight*i), 7)

    for iy, ix in np.ndindex(probs_numpy.shape):
        val = probs_numpy[iy, ix]
        if val != 0:
            draw_number(mapnumtoimg[val], iy, ix)
    draw_status()
    
def draw_number(x_img, x, y):
    posx = cellwidth*x + cellwidth/4
    posy = cellheight*y + cellheight/4
    screen.blit(x_img, (posy, posx))

def draw_status():
    global solved

    if solved is None:
        message = "Working"
    else:
        message = solved

    font = pg.font.Font(None, 30)

    text = font.render(message, 1, (255,255,255))

    screen.fill((0,0,0), (0,500,500,500))
    text_rect = text.get_rect(center = (width/2, 600-50))
    screen.blit(text, text_rect)
    pg.display.update()

def eventloop():
    i = 0
    while(True):

        for event in pg.event.get():
            #print(event.type)
    
            if event.type == QUIT:
                print("quit")
                pg.quit()
                sys.exit()
    
        pg.display.update()
        CLOCK.tick(fps)

## Algorithm to solve sudoko problem
def solve_sudoko():
    global probs_numpy, solved
    (r, c) = probs_numpy.shape

    if r != 9 and c != 9:
        return

    if helper(0, 0):
        solved = "Solved"
    else:
        solved = "Invalid Problem"
    draw_status()
    
def helper(r, c):
    global probs_numpy
    if c == 9:
        c = 0
        r = r+1    
    if r == 9:
        return 1

    if probs_numpy[r, c] != 0:
        return helper(r, c+1)
    for i in range(1, 10):
        probs_numpy[r, c] = i
        draw_number(mapnumtoimg[i], r, c)
        time.sleep(0.1)
        if isValid(r, c):
            if helper(r, c+1):
                return 1
        probs_numpy[r, c] = 0
    return 0

def isValid(r, c):
    global probs_numpy

    # check horizontal
    for j in range(0, 9):
        if j != c and probs_numpy[r, j] == probs_numpy[r, c]:
            return 0
    
    # check vertical
    for j in range(0, 9):
        if j != r and probs_numpy[j, c] == probs_numpy[r, c]:
            return 0

    # check grid
    r_s = int(r / 3)*3
    c_s = int(c / 3)*3
    for j in range(r_s, r_s+3):
        for k in range(c_s, c_s+3):
            if j != r and k != c and probs_numpy[j, k] == probs_numpy[r, c]:
                return 0

    return 1

## Sudoko is solved on a thread 
t1 = threading.Thread(target=solve_sudoko)

## Read the Problem into memory
probs = pd.read_csv('example.csv', header = None)
probs.fillna(0, inplace=True)
probs_numpy = probs.values
#print(probs_numpy)

game_initiating_window()

time.sleep(10)

t1.start()

eventloop()