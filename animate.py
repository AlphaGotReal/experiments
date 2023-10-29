import os
import time
import sys

import pygame
from pygame.locals import OPENGL, DOUBLEBUF
os.system("clear")

from OpenGL.GL import *
from OpenGL.GLU import gluPerspective

FPS = 100

class Screen():

    def __init__(self, width, height, caption, field_of_view, range_):

        self.width = width
        self.height = height

        self.caption = caption
        self.window = pygame.display.set_mode((width, height), OPENGL|DOUBLEBUF)
        pygame.display.set_caption(caption)

        gluPerspective(field_of_view, (width/height), range_[0], range_[1])
        glTranslate(0, 0, 0)		

        self.clock = pygame.time.Clock()

    def mainloop(self, render_callback):

        for event in pygame.event.get():
            if (event.type == pygame.QUIT):
                pygame.quit()
                exit()

            """key = pygame.key.get_pressed()
            for k in self.actions:
                if (key[ord(k)]):
                    pass"""

        glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT)

        render_callback()

        pygame.display.flip()
        self.clock.tick(FPS)

pygame.font.init()
def rendertext(text, cord, color=(255, 255, 0), font=pygame.font.SysFont("arial", 50), bg=(0,0,0,0)):

    surface = font.render(text, True, color, bg)
    data = pygame.image.tostring(surface, "RGBA", True)
    glWindowPos2d(cord[0], cord[1])
    glDrawPixels(surface.get_width(), surface.get_height(), GL_RGBA, GL_UNSIGNED_BYTE, data)

def plot(x, y, color):

    glColor3fv(color)
    glBegin(GL_LINES)

    for r in range(x.shape[0]-1):

        glVertex3fv((x[r, 0], y[r, 0], -5))
        glVertex3fv((x[r+1, 0], y[r+1, 0], -5))

    glEnd()


