import taichi as ti
import taichi.math as tm
import numpy as np
from ray import Ray
from hittable import Sphere
from world import World
from camera import Camera

ti.init(arch=ti.gpu)

vec3 = ti.types.vector(3, float)

sphere = Sphere(center=vec3(0, 0, -1), radius=0.5)
floor = Sphere(center=vec3(0, -100.5, -1), radius=100)
world = World([sphere, floor])
cam = Camera(800, 600)

def main():
    gui = ti.GUI('Taichi Raytracing', cam.img_res, fast_gui=True)

    while gui.running:
        cam.render(world)
        gui.set_image(cam.frame)
        gui.show()
    ti.tools.imwrite(cam.frame, "output.png")


if __name__ == '__main__':
    main()
