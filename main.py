import taichi as ti
import taichi.math as tm
import numpy as np
from ray import Ray
from hittable import Sphere
from world import World
from camera import Camera
import utils
import time

ti.init(arch=ti.gpu)

vec3 = ti.types.vector(3, float)

sphere = Sphere(center=vec3(0, 0, -1), radius=0.5)
floor = Sphere(center=vec3(0, -100.5, -1), radius=100)
world = World([sphere, floor])
cam = Camera(800, 600)

def main():
    # start_time = time.perf_counter()
    # print(f"Rendering time: {time.perf_counter() - start_time:.2f}s")

    gui = ti.GUI('Taichi Raytracing', cam.img_res, fast_gui=True)
    current_frame = ti.Vector.field(n=3, dtype=ti.f32, shape=cam.img_res)
    rendered_frames = 0
    while gui.running:
        cam.render(world)
        weight = 1.0 / (rendered_frames + 1)
        average_frames(current_frame, cam.frame, weight)
        gui.set_image(current_frame)
        gui.show()
        rendered_frames += 1
    ti.tools.imwrite(cam.frame, "output.png")

@ti.kernel
def average_frames(current_frame: ti.template(), new_frame: ti.template(), weight: float):
    for i, j in new_frame:
        current_frame[i, j] = (1.0 - weight) * current_frame[i, j] + weight * utils.linear_to_gamma_vec3(new_frame[i, j])


if __name__ == '__main__':
    main()
