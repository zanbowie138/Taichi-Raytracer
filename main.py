import taichi as ti
from hittable import Sphere
from world import World
from camera import Camera
import material
import utils
import random
import time

ti.init(arch=ti.gpu)

vec3 = ti.types.vector(3, float)
spheres = []
materials = []

floor = Sphere(center=vec3(0, -1000, -1), radius=1000)
floor_mat = material.Lambert(vec3(0.5, 0.5, 0.5))
spheres.append(floor)
materials.append(floor_mat)

for a in range(-11, 11):
    for b in range(-11, 11):
        choose_mat = random.random()
        center = vec3(a + 0.9 * random.random(), 0.2, b + 0.9 * random.random())
        if (center - vec3(4, 0.2, 0)).norm() > 0.9:
            if choose_mat < 0.8:
                # diffuse
                spheres.append(Sphere(center=center, radius=0.2))
                materials.append(material.Lambert(utils.rand_vec(0,1) * utils.rand_vec(0,1)))
            elif choose_mat < 0.95:
                # metal
                spheres.append(Sphere(center=center, radius=0.2))
                materials.append(material.Metal(utils.rand_vec(0.5, 1), 0.5 * random.random()))
            else:
                # glass
                spheres.append(Sphere(center=center, radius=0.2))
                materials.append(material.Dielectric(1.5))

sph_1 = Sphere(center=vec3(0, 1, 0), radius=1)
sph_1_mat = material.Dielectric(1.5)
spheres.append(sph_1)
materials.append(sph_1_mat)

sph_2 = Sphere(center=vec3(-4, 1, 0), radius=1)
sph_2_mat = material.Lambert(vec3(0.4, 0.2, 0.1))
spheres.append(sph_2)
materials.append(sph_2_mat)

sph_3 = Sphere(center=vec3(4, 1, 0), radius=1)
sph_3_mat = material.Metal(vec3(0.7, 0.6, 0.5), 0.0)
spheres.append(sph_3)
materials.append(sph_3_mat)

world = World(spheres, materials)
cam = Camera()

def main():
    # start_time = time.perf_counter()
    # print(f"Rendering time: {time.perf_counter() - start_time:.2f}s")

    gui = ti.GUI('Taichi Raytracing', cam.img_res, fast_gui=True)
    current_frame = ti.Vector.field(n=3, dtype=ti.f32, shape=cam.img_res)
    rendered_frames = 0
    while gui.running:
        weight = 1.0 / (rendered_frames + 1)
        cam.render(world)
        average_frames(current_frame, cam.frame, weight)
        gui.set_image(current_frame)
        gui.show()
        rendered_frames += 1
    ti.tools.imwrite(current_frame, "output.png")

@ti.kernel
def average_frames(current_frame: ti.template(), new_frame: ti.template(), weight: float):
    for i, j in new_frame:
        current_frame[i, j] = (1.0 - weight) * current_frame[i, j] + weight * utils.linear_to_gamma_vec3(new_frame[i, j])


if __name__ == '__main__':
    main()
