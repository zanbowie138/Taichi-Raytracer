import taichi as ti
from hittable import Sphere
from world import World
from camera import Camera
import material
import utils
import random
import time

ti.init(arch=ti.cuda)

vec3 = ti.types.vector(3, float)
spheres = []
materials = []

right = Sphere(center=vec3(0, 0, -1001.5), radius=1000)
right_mat = material.Lambert(vec3(0.01, 0.9, 0.01))
spheres.append(right)
materials.append(right_mat)

left = Sphere(center=vec3(0, 0, 1001.5), radius=1000)
left_mat = material.Lambert(vec3(0.9, 0.01, 0.01))
spheres.append(left)
materials.append(left_mat)

front = Sphere(center=vec3(1000, 0, 0), radius=1000)
front_mat = material.Lambert(vec3(0.9, 0.9, 0.9))
spheres.append(front)
materials.append(front_mat)

back = Sphere(center=vec3(-1001, 0, 0), radius=1000)
back_mat = material.Lambert(vec3(0.9, 0.9, 0.9))
spheres.append(back)
materials.append(back_mat)


ceil_l = Sphere(center=vec3(0, 1002, 1.0), radius=1000.0)
ceil_l_mat = material.Lambert(vec3(0.8, 0.8, 0.8))
spheres.append(ceil_l)
materials.append(ceil_l_mat)

ceil_r = Sphere(center=vec3(0, 1002, -1.0), radius=1000.0)
ceil_r_mat = material.Lambert(vec3(0.8, 0.8, 0.8))
spheres.append(ceil_r)
materials.append(ceil_r_mat)

light = Sphere(center=vec3(0, 1002, 0), radius=1000)
light_mat = material.Light(vec3(0.8, 0.8, 0.8), 10.0)
spheres.append(light)
materials.append(light_mat)

ceil_f = Sphere(center=vec3(-2, 1002, 0), radius=1000.01)
ceil_f_mat = material.Lambert(vec3(0.8, 0.8, 0.8))
spheres.append(ceil_f)
materials.append(ceil_f_mat)

ceil_b = Sphere(center=vec3(15, 1002, 0), radius=1000.0)
ceil_b_mat = material.Lambert(vec3(0.8, 0.8, 0.8))
spheres.append(ceil_b)
materials.append(ceil_b_mat)

floor = Sphere(center=vec3(0, -1001, 0), radius=1000)
floor_mat = material.Lambert(vec3(0.8, 0.8, 0.8))
spheres.append(floor)
materials.append(floor_mat)

sph1 = Sphere(center=vec3(0, -0.2, -0.3), radius=0.5)
sph1_mat = material.Metal(vec3(0.8, 0.6, 0.2), 0.1)
spheres.append(sph1)
materials.append(sph1_mat)

sph2 = Sphere(center=vec3(1.5, 0.5, 0.6), radius=0.8)
sph2_mat = material.Dielectric(1.5)
spheres.append(sph2)
materials.append(sph2_mat)

sph3 = Sphere(center=vec3(1, 0.9, 0.7), radius=0.3)
sph3_mat = material.Lambert(vec3(0.2, 0.6, 0.8))
spheres.append(sph3)
materials.append(sph3_mat)

for a in range(-2, 2):
    for b in range(-2, 2):
        choose_mat = random.random()
        center = vec3(a/2.0 + 2 * random.random() + 4, random.random() * 2 - 0.45, b/2.0 + 0.9 * random.random())
        if (center - vec3(4, 0.2, 0)).norm() > 0.9:
            if choose_mat < 0.6:
                # diffuse
                spheres.append(Sphere(center=center, radius=random.random() * 0.2 + 0.1))
                materials.append(material.Lambert(utils.rand_vec(0,1) * utils.rand_vec(0,1)))
            elif choose_mat < 0.85:
                # metal
                spheres.append(Sphere(center=center, radius=random.random() * 0.2 + 0.1))
                materials.append(material.Metal(utils.rand_vec(0.5, 1), 0.5 * random.random()))
            else:
                # glass
                spheres.append(Sphere(center=center, radius=random.random() * 0.2 + 0.1))
                materials.append(material.Dielectric(1.5))


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
