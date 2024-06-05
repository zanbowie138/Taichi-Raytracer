import taichi as ti
import taichi.math as tm

import utils
from ray import Ray

vec3 = ti.types.vector(3, float)


@ti.dataclass
class hit_record:
    """
    Struct type storing the hit record of a ray and the object it hit.

    Parameters
    ----------
    p : vec3
        The point where the ray hit the object.
    normal : vec3
        The normal vector of the object at the hit point.
    id : int
        The index of the object in the world.
    t : float
        The time at which the ray hit the object.
    front_face : bool
        Whether the ray hit the front face of the object.
    """

    p: vec3
    normal: vec3
    id: int
    t: float
    front_face: bool

    @ti.func
    def set_face_normal(self, ray: Ray, outward_normal: vec3):
        self.front_face = ray.direction.dot(self.normal) < 0
        self.normal = outward_normal if self.front_face else -outward_normal


@ti.dataclass
class scatter_return:
    did_scatter: bool
    attenuation: vec3
    scattered: Ray

@ti.func
def reflect(vec: vec3, norm: vec3) -> vec3:
    return vec - 2 * vec.dot(norm) * norm


@ti.data_oriented
class Materials:
    LAMBERT = 0
    METAL = 1
    DIELECTRIC = 2

    def __init__(self, n: int):
        self.roughness = ti.field(ti.f32)
        self.albedo = ti.field(vec3)
        self.mat_index = ti.field(ti.i32)
        self.ior = ti.field(ti.f32)
        ti.root.dense(ti.i, n).place(self.roughness, self.albedo, self.mat_index, self.ior)

    def set(self, i: int, material):
        self.roughness[i] = material.roughness
        self.albedo[i] = material.albedo
        self.mat_index[i] = material.index
        self.ior[i] = material.ior

    @ti.func
    def scatter(self, ray: Ray, record: hit_record) -> scatter_return:
        mat_idx = self.mat_index[record.id]
        scatter_ret = scatter_return()
        if mat_idx == Materials.LAMBERT:
            scatter_ret = Lambert.scatter(ray, record, self.albedo[record.id])
        if mat_idx == Materials.METAL:
            scatter_ret = Metal.scatter(ray, record, self.albedo[record.id], self.roughness[record.id])
        return scatter_ret


class Lambert:
    def __init__(self, albedo: vec3):
        self.albedo = albedo
        self.index = Materials.LAMBERT
        self.roughness = 0.0
        self.ior = 1.0

    @staticmethod
    @ti.func
    def scatter(ray: Ray, record: hit_record, albedo: vec3) -> scatter_return:
        scatter_direction = record.normal + utils.random_unit_vector()
        scattered = Ray(record.p, scatter_direction)
        attenuation = albedo
        return scatter_return(did_scatter=True, attenuation=attenuation, scattered=scattered)

class Metal:
    def __init__(self, albedo: vec3, roughness: float):
        self.albedo = albedo
        self.index = Materials.METAL
        self.roughness = min(roughness, 1.0)
        self.ior = 1.0

    @staticmethod
    @ti.func
    def scatter(ray: Ray, record: hit_record, albedo: vec3, roughness: float) -> scatter_return:
        reflected = reflect(ray.direction.normalized(), record.normal)
        scattered = Ray(record.p, reflected + roughness * utils.random_unit_vector())
        attenuation = albedo
        return scatter_return(did_scatter=True, attenuation=attenuation, scattered=scattered)
