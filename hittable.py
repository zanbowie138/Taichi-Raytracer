import taichi as ti
import taichi.math as tm

from ray import Ray

vec3 = ti.types.vector(3, float)
hit_record = ti.types.struct(p=vec3, normal=vec3, t=float)
@ti.dataclass
class hit_record:
    p: vec3
    normal: vec3
    t: float
    front_face: bool

    @ti.func
    def set_face_normal(self, ray: Ray, outward_normal: vec3):
        self.front_face = ray.direction.dot(self.normal) < 0
        self.normal = outward_normal if self.front_face else -outward_normal
hit_return = ti.types.struct(did_hit=bool, record=hit_record)

@ti.dataclass
class Sphere:
    center: vec3
    radius: float

    @ti.func
    def hit(self, ray: Ray, ray_tmin: float, ray_tmax: float) -> hit_return:
        oc = self.center - ray.origin
        a = ray.direction.dot(ray.direction)
        h = oc.dot(ray.direction)
        c = oc.dot(oc) - self.radius * self.radius
        discriminant = h * h - a * c

        hit = hit_record()
        root = 0.0
        ret = True

        if discriminant < 0:
            ret = False
        else:
            sqrt_d = tm.sqrt(discriminant)
            root = (h - sqrt_d) / a
            if root <= ray_tmin or root >= ray_tmax:
                root = (h + sqrt_d) / a
                if root <= ray_tmin or root >= ray_tmax:
                    ret = False

        if ret:
            point = ray.at(root)
            hit = hit_record(p=point, normal=(point - self.center) / self.radius, t=root)
            outward_normal = (point - self.center) / self.radius
            hit.set_face_normal(ray, outward_normal)

        return hit_return(did_hit=ret, record=hit)
