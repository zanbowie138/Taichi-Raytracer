import taichi as ti
import taichi.math as tm

from ray import Ray
from material import hit_record

vec3 = ti.types.vector(3, float)

@ti.dataclass
class hit_return:
    """
    Struct type storing the hit record and a boolean indicating whether the ray hit the object.
    """
    did_hit: bool
    record: hit_record

@ti.dataclass
class Sphere:
    center: vec3
    radius: float
    id: int

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
            hit = hit_record(p=point, normal=(point - self.center) / self.radius, t=root, id=self.id)
            outward_normal = (point - self.center) / self.radius
            hit.set_face_normal(ray, outward_normal)

        return hit_return(did_hit=ret, record=hit)
