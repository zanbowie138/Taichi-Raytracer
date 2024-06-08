import taichi as ti
import taichi.math as tm

from ray import Ray
from material import hit_record

vec3 = ti.types.vector(3, float)

@ti.dataclass
class Sphere:
    center: vec3
    radius: float
    id: int

    @ti.func
    def hit(self, ray: Ray, ray_tmin: ti.f32, ray_tmax: ti.f32):
        oc = self.center - ray.origin
        a = ray.direction.dot(ray.direction)
        h = oc.dot(ray.direction)
        c = oc.dot(oc) - self.radius * self.radius
        discriminant = h * h - a * c

        hit = hit_record()
        root = 0.0
        did_hit = True

        if discriminant < 0:
            did_hit = False
        else:
            sqrt_d = tm.sqrt(discriminant)
            root = (h - sqrt_d) / a
            if root <= ray_tmin or root >= ray_tmax:
                root = (h + sqrt_d) / a
                if root <= ray_tmin or root >= ray_tmax:
                    did_hit = False

        if did_hit:
            point = ray.at(root)
            outward_normal = (point - self.center) / self.radius
            hit = hit_record(p=point, normal=outward_normal, t=root, id=self.id)
            hit.set_face_normal(ray, outward_normal)

        return did_hit, hit
