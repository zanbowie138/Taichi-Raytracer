import taichi as ti
import taichi.math as tm
from ray import Ray
from world import World

vec3 = ti.types.vector(3, float)


@ti.data_oriented
class Camera:
    def __init__(self, width, height):
        self.img_res = (width, height)

        # Virtual rectangle in scene that camera sends rays through
        focal_length = 1.0
        viewport_height = 2.0
        viewport_width = viewport_height * width / height
        self.camera_origin = vec3(0, 0, 0)

        # Calculate the vectors across the horizontal and down the vertical viewport edges.
        viewport_u = vec3(viewport_width, 0, 0)
        viewport_v = vec3(0, -viewport_height, 0)

        # Calculate per-pixel deltas
        self.px_delta_u = viewport_u / width
        self.px_delta_v = viewport_v / height

        # Calculate the upper-left corner of the viewport
        viewport_ul = self.camera_origin - ti.Vector([0, 0, focal_length]) - viewport_u / 2 + viewport_v / 2
        self.first_px = viewport_ul + self.px_delta_u / 2 - self.px_delta_v / 2
        self.frame = ti.Vector.field(n=3, dtype=ti.f32, shape=self.img_res)

    @ti.kernel
    def render(self, world: ti.template()):
        for x, y in self.frame:
            px_center = self.first_px + x * self.px_delta_u - y * self.px_delta_v
            ray_dir = px_center - self.camera_origin
            view_ray = Ray(origin=self.camera_origin, direction=ray_dir)
            self.frame[x, y] = self.ray_color(view_ray, world)

    @ti.func
    def ray_color(self, ray: Ray, world: ti.template()) -> vec3:
        color = vec3(0, 0, 0)
        hit = world.hit(ray, 0, tm.inf)
        if hit.did_hit:
            norm = hit.record.normal
            color = 0.5 * vec3(norm[0] + 1, norm[1] + 1, norm[2] + 1)
        else:
            unit_direction = ray.direction.normalized()
            a = 0.5 * (unit_direction[1] + 1.0)
            color = (1.0 - a) * vec3(1.0, 1.0, 1.0) + a * vec3(0.5, 0.7, 1.0)
        return color
