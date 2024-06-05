import taichi as ti
import taichi.math as tm
import utils
from ray import Ray
from world import World


vec3 = ti.types.vector(3, float)

ray_return = ti.types.struct(hit_surface=bool, resulting_ray=Ray, color=vec3)


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

        self.samples_per_pixel = 10
        self.max_ray_depth = 500

    @ti.kernel
    def render(self, world: ti.template()):
        for x, y in self.frame:
            pixel_color = vec3(0, 0, 0)
            for _ in range(self.samples_per_pixel):
                view_ray = self.get_offset_ray(x, y)
                pixel_color += self.get_ray_color(view_ray, world) / self.samples_per_pixel
            self.frame[x, y] = pixel_color

    @ti.func
    def get_ray_color(self, ray: Ray, world: ti.template()) -> vec3:
        color = vec3(1, 1, 1)
        for i in range(self.max_ray_depth):
            ray_ret = self.step_ray(ray, world)
            if ray_ret.hit_surface:
                ray = ray_ret.resulting_ray
                color *= ray_ret.color
            else:
                color *= ray_ret.color
                break
        return color

    @ti.func
    def get_offset_ray(self, x: int, y: int) -> Ray:
        # Generate a random offset within the pixel
        u_offset = ti.random(ti.f32) - 0.5
        v_offset = ti.random(ti.f32) - 0.5

        # Calculate the target point on the viewport
        target = self.first_px + (x + u_offset) * self.px_delta_u - (y + v_offset) * self.px_delta_v

        # Calculate the direction of the ray
        direction = target - self.camera_origin

        # Return the ray
        return Ray(origin=self.camera_origin, direction=direction)

    @ti.func
    def step_ray(self, ray: Ray, world: ti.template()) -> ray_return:
        # TODO: Can optimize by using same space for color and ray
        color = vec3(0, 0, 0)
        hit = world.hit(ray, 0.00, tm.inf)
        resulting_ray = Ray()
        if hit.did_hit:
            scatter = world.materials.scatter(ray, hit.record)
            if scatter.did_scatter:
                color = scatter.attenuation
                origin = scatter.scattered.origin + tm.normalize(scatter.scattered.direction) * .0002
                resulting_ray = Ray(origin=origin, direction=scatter.scattered.direction)
            else:
                color = vec3(0, 0, 0)
        else:
            # Background color
            unit_direction = ray.direction.normalized()
            a = 0.5 * (unit_direction[1] + 1.0)
            color = (1.0 - a) * vec3(1.0, 1.0, 1.0) + a * vec3(0.5, 0.7, 1.0)
        return ray_return(hit_surface=hit.did_hit, resulting_ray=resulting_ray, color=color)


