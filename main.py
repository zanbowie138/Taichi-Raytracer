import taichi as ti
import taichi.math as tm

ti.init(arch=ti.gpu)

vec3 = ti.types.vector(3, float)

width = 800
height = 600
img_res = (width, height)

# Virtual rectangle in scene that camera sends rays through
focal_length = 1.0
viewport_height = 2.0
viewport_width = viewport_height * width / height
camera_origin = vec3(0, 0, 0)

# Calculate the vectors across the horizontal and down the vertical viewport edges.
viewport_u = vec3(viewport_width, 0, 0)
viewport_v = vec3(0, -viewport_height, 0)

# Calculate per-pixel deltas
px_delta_u = viewport_u / width
px_delta_v = viewport_v / height

# Calculate the upper-left corner of the viewport
viewport_ul = camera_origin - ti.Vector([0, 0, focal_length]) - viewport_u / 2 + viewport_v / 2
first_px = viewport_ul + px_delta_u / 2 - px_delta_v / 2

frame = ti.Vector.field(n=3, dtype=ti.f32, shape=img_res)

@ti.dataclass
class Ray:
    origin: vec3
    direction: vec3

    @ti.func
    def at(self, t: ti.f32) -> ti.math.vec3:
        return self.origin + t * self.direction


@ti.kernel
def render_frame():
    for x, y in frame:
        px_center = first_px + x * px_delta_u - y * px_delta_v
        ray_dir = px_center - camera_origin
        view_ray = Ray(origin=camera_origin, direction=ray_dir)
        frame[x, y] = ray_color(view_ray)


@ti.func
def ray_color(ray: Ray) -> vec3:
    color = vec3(0, 0, 0)
    t = hit_sphere(vec3(0, 0, -1), 0.5, ray)
    if t > 0.0:
        norm = (ray.at(t) - vec3(0, 0, -1)).normalized()
        color = 0.5 * vec3(norm[0] + 1, norm[1] + 1, norm[2] + 1)
    else:
        unit_direction = ray.direction.normalized()
        a = 0.5 * (unit_direction[1] + 1.0)
        color = (1.0 - a) * vec3(1.0, 1.0, 1.0) + a * vec3(0.5, 0.7, 1.0)
    return color

@ti.func
def hit_sphere(center: vec3, radius: float, ray: Ray) -> float:
    oc = center - ray.origin
    a = ray.direction.dot(ray.direction)
    h = oc.dot(ray.direction)
    c = oc.dot(oc) - radius*radius
    discriminant = h*h - a*c

    ret = -1.0 if discriminant < 0 else (h - tm.sqrt(discriminant)) / a
    return ret


def main():
    gui = ti.GUI('Taichi Raytracing', img_res, fast_gui=True)

    while gui.running:
        render_frame()
        gui.set_image(frame)
        gui.show()
    ti.tools.imwrite(frame, "output.png")


if __name__ == '__main__':
    main()
