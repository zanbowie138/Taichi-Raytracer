import taichi as ti
import taichi.math as tm

vec3 = ti.types.vector(3, float)


@ti.func
def random_on_hemisphere(normal: vec3) -> vec3:
    vec = random_unit_vector()
    if vec.dot(normal) < 0.0:
        vec = -vec
    return vec


@ti.func
def random_unit_vector() -> vec3:
    p = vec3(0, 0, 0)
    while True:
        p = 2.0 * vec3(ti.random(ti.f32), ti.random(ti.f32), ti.random(ti.f32)) - vec3(1, 1, 1)
        if p.dot(p) < 1.0:
            p = tm.normalize(p)
            break
    return p

@ti.func
def linear_to_gamma(x: float) -> float:
    return tm.pow(x, 1 / 2.2) if x > 0 else 0

@ti.func
def linear_to_gamma_vec3(x: vec3) -> vec3:
    return vec3(linear_to_gamma(x[0]), linear_to_gamma(x[1]), linear_to_gamma(x[2]))
