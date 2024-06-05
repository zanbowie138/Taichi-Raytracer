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
def near_zero(v: vec3) -> bool:
    s = 1e-8
    return (v[0] < s) and (v[1] < s) and (v[2] < s)

@ti.func
def random_unit_vector() -> vec3:
    p = vec3(0, 0, 0)
    while True:
        p = vec3(ti.random(ti.f32), ti.random(ti.f32), ti.random(ti.f32)) - vec3(0.5, 0.5, 0.5)
        if p.dot(p) <= 1.0:
            p = tm.normalize(p)
            break
    return p

@ti.func
def linear_to_gamma(x: float) -> float:
    return tm.pow(x, 1 / 2.2) if x > 0 else 0

@ti.func
def reflectance(cosine: float, ref_idx: float) -> float:
    # Schlick's approximation
    r0 = (1 - ref_idx) / (1 + ref_idx)
    r0 = r0 * r0
    return r0 + (1 - r0) * tm.pow(1 - cosine, 5)

@ti.func
def linear_to_gamma_vec3(x: vec3) -> vec3:
    return vec3(linear_to_gamma(x[0]), linear_to_gamma(x[1]), linear_to_gamma(x[2]))
