import taichi as ti

from material import Materials
from ray import Ray
from hittable import hit_return, Sphere


@ti.data_oriented
class World:
    def __init__(self, entities, materials):
        self.entities = Sphere.field(shape=len(entities))
        self.materials = Materials(len(materials))
        for i, entity in enumerate(entities):
            self.entities[i] = entity
            self.entities[i].id = i
            self.materials.set(i, materials[i])


    @ti.func
    def hit(self, ray: Ray, tmin: float, tmax: float) -> hit_return:
        hit = hit_return()
        closest_so_far = tmax

        for i in range(self.entities.shape[0]):
            entity_hit_return = self.entities[i].hit(ray, tmin, closest_so_far)
            if entity_hit_return.did_hit:
                closest_so_far = entity_hit_return.record.t
                hit = entity_hit_return

        return hit
