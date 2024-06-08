import taichi as ti

from material import Materials, hit_record
from ray import Ray
from hittable import Sphere


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
    def hit_world(self, ray: Ray, tmin: ti.f32, tmax: ti.f32):
        """
        Check if the ray hits any of the entities in the world. Returns the closest hit.
        """

        res_did_hit = False
        res_record = hit_record()
        closest_so_far = tmax

        for i in range(self.entities.shape[0]):
            did_hit, record = self.entities[i].hit(ray, tmin, closest_so_far)
            if did_hit:
                closest_so_far = record.t
                res_did_hit = True
                res_record = record

        return res_did_hit, res_record
