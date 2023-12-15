import argparse
import os
import sys
import math
from pathlib import Path
import itertools
import logging
import random
from copy import copy

logging.basicConfig(
    format='[%(asctime)s.%(msecs)03d] [%(name)s] [%(levelname)s] | %(message)s',
    datefmt='%H:%M:%S',
    level=logging.WARNING
)

import bpy
import mathutils

import gin
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pprint import pformat
import imageio

from infinigen.assets.lighting import sky_lighting
from infinigen.assets.scatters import pine_needle

from infinigen.core.placement.factory import AssetFactory
from infinigen.core.placement.camera import spawn_camera, set_active_camera
from infinigen.core.nodes import Nodes, NodeWrangler
from infinigen.core.util import blender as butil
from infinigen.core.util.math import FixedSeed
from infinigen.terrain import Terrain
from infinigen.core.placement import density


from infinigen.core import execute_tasks, surface, init

logging.basicConfig(level=logging.INFO)

# Set ground color
def ground_shader(nw: NodeWrangler):
    # Code generated using version 2.6.5 of the node_transpiler (Blender)
    noise_texture_1 = nw.new_node(Nodes.NoiseTexture, input_kwargs={'Scale': 14.8000, 'Roughness': 0.5375})
    color_ramp = nw.new_node(Nodes.ColorRamp, input_kwargs={'Fac': noise_texture_1.outputs["Fac"]})
    color_ramp.color_ramp.interpolation = "CONSTANT"
    color_ramp.color_ramp.elements[0].position = 0.0000
    color_ramp.color_ramp.elements[0].color = [0.1093, 0.0845, 0.0596, 1.0000]
    color_ramp.color_ramp.elements[1].position = 0.6591
    color_ramp.color_ramp.elements[1].color = [0.0695, 0.0659, 0.0645, 1.0000]
    principled_bsdf = nw.new_node(Nodes.PrincipledBSDF,
        input_kwargs={'Base Color': color_ramp.outputs["Color"], 'Subsurface Color': (0.0711, 0.0814, 0.8000, 1.0000), 'Roughness': 0.3659})
    material_output = nw.new_node(Nodes.MaterialOutput, input_kwargs={'Surface': principled_bsdf}, attrs={'is_active_output': True})

# Create ground geometry/texture
def ground_geometry(nw: NodeWrangler, params: dict):
    # Code generated using version 2.6.5 of the node_transpiler (Blender)
    group_input = nw.new_node(Nodes.GroupInput, expose_input=[('NodeSocketGeometry', 'Geometry', None)])
    subdivide_mesh = nw.new_node(Nodes.SubdivideMesh, input_kwargs={'Mesh': group_input.outputs["Geometry"], 'Level': 6})
    normal = nw.new_node(Nodes.InputNormal)
    noise_texture = nw.new_node(Nodes.NoiseTexture,
        input_kwargs={'Scale': 2.9000, 'Detail': 0.8000, 'Roughness': 0.6042, 'Distortion': 2.6000})
    multiply = nw.new_node(Nodes.Math, input_kwargs={0: noise_texture.outputs["Fac"], 1: params['ground_noise_multiply']}, attrs={'operation': 'MULTIPLY'})
    scale = nw.new_node(Nodes.VectorMath, input_kwargs={0: normal, 'Scale': multiply}, attrs={'operation': 'SCALE'})
    set_position = nw.new_node(Nodes.SetPosition, input_kwargs={'Geometry': subdivide_mesh, 'Offset': scale.outputs["Vector"]})
    group_output = nw.new_node(Nodes.GroupOutput, input_kwargs={'Geometry': set_position}, attrs={'is_active_output': True})

def my_shader(nw: NodeWrangler, params: dict):

    ## TODO: Implement a more complex procedural shader

    noise_texture = nw.new_node(
        Nodes.NoiseTexture, 
        input_kwargs={
            'Scale': params['noise_scale'], 
            'Distortion': params['noise_distortion']
        }
    )
    principled_bsdf = nw.new_node(Nodes.PrincipledBSDF, input_kwargs={'Base Color': noise_texture.outputs["Color"]})
    normal = nw.new_node('ShaderNodeNormal')   
    displacement = nw.new_node('ShaderNodeDisplacement',
        input_kwargs={'Height': noise_texture.outputs["Fac"], 'Scale': 0.02, 'Normal': normal.outputs["Normal"]})
    material_output = nw.new_node(Nodes.MaterialOutput,
        input_kwargs={'Surface': principled_bsdf, 'Displacement': displacement},
        attrs={'is_active_output': True})

# Set leaf color
def leaf_shader(nw: NodeWrangler):
    # Code generated using version 2.6.5 of the node_transpiler (Blender)
    principled_bsdf = nw.new_node(Nodes.PrincipledBSDF, input_kwargs={'Base Color': (0.8000, 0.2148, 0.6995, 1.0000)})
    material_output = nw.new_node(Nodes.MaterialOutput, input_kwargs={'Surface': principled_bsdf}, attrs={'is_active_output': True})

# Set branch color
def branch_shader(nw: NodeWrangler):
    # Code generated using version 2.6.5 of the node_transpiler (Blender)
    principled_bsdf = nw.new_node(Nodes.PrincipledBSDF,
        input_kwargs={'Base Color': (0.0152, 0.0086, 0.0037, 1.0000), 'Roughness': 0.6818, 'Anisotropic Rotation': 0.4227, 'Sheen Tint': 0.2636, 'Clearcoat Roughness': 0.4164, 'IOR': 13.8500, 'Transmission Roughness': 0.3909})
    material_output = nw.new_node(Nodes.MaterialOutput, input_kwargs={'Surface': principled_bsdf}, attrs={'is_active_output': True})

# Create branch geometry/texture
def branch_geometry(nw: NodeWrangler, params: dict):
    # Code generated using version 2.6.5 of the node_transpiler (Blender)
    group_input = nw.new_node(Nodes.GroupInput, expose_input=[('NodeSocketGeometry', 'Geometry', None)])
    distribute_points_on_faces = nw.new_node(Nodes.DistributePointsOnFaces,
        input_kwargs={'Mesh': group_input.outputs["Geometry"], 'Density': params['leaf_density']})
    object_info = nw.new_node(Nodes.ObjectInfo, input_kwargs={'Object': bpy.data.objects['Sphere']})
    instance_on_points = nw.new_node(Nodes.InstanceOnPoints,
        input_kwargs={'Points': distribute_points_on_faces.outputs["Points"], 'Instance': object_info.outputs["Geometry"], 'Rotation': distribute_points_on_faces.outputs["Rotation"]})
    normal = nw.new_node(Nodes.InputNormal)
    noise_texture = nw.new_node(Nodes.NoiseTexture,
        input_kwargs={'Scale': -6.1000, 'Detail': 0.0000, 'Roughness': 1.0000, 'Distortion': 7.0000})
    multiply = nw.new_node(Nodes.Math, input_kwargs={0: 0.1000, 1: noise_texture.outputs["Fac"]}, attrs={'operation': 'MULTIPLY'})
    scale = nw.new_node(Nodes.VectorMath, input_kwargs={0: normal, 'Scale': multiply}, attrs={'operation': 'SCALE'})
    set_position = nw.new_node(Nodes.SetPosition,
        input_kwargs={'Geometry': group_input.outputs["Geometry"], 'Offset': scale.outputs["Vector"]})
    join_geometry = nw.new_node(Nodes.JoinGeometry, input_kwargs={'Geometry': [instance_on_points, set_position]})
    group_output = nw.new_node(Nodes.GroupOutput, input_kwargs={'Geometry': join_geometry}, attrs={'is_active_output': True})

class MyAsset(AssetFactory):

    def __init__(self, factory_seed: int, overrides=None):
        super().__init__(factory_seed)
        
        with FixedSeed(factory_seed):
            self.params = self.sample_params()
            if overrides is not None:
                self.params.update(overrides)

    def sample_params(self):
        return {

            # TODO: Add more randomized parameters

            'length': np.random.uniform(1, 1.5),
            'max_angle': np.random.uniform(45, 75),
            'radius': np.random.uniform(0.02, 0.04),
            'max_branches': np.random.uniform(3, 4),
            'ground_noise_multiply': np.random.uniform(-2, 1),
            'leaf_density': np.random.uniform(7, 14),

        }

    def create_asset(self, **_):

        ## TODO: Implement a more complex procedural mesh

        # Set up initial branch parameters
        main_branch_length = self.params['length']
        max_branches = int(self.params['max_branches'])
        max_angle = math.radians(self.params['max_angle'])

        # Set up initial leaf
        bpy.ops.mesh.primitive_uv_sphere_add(radius=1, location=(0, 0, 0))
        sphere_object = bpy.context.active_object
        surface.add_material(bpy.context.active_object, leaf_shader)
        sphere_object.name = "Sphere"
        sphere_object.location = (2.3006, 0.044221, -1)
        sphere_object.scale = (0.34, 0.12, 0.04)
        bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)

        # Create initial (main) branch
        bpy.ops.mesh.primitive_cylinder_add(radius=0.1, depth=main_branch_length)
        surface.add_geomod(bpy.context.active_object, branch_geometry, input_kwargs=dict(params=self.params))
        surface.add_material(bpy.context.active_object, branch_shader)
        main_branch = bpy.context.active_object

        # Recursively add branches to the end of the main branch
        def recur(main_branch, count):
            num_branches = random.randint(1, max_branches)
            radius_val = self.params['radius']
            if count >= 3:
                return
            for i in range(num_branches):
                bpy.ops.mesh.primitive_cylinder_add(radius=radius_val, depth=main_branch_length)
                branch = bpy.context.active_object

                # Position the branch at the end of the main branch
                branch.location.z = main_branch.location.z + 2 * main_branch.data.vertices[-1].co.z
                old_angle = main_branch.rotation_euler.y
                old_delta_x = math.sin(old_angle) * (main_branch_length / 2)
                old_delta_z = (1 - math.cos(old_angle)) * (main_branch_length / 2)
                branch.location.x = main_branch.location.x
                branch.location.x += old_delta_x
                branch.location.z -= old_delta_z
                branch.location.y = 0
                

                # Rotate the branch
                rand_angle = random.uniform(-max_angle, max_angle)
                branch.rotation_euler.y += rand_angle
                delta_x = math.sin(rand_angle) * (main_branch_length / 2)
                branch.location.x += delta_x
                delta_z = (1 - math.cos(rand_angle)) * (main_branch_length / 2)
                branch.location.z -= delta_z

                # Add shader and geometry
                surface.add_geomod(bpy.context.active_object, branch_geometry, input_kwargs=dict(params=self.params))
                surface.add_material(bpy.context.active_object, branch_shader)
                
                recur(branch, count + 1)
            
        recur(main_branch, 0)
        obj = bpy.context.active_object
        bpy.ops.object.shade_smooth()
        
        return obj

@gin.configurable
def compose_scene(output_folder, scene_seed, overrides=None, **params):

    ## TODO: Customize this function to arrange your scene, or add other assets

    sky_lighting.add_lighting()

    cam = spawn_camera()
    cam.location = (-7, -7, 3.5)
    cam.rotation_euler = np.deg2rad((80, 0, -44))
    set_active_camera(cam)

    factory = MyAsset(factory_seed=np.random.randint(0, 1e7))
    if overrides is not None:
        factory.params.update(overrides)

    factory.spawn_asset(i=np.random.randint(0, 1e7))
    add_ground(factory.params)

def add_ground(params):
    # Create a plane (floor)
    bpy.ops.mesh.primitive_plane_add(size=20, enter_editmode=False, align='WORLD', location=(0, 0, 0))

    # Move the plane to the bottom of the scene
    bpy.context.active_object.location.z = 0

    surface.add_geomod(bpy.context.active_object, ground_geometry, input_kwargs=dict(params=params))
    surface.add_material(bpy.context.active_object, ground_shader)
    
    # Set the plane to shade smooth
    bpy.ops.object.shade_smooth()


def iter_overrides(ranges):
    mid_vals = {k: v[len(v)//2] for k, v in ranges.items()}
    for k, v in ranges.items():
        for vi in v:
            res = copy(mid_vals)
            res[k] = vi
            yield res

def create_param_demo(args, seed):

    override_ranges = {
        'length': np.linspace(1, 1.5, num=3),
        'max_angle': np.linspace(45, 75, num=3),
        'radius': np.linspace(0.02, 0.04, num=3),
        'max_branches': np.linspace(3, 4, num=3),
        'ground_noise_multiply': np.linspace(-2, 1, num=3),
        'leaf_density': np.linspace(7, 14, num=3),
    }
    for i, overrides in enumerate(iter_overrides(override_ranges)):
        
        
        butil.clear_scene()
        print(f'{i=} {overrides=}')
        with FixedSeed(seed):
            compose_scene(args.output_folder, seed, overrides=overrides)
        
        if args.save_blend:
            butil.save_blend(args.output_folder/f'scene_{i}.blend', verbose=True)

        bpy.context.scene.frame_set(i)
        bpy.context.scene.frame_start = i
        bpy.context.scene.frame_end = i
        bpy.ops.render.render(animation=True)

        imgpath = args.output_folder/f'{i:04d}.png'
        img = Image.open(imgpath)
        ImageDraw.Draw(img).text(
            xy=(10, 10), 
            text='\n'.join(f'{k}: {v:.2f}' for k, v in overrides.items()), 
            fill=(76, 252, 85),
            # font=ImageFont.truetype("arial.ttf", size=50)
            font=ImageFont.load_default()
        )
        img.save(imgpath)
        

def create_video(args, seed):
    butil.clear_scene()
    with FixedSeed(seed):
        compose_scene(args.output_folder, seed)

    butil.save_blend(args.output_folder/'scene.blend', verbose=True)

    bpy.context.scene.frame_start = 0
    bpy.context.scene.frame_end = args.duration_frames
    bpy.ops.render.render(animation=True)



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--output_folder', type=Path)
    parser.add_argument('--mode', type=str, choices=['param_demo', 'video'])
    parser.add_argument('--duration_frames', type=int, default=1)
    parser.add_argument('--save_blend', action='store_true')
    parser.add_argument('-s', '--seed', default=None, help="The seed used to generate the scene")
    parser.add_argument('-g', '--configs', nargs='+', default=['base'],
                        help='Set of config files for gin (separated by spaces) '
                             'e.g. --gin_config file1 file2 (exclude .gin from path)')
    parser.add_argument('-p', '--overrides', nargs='+', default=[],
                        help='Parameter settings that override config defaults '
                             'e.g. --gin_param module_1.a=2 module_2.b=3')
    parser.add_argument('-d', '--debug', action="store_const", dest="loglevel", const=logging.DEBUG, default=logging.INFO)

    args = init.parse_args_blender(parser)
    logging.getLogger("infinigen").setLevel(args.loglevel)

    seed = init.apply_scene_seed(args.seed)
    init.apply_gin_configs(
        configs=args.configs, 
        overrides=args.overrides,
        configs_folder='infinigen_examples/configs'
    )

    bpy.context.scene.render.engine = 'CYCLES'
    bpy.context.scene.cycles.samples = 50

    args.output_folder.mkdir(exist_ok=True, parents=True)
    bpy.context.scene.render.filepath = str(args.output_folder.absolute()) + '/'


    if args.mode == 'param_demo':
        create_param_demo(args, seed)
    elif args.mode == 'video':
        create_video(args, seed)
    else:
        raise ValueError(f'Unrecognized {args.mode=}')
    