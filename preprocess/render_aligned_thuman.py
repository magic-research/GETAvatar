"""
Rendering script for THuman2.0 dataset. 

Randomizes views in a sphere of fixed radius around a unit normalized
mesh centered at the origin. Applies environment lighting using HDRI.

Use as 
    `blender --background test.blend --python render_aligned_thuman.py -- \
    --device_id 0 --tot 1 --id 0`
asd

"""

import argparse, sys, os, math, re
import bpy
import json
from glob import glob
import numpy as np
from mathutils import Vector
import copy

np.random.seed(1111)
parser = argparse.ArgumentParser()
parser.add_argument('--device_id', type=int, default=0)
parser.add_argument('--root_dir', type=str, default='datasets/THuman2.0/THuman2.0_aligned_scans')
parser.add_argument('--output_dir', type=str, default='datasets/THuman2.0/THuman2.0_res512')
parser.add_argument('--id', type=int, default=0)
parser.add_argument('--tot', type=int, default=1)
parser.add_argument('--smpl_json_path', type=str, default='thuman_json_files/thuman_smpl_params.json')


def listify_matrix(matrix):
    matrix_list = []
    for row in matrix:
        matrix_list.append(list(row))
    return matrix_list

def listify_vector(vec):
    return list(vec)

def extract_args(input_argv=None):
	"""
	Pull out command-line arguments after "--". Blender ignores command-line flags
	after --, so this lets us forward command line arguments from the blender
	invocation to our own script.
	"""
	if input_argv is None:
		input_argv = sys.argv
	output_argv = []
	if '--' in input_argv:
		idx = input_argv.index('--')
		output_argv = input_argv[(idx + 1):]
	return output_argv

class THuman:
    
    def __init__(self):
        self.hdri = 'white_bg.png'
        # Set filenames for albedo, normal and mask images
        self.albedo_file_output = None
        self.normal_file_output = None
        self.id_file_output = None

        # Configs
        self.format = 'PNG'
        self.color_depth = '8'
        self.color_mode = 'RGBA'
        self.resolution = 512
        self.tile_size = 256
        # NOTE: for debug
        #self.views = 2
        self.views = 100
        self.scale = 1
        # NOTE: for debug
        self.RANDOM = True
        
        # Set up rendering
        self.context = bpy.context
        self.scene = bpy.context.scene
        self.render = bpy.context.scene.render

    
    def setup_scene(self, device_id):
    
        # Configs
        self.render.engine = 'CYCLES' # or BLENDER_EEVEE
        self.scene.cycles.device = 'GPU'

        # Use GPUs
        cycles_prefs = self.context.preferences.addons['cycles'].preferences
        cycles_prefs.get_devices()
        cycles_prefs.compute_device_type = 'CUDA'

        for id, device in enumerate(cycles_prefs.devices):
            
            if id in [device_id]:
                device.use = True
                print(f"GPU {id} enabled")
            else:
                device.use = False

        self.render.image_settings.file_format = self.format # ('PNG', 'OPEN_EXR', 'JPEG, ...)
        self.render.image_settings.color_mode = self.color_mode # ('RGB', 'RGBA', ...)
        self.render.image_settings.color_depth = self.color_depth # ('8', '16')
        self.render.resolution_x = self.resolution
        self.render.resolution_y = self.resolution
        self.scene.cycles.tile_size = self.tile_size
        self.render.resolution_percentage = 100
        self.render.film_transparent = True

        # Improve rendering time -- gives more mem. to renderer
        self.scene.render.use_lock_interface = True        

        # Compositing nodes (albedo, normal, mask, depth)
        self.scene.use_nodes = True
        #self.scene.view_settings.view_transform = 'Filmic'
        self.scene.view_settings.view_transform = 'Standard'
        self.scene.view_layers["ViewLayer"].use_pass_normal = True
        self.scene.view_layers["ViewLayer"].use_pass_diffuse_color = True
        self.scene.view_layers["ViewLayer"].use_pass_object_index = True

        nodes = self.scene.node_tree.nodes
        links = self.scene.node_tree.links

        # Clear default nodes
        for n in nodes:
            nodes.remove(n)

        # Create input render layer node
        render_layers = nodes.new('CompositorNodeRLayers')
            
        # Create normal output nodes
        scale_node = nodes.new(type="CompositorNodeMixRGB")
        scale_node.blend_type = 'MULTIPLY'
        # scale_node.use_alpha = True
        scale_node.inputs[2].default_value = (0.5, 0.5, 0.5, 1)
        links.new(render_layers.outputs['Normal'], scale_node.inputs[1])

        bias_node = nodes.new(type="CompositorNodeMixRGB")
        bias_node.blend_type = 'ADD'
        # bias_node.use_alpha = True
        bias_node.inputs[2].default_value = (0.5, 0.5, 0.5, 0)
        links.new(scale_node.outputs[0], bias_node.inputs[1])

        self.normal_file_output = nodes.new(type="CompositorNodeOutputFile")
        self.normal_file_output.label = 'Normal Output'
        self.normal_file_output.base_path = ''
        self.normal_file_output.file_slots[0].use_node_format = True
        self.normal_file_output.format.file_format = self.format
        links.new(bias_node.outputs[0], self.normal_file_output.inputs[0])

        # Create albedo output nodes
        alpha_albedo = nodes.new(type="CompositorNodeSetAlpha")
        links.new(render_layers.outputs['DiffCol'], alpha_albedo.inputs['Image'])
        links.new(render_layers.outputs['Alpha'], alpha_albedo.inputs['Alpha'])

        self.albedo_file_output = nodes.new(type="CompositorNodeOutputFile")
        self.albedo_file_output.label = 'Albedo Output'
        self.albedo_file_output.base_path = ''
        self.albedo_file_output.file_slots[0].use_node_format = True
        self.albedo_file_output.format.file_format = self.format
        self.albedo_file_output.format.color_mode = self.color_mode
        self.albedo_file_output.format.color_depth = self.color_depth
        links.new(alpha_albedo.outputs['Image'], self.albedo_file_output.inputs[0])

        # Create id map output nodes
        self.id_file_output = nodes.new(type="CompositorNodeOutputFile")
        self.id_file_output.label = 'ID Output'
        self.id_file_output.base_path = ''
        self.id_file_output.file_slots[0].use_node_format = True
        self.id_file_output.format.file_format = self.format
        self.id_file_output.format.color_depth = self.color_depth

        if format == 'OPEN_EXR':
            links.new(render_layers.outputs['IndexOB'], self.id_file_output.inputs[0])
        else:
            self.id_file_output.format.color_mode = 'BW'

            divide_node = nodes.new(type='CompositorNodeMath')
            divide_node.operation = 'DIVIDE'
            divide_node.use_clamp = False
            divide_node.inputs[1].default_value = 2**int(self.color_depth)

            links.new(render_layers.outputs['IndexOB'], divide_node.inputs[0])
            links.new(divide_node.outputs[0], self.id_file_output.inputs[0])
        

    def render_obj(self, in_dir, out_dir, y_median=0.0):
        """    
        Render a single human in turntable setup
        """    
        in_file = None     
        obj_name = None
       
        # +ve values -- camera looks upward i.e. camera below origin
        CIRCLE_FIXED_START = (-.1,0,0)
        CIRCLE_FIXED_END = (.1,0,0)

        mesh = []
        for file in os.listdir(in_dir):
            if '.obj' in file.lower():
                mesh.append(file)
                
        in_file = os.path.join(in_dir, mesh[0])
        obj_name = mesh[0][:-4] # Remove file ext

        if in_file is None:
            print("FILE NOT FOUND IN " + in_dir)
        
        world_nodes = self.scene.world.node_tree.nodes
        world_links = self.scene.world.node_tree.links

        world_nodes.clear()

        # Add HDRI environment lighting
        node_background = world_nodes.new(type='ShaderNodeBackground')

        # Add Environment Texture node
        node_environment = world_nodes.new(type='ShaderNodeTexEnvironment')
        # Load and assign the image to the node property
        #node_environment.image = bpy.data.images.load(hdri)
        node_environment.image = bpy.data.images.load(self.hdri)
        node_environment.location = -300,0

        # Add Output node
        node_output = world_nodes.new(type='ShaderNodeOutputWorld')   
        node_output.location = 200,0

        # Link all nodes
        link = world_links.new(node_environment.outputs["Color"], node_background.inputs["Color"])
        link = world_links.new(node_background.outputs["Background"], node_output.inputs["Surface"])
        
        # Import textured mesh
        bpy.ops.object.select_all(action='DESELECT')

        bpy.ops.import_scene.obj(filepath=in_file)

        obj = bpy.context.selected_objects[0]
        self.context.view_layer.objects.active = obj
        obj.select_set(True)


        median = listify_vector(copy.deepcopy(obj.location))
        obj.location = (0.0, 0.0, 0.0)
        median_after = listify_vector(copy.deepcopy(obj.location))

        obj.select_set(False)
        self.context.view_layer.objects.active = None

        # Disable specular and metallic shading
        for slot in obj.material_slots:
            node = slot.material.node_tree.nodes['Principled BSDF']
            node.inputs['Specular'].default_value = 0.0
            node.inputs['Metallic'].default_value = 0.0

        # Set object IDs
        obj.pass_index = 255

        # Place camera
        cam = bpy.data.scenes['Scene'].camera #scene.objects['Camera']
        cam.location = (0, -2.3, y_median)
        ##cam.rotation_euler = (math.pi/2, 0, 0)
        cam.data.lens = 35
        cam.data.sensor_width = 32
        cam.data.clip_start = 0.01
        cam.data.clip_end = 100

        cam_constraint = cam.constraints.new(type='TRACK_TO')
        cam_constraint.track_axis = 'TRACK_NEGATIVE_Z'
        cam_constraint.up_axis = 'UP_Y'

        cam_empty = bpy.data.objects.new("Empty", None)
        cam_empty.location = (0, 0, y_median)
        cam.parent = cam_empty

        self.scene.collection.objects.link(cam_empty)
        self.context.view_layer.objects.active = cam_empty
        cam_constraint.target = cam_empty

        stepsize = 360.0 / self.views
        vertical_diff = CIRCLE_FIXED_END[0] - CIRCLE_FIXED_START[0]
        rotation_mode = 'XYZ'

        # -------------------------------------------------------- #
        out_data = {
            'camera_angle_x': bpy.data.scenes['Scene'].camera.data.angle_x,
            'hdri': self.hdri,
            'median': median,
            'median_after': median_after,
            'y_median': y_median
        }

        out_data['frames'] = list()
        
        cam_empty.rotation_euler = (0.0, 0.0, 0.0)

        for i in range(0, self.views):
            print("Rotation {}, {}".format((stepsize * i), math.radians(stepsize * i)))

            render_file_path = os.path.join(out_dir, str(i))

            self.scene.render.filepath = render_file_path

            bpy.ops.render.render(write_still=True)

            frame_data = {
                'file_path': render_file_path + '.png',
                'blender_transform_matrix': listify_matrix(cam.matrix_world)
            }
            out_data['frames'].append(frame_data)


            if self.RANDOM:
                cam_empty.rotation_euler[0] = CIRCLE_FIXED_START[0] + np.random.rand() * vertical_diff
                cam_empty.rotation_euler[2] = np.random.rand() * 2 * np.pi 
            
            else:
                cam_empty.rotation_euler[0] = CIRCLE_FIXED_START[0] + (np.cos(math.radians(stepsize*i))+1)/2 * vertical_diff
                cam_empty.rotation_euler[2] += math.radians(stepsize)

        # Remove weird naming issue for saved files
        
        for f in os.listdir(out_dir):        
            if self.format.lower() in f and '_' in f:
                fname = '_'.join(f.split('_')[:-1]) + f'.{self.format.lower()}'
                os.rename(os.path.join(out_dir, f), os.path.join(out_dir, fname))

        # Save camera mtx and other metadata
        with open(os.path.join(out_dir, 'blender_transforms.json'), 'w') as out_file:
                json.dump(out_data, out_file, indent=4)

        obj = bpy.data.objects[obj_name]
        obj.select_set(True)
        bpy.ops.export_scene.obj(filepath=os.path.join(out_dir, 'mesh.obj'), use_selection=True, use_animation=False, use_edges=True, use_normals=True, use_uvs=True, use_materials=True, use_triangles=True, path_mode='ABSOLUTE')

        bpy.data.objects.remove(bpy.data.objects[obj_name], do_unlink=True)
        bpy.data.objects.remove(bpy.data.objects['Empty'], do_unlink=True)
        
def split(a, n):
    k, m = divmod(len(a), n)
    return [ a[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(n)]

def main(args):    
    in_dir = args.root_dir
    out_dir = args.output_dir
    os.makedirs(args.output_dir, exist_ok=True)
 
    rp = THuman() 
    rp.setup_scene(args.device_id)
     
    scan_list = sorted(os.listdir(args.root_dir))
    split_list = split(scan_list , args.tot)[args.id]    
    
    with open(args.smpl_json_path) as f:
        thuman_smpl_dict = json.load(f)

    for i, scan_dir in enumerate(split_list):     
        sub_dir = os.path.join(out_dir, scan_dir)
        scan_info = thuman_smpl_dict[scan_dir]
        # if not os.path.exists(sub_dir):
        os.makedirs(sub_dir, exist_ok=True)
        rp.render_obj(
            os.path.join(in_dir, scan_dir), 
            sub_dir, 
            y_median = scan_info['y_median'])

if __name__ == "__main__":
    argv = extract_args()
    args = parser.parse_args(argv)
    main(args)
