import bpy
import numpy as np

root_dir = r"E:\didaktoriko\diffusion_solution\dataset"
character = "x_bot"
POINT_RADIUS = 0.3

rest_pose_path = rf"{root_dir}\{character}\static_data\rest_pose.npy"
verts = np.load(rest_pose_path).astype(float)
print(f"Loaded {verts.shape[0]} vertices from {rest_pose_path}")
print(f"bbox: {verts.max(0) - verts.min(0)}")

OBJ_NAME = f"RestPoseCloud_{character}"
if OBJ_NAME in bpy.data.objects:
    bpy.data.objects.remove(bpy.data.objects[OBJ_NAME], do_unlink=True)
if f"{OBJ_NAME}_Mesh" in bpy.data.meshes:
    bpy.data.meshes.remove(bpy.data.meshes[f"{OBJ_NAME}_Mesh"])
if f"{OBJ_NAME}_Nodes" in bpy.data.node_groups:
    bpy.data.node_groups.remove(bpy.data.node_groups[f"{OBJ_NAME}_Nodes"])

mesh = bpy.data.meshes.new(f"{OBJ_NAME}_Mesh")
mesh.from_pydata(verts.tolist(), [], [])
mesh.update()

obj = bpy.data.objects.new(OBJ_NAME, mesh)
bpy.context.collection.objects.link(obj)

mod = obj.modifiers.new("PointCloud", 'NODES')
node_group = bpy.data.node_groups.new(f"{OBJ_NAME}_Nodes", 'GeometryNodeTree')
mod.node_group = node_group

try:
    node_group.interface.new_socket(name="Geometry", in_out='INPUT', socket_type='NodeSocketGeometry')
    node_group.interface.new_socket(name="Geometry", in_out='OUTPUT', socket_type='NodeSocketGeometry')
except AttributeError:
    node_group.inputs.new('NodeSocketGeometry', 'Geometry')
    node_group.outputs.new('NodeSocketGeometry', 'Geometry')

nodes = node_group.nodes
links = node_group.links
nodes.clear()

input_node = nodes.new('NodeGroupInput')
output_node = nodes.new('NodeGroupOutput')
mesh_to_points = nodes.new('GeometryNodeMeshToPoints')
mesh_to_points.inputs['Radius'].default_value = POINT_RADIUS

input_node.location = (-400, 0)
mesh_to_points.location = (-100, 0)
output_node.location = (200, 0)

links.new(input_node.outputs['Geometry'], mesh_to_points.inputs['Mesh'])
links.new(mesh_to_points.outputs['Points'], output_node.inputs['Geometry'])

mat_name = f"{OBJ_NAME}_Mat"
if mat_name in bpy.data.materials:
    bpy.data.materials.remove(bpy.data.materials[mat_name])
mat = bpy.data.materials.new(mat_name)
mat.use_nodes = True
bsdf = mat.node_tree.nodes.get("Principled BSDF")
if bsdf is not None:
    bsdf.inputs["Base Color"].default_value = (1.0, 0.5, 0.05, 1.0)
obj.data.materials.append(mat)

print(f"Done -- '{OBJ_NAME}' added to the scene as a point cloud "
      f"({verts.shape[0]} points, radius={POINT_RADIUS}).")
print("Compare this against the predicted/blended motion's point-cloud render: "
      "if THIS also looks broken/scattered, the issue is in this character's "
      "static data or the point-cloud setup, not the model's prediction.")
