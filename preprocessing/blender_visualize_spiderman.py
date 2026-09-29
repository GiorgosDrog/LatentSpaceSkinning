import bpy
import numpy as np
import os

folder_path = r"E:\didaktoriko\diffusion_solution\results"
npy_path = os.path.join(folder_path, "walk_turn_walk_jump.npy")
rest_pose_path = os.path.join(folder_path, "rest_pose.npy")
faces_path = os.path.join(folder_path, "faces.npy")

if not os.path.exists(npy_path) or not os.path.exists(rest_pose_path) or not os.path.exists(faces_path):
    print("Error: Could not find files in the Downloads folder.")
else:
    print("Loading data and building the mesh from rest_pose.npy + faces.npy...")
    data = np.load(npy_path).astype(np.float32)
    rest_pose = np.load(rest_pose_path).astype(np.float32)
    faces = np.load(faces_path)
    num_frames = data.shape[0]

    assert rest_pose.shape[0] == data.shape[1], (
        f"Vertex-count mismatch: rest_pose.npy has {rest_pose.shape[0]} vertices, "
        f"animation data has {data.shape[1]} -- they must be the SAME mesh export "
        f"(same vertex order), otherwise the nearest-point offset tracking "
        f"below is meaningless."
    )
    assert faces.max() < rest_pose.shape[0], (
        f"faces.npy references vertex index {faces.max()}, but rest_pose.npy only "
        f"has {rest_pose.shape[0]} vertices -- not the same mesh."
    )

    mesh_skin = bpy.data.meshes.new(name="Spiderman_Mesh")
    mesh_skin.from_pydata(rest_pose.tolist(), [], faces.tolist())
    mesh_skin.update()
    skin_obj = bpy.data.objects.new(name="Spiderman_Final", object_data=mesh_skin)
    bpy.context.collection.objects.link(skin_obj)

    mesh_rest = bpy.data.meshes.new(name="Driver_Rest_Mesh")
    obj_rest = bpy.data.objects.new(name="Driver_Rest", object_data=mesh_rest)
    bpy.context.collection.objects.link(obj_rest)
    mesh_rest.from_pydata(rest_pose.tolist(), [], [])
    obj_rest.hide_set(True)
    obj_rest.hide_render = True

    mesh_anim = bpy.data.meshes.new(name="Driver_Anim_Mesh")
    obj_anim = bpy.data.objects.new(name="Driver_Anim", object_data=mesh_anim)
    bpy.context.collection.objects.link(obj_anim)
    mesh_anim.from_pydata(rest_pose.tolist(), [], [])

    obj_anim.shape_key_add(name="Basis")
    for f in range(num_frames):
        sk = obj_anim.shape_key_add(name=f"Frame_{f}")
        sk.data.foreach_set('co', data[f].ravel())
        sk.value = 0.0
        sk.keyframe_insert(data_path="value", frame=f-1)
        sk.value = 1.0
        sk.keyframe_insert(data_path="value", frame=f)
        sk.value = 0.0
        sk.keyframe_insert(data_path="value", frame=f+1)

    obj_anim.hide_set(True)
    obj_anim.hide_render = True

    root_empty = bpy.data.objects.new("Spiderman_Controller", None)
    bpy.context.collection.objects.link(root_empty)
    root_empty.empty_display_size = 2.0
    root_empty.empty_display_type = 'ARROWS'

    skin_obj.parent = root_empty
    obj_rest.parent = root_empty
    obj_anim.parent = root_empty

    mod = skin_obj.modifiers.new(name="Magnetic_Bind", type='NODES')
    tree = bpy.data.node_groups.new(name="Bind_Tree", type='GeometryNodeTree')
    mod.node_group = tree

    in_node = tree.nodes.new('NodeGroupInput')
    tree.interface.new_socket(name="Geometry", in_out='INPUT', socket_type='NodeSocketGeometry')
    out_node = tree.nodes.new('NodeGroupOutput')
    tree.interface.new_socket(name="Geometry", in_out='OUTPUT', socket_type='NodeSocketGeometry')

    info_rest = tree.nodes.new('GeometryNodeObjectInfo')
    info_rest.inputs['Object'].default_value = obj_rest
    info_rest.transform_space = 'RELATIVE'

    info_anim = tree.nodes.new('GeometryNodeObjectInfo')
    info_anim.inputs['Object'].default_value = obj_anim
    info_anim.transform_space = 'RELATIVE'

    sample_near = tree.nodes.new('GeometryNodeSampleNearest')
    pos_input = tree.nodes.new('GeometryNodeInputPosition')

    sample_idx_anim = tree.nodes.new('GeometryNodeSampleIndex')
    sample_idx_anim.data_type = 'FLOAT_VECTOR'

    sample_idx_rest = tree.nodes.new('GeometryNodeSampleIndex')
    sample_idx_rest.data_type = 'FLOAT_VECTOR'

    sub = tree.nodes.new('ShaderNodeVectorMath')
    sub.operation = 'SUBTRACT'

    set_pos = tree.nodes.new('GeometryNodeSetPosition')

    links = tree.links
    links.new(info_rest.outputs['Geometry'], sample_near.inputs['Geometry'])

    links.new(info_anim.outputs['Geometry'], sample_idx_anim.inputs['Geometry'])
    links.new(pos_input.outputs['Position'], sample_idx_anim.inputs['Value'])
    links.new(sample_near.outputs['Index'], sample_idx_anim.inputs['Index'])

    links.new(info_rest.outputs['Geometry'], sample_idx_rest.inputs['Geometry'])
    links.new(pos_input.outputs['Position'], sample_idx_rest.inputs['Value'])
    links.new(sample_near.outputs['Index'], sample_idx_rest.inputs['Index'])

    links.new(sample_idx_anim.outputs['Value'], sub.inputs[0])
    links.new(sample_idx_rest.outputs['Value'], sub.inputs[1])

    links.new(in_node.outputs['Geometry'], set_pos.inputs['Geometry'])
    links.new(sub.outputs['Vector'], set_pos.inputs['Offset'])
    links.new(set_pos.outputs['Geometry'], out_node.inputs['Geometry'])


    mat = bpy.data.materials.get("Material.003")
    if mat is None:
        mat = bpy.data.materials.new(name="Spiderman_Blue")
        mat.use_nodes = True

    nodes = mat.node_tree.nodes
    bsdf = nodes.get("Principled BSDF")
    if bsdf is None:
        bsdf = nodes.new(type="ShaderNodeBsdfPrincipled")
        mat.node_tree.links.new(bsdf.outputs["BSDF"], nodes["Material Output"].inputs["Surface"])

    bsdf.inputs["Base Color"].default_value = (0.01, 0.06, 0.3, 1.0)

    skin_obj.data.materials.clear()
    skin_obj.data.materials.append(mat)

    set_mat = tree.nodes.new('GeometryNodeSetMaterial')
    set_mat.inputs['Material'].default_value = mat
    set_mat.location = (set_pos.location.x + 200, set_pos.location.y)

    links.new(set_pos.outputs['Geometry'], set_mat.inputs['Geometry'])
    links.new(set_mat.outputs['Geometry'], out_node.inputs['Geometry'])

    bpy.context.view_layer.objects.active = skin_obj
    bpy.ops.object.shade_smooth()

    bpy.context.scene.frame_start = 0
    bpy.context.scene.frame_end = num_frames - 1

    print("SUCCESS! Use the 'Spiderman_Controller' Empty to move and scale your character!")
