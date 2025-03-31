import os
import shutil
import sys

import bpy
import mathutils

# Get the arguments after the "--" delimiter
argv = sys.argv[sys.argv.index("--") + 1 :]
output_dir = argv[0]
input_file = argv[1]

# Clear existing objects
bpy.ops.object.select_all(action="SELECT")
bpy.ops.object.delete()

# Import GLTF file
bpy.ops.import_scene.gltf(filepath=input_file)

# Scale all objects (1/18)
scale_factor = 1 / 18
for obj in bpy.data.objects:
    if obj.type == "MESH":
        bpy.context.view_layer.objects.active = obj
        obj.select_set(True)
        obj.scale = obj.scale * scale_factor
        bpy.ops.object.transform_apply(
            location=False, rotation=False, scale=True
        )

# Get the base filename without extension
base_name = os.path.splitext(os.path.basename(input_file))[0]
output_path = os.path.join(output_dir, base_name + ".obj")

# Process and save textures first
for material in bpy.data.materials:
    if material.use_nodes:
        for node in material.node_tree.nodes:
            if node.type == "TEX_IMAGE" and node.image:
                # Ensure the image has a proper name and path
                if not node.image.filepath:
                    image_name = f"{node.image.name}.png"
                    node.image.filepath = os.path.join(output_dir, image_name)
                if not node.image.packed_file:
                    node.image.pack()

# Export as OBJ
bpy.ops.wm.obj_export(
    filepath=output_path,
    export_selected_objects=True,
    export_uv=True,
    export_normals=True,
    export_materials=True,
    export_triangulated_mesh=True,
    export_curves_as_nurbs=False,
    forward_axis="NEGATIVE_Z",
    up_axis="Y",
    export_colors=True,
    export_material_groups=True,
    export_object_groups=True,
    path_mode="RELATIVE",
)

# Save textures
for image in bpy.data.images:
    if image.packed_file:
        texture_path = os.path.join(output_dir, f"{image.name}.png")
        image.filepath_raw = texture_path
        image.save_render(texture_path)
        image.unpack(method="WRITE_LOCAL")

# Fix MTL file
mtl_path = os.path.splitext(output_path)[0] + ".mtl"
if os.path.exists(mtl_path):
    with open(mtl_path, "r") as file:
        mtl_content = file.read()

    # Find material names and their corresponding textures
    for material in bpy.data.materials:
        if material.use_nodes:
            for node in material.node_tree.nodes:
                if node.type == "TEX_IMAGE" and node.image:
                    texture_path = f"{node.image.name}.png"
                    # Add or update texture reference in MTL
                    if (
                        f"newmtl {material.name}" in mtl_content
                        and "map_Kd" not in mtl_content
                    ):
                        mtl_content = mtl_content.replace(
                            f"newmtl {material.name}",
                            f"newmtl {material.name}\nmap_Kd {texture_path}",
                        )

    # Write updated MTL content
    with open(mtl_path, "w") as file:
        file.write(mtl_content)
