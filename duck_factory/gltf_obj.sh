#!/bin/bash
if [ "$#" = 2 ]
then
    blender -b -P duck_factory/gltf_obj.py -- "$2" "$1"
else
    echo From glTF 2.0 converter to OBJ.
    echo Supported file formats: .gltf, .glb
    echo
    echo "gltf_obj.sh [filename] [output_directory]"
fi
