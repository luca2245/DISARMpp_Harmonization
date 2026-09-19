#!/usr/bin/env bash
set -euo pipefail

usage() {
    echo "Usage: $0 <input_folder> <intermediate_folder> <final_output_folder> [--reference <0.5mm|0.7mm|0.8mm|1mm|2mm>] [--cost <normcorr|leastsq|corratio|mutualinfo|normmi>]"
}

if [ "$#" -lt 3 ]; then
    usage
    exit 1
fi

input_folder="$1"
intermediate_folder="$2"
final_output_folder="$3"
shift 3

reference="1mm"
cost_function="normcorr"

while [ "$#" -gt 0 ]; do
    case "$1" in
        --reference)
            reference="$2"
            shift 2
            ;;
        --cost)
            cost_function="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            usage
            exit 1
            ;;
    esac
done

if [ -z "${FSLDIR:-}" ]; then
    echo "FSLDIR is not set. Please load or configure FSL before running this script."
    exit 1
fi

case "$reference" in
    0.5mm) ref_image="$FSLDIR/data/standard/MNI152_T1_0.5mm.nii.gz" ;;
    0.7mm) ref_image="$FSLDIR/data/standard/MNI152_T1_0.7mm.nii.gz" ;;
    0.8mm) ref_image="$FSLDIR/data/standard/MNI152_T1_0.8mm.nii.gz" ;;
    1mm)   ref_image="$FSLDIR/data/standard/MNI152_T1_1mm.nii.gz" ;;
    2mm)   ref_image="$FSLDIR/data/standard/MNI152_T1_2mm.nii.gz" ;;
    *)
        echo "Unsupported reference resolution: $reference"
        exit 1
        ;;
esac

case "$cost_function" in
    normcorr|leastsq|corratio|mutualinfo|normmi) ;;
    *)
        echo "Unsupported FLIRT cost function: $cost_function"
        exit 1
        ;;
esac

if [ ! -f "$ref_image" ]; then
    echo "Reference image not found: $ref_image"
    exit 1
fi

mkdir -p "$intermediate_folder" "$final_output_folder"

# FAST settings used in the paper experiments.
common_options=(-t 1 -n 3 -H 0.1 -I 4 -l 20.0 --nopve -B -b)

shopt -s nullglob
images=("$input_folder"/*.nii.gz)
if [ "${#images[@]}" -eq 0 ]; then
    echo "No .nii.gz files found in $input_folder"
    exit 1
fi

for image in "${images[@]}"; do
    filename=$(basename "$image")
    image_basename="${filename%.nii.gz}"

    output_prefix="$intermediate_folder/output_${image_basename}"
    restored_output="${output_prefix}_restore.nii.gz"
    reoriented_image="$intermediate_folder/reoriented_${image_basename}.nii.gz"
    registered_image="$final_output_folder/registered_${image_basename}.nii.gz"

    echo "Processing: $filename"

    fslreorient2std "$image" "$reoriented_image"
    "$FSLDIR/bin/fast" "${common_options[@]}" -o "$output_prefix" "$reoriented_image"

    "$FSLDIR/bin/flirt" \
        -in "$restored_output" \
        -ref "$ref_image" \
        -out "$registered_image" \
        -cost "$cost_function" \
        -searchrx -90 90 \
        -searchry -90 90 \
        -searchrz -90 90 \
        -dof 12 \
        -interp trilinear

done
