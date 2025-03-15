ade_val_root=data/ADEChallengeData2016/images
corrupt_out_root=data/ADE20K_val-c/
# c_type/severity/

all_corruptions=(gaussian_noise shot_noise impulse_noise defocus_blur glass_blur motion_blur zoom_blur snow frost fog brightness contrast elastic_transform pixelate jpeg_compression)

for corruption in ${all_corruptions[@]}
do
  python core/dataset/make_ade_c.py \
    -c ${corruption} \
    -s 5 \
    --ade20k-val-img-dir ${ade_val_root} \
    --corruptions-out-dir ${corrupt_out_root}
done

