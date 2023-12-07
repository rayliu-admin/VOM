python inference.py \
    --variant mobilenetv3 \
    --checkpoint "/home/liurui/Model/rvm_mobilenetv3.pth" \
    --device cuda \
    --input-source "/home/liurui/DATA/realhuman_reformat/3048929/frames" \
    --output-type png_sequence \
    --output-alpha "/home/liurui/DATA/rvm_res" \