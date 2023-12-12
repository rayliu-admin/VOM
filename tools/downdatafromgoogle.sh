#! /bin/bash
  
# cd scratch place
# cd DATA/

# Download zip dataset from Google Drive
filename='ResNet34_En_nomixup.pth'
fileid='1kNj33D7x7tR-5hXOvxO53QeCEC8ih3-A'
mkdir cookie
curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=${fileid}" > /dev/null
curl -Lb ./cookie "https://drive.google.com/uc?export=download&confirm=`awk '/download/ {print $NF}' ./cookie`&id=${fileid}" -o ${filename}
rm -rf ./cookie
  
# Unzip
# unzip -q ${filename}
# rm ${filename}
  
# cd out