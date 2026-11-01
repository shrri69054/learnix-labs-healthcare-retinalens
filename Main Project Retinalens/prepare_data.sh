#!/usr/bin/env bash
# Unpacks the dataset zip into data/ :   bash prepare_data.sh path/to/archive.zip
set -e
ZIP="${1:-archive.zip}"
mkdir -p data
unzip -q "$ZIP" -d data/_tmp
# zip layout: release-crop/release-crop/{train,validation,test}/{NRG,RG}
for v in crop pad raw; do
  mv "data/_tmp/release-$v/release-$v" "data/release-$v"
done
rm -rf data/_tmp
echo "Done. Using data/release-crop by default (edit DATA_DIR in config.py to change)."
