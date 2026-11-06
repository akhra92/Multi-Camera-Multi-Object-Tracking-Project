#!/usr/bin/env bash
# Download WILDTRACK (EPFL, 6.8 GB zip, 7.2 GB unpacked) and unpack it to <target_dir>/Wildtrack.
# Usage: bash scripts/download_wildtrack.sh [target_dir]   (default: data)
# To keep the data outside the repo: bash scripts/download_wildtrack.sh ~/datasets/wildtrack
#                                    ln -s ~/datasets/wildtrack data
# Needs 7-Zip (sudo apt install 7zip): plain `unzip` cannot read this >4 GB archive.
set -euo pipefail
DIR="${1:-data}"
URL=https://documents.epfl.ch/groups/c/cv/cvlab-unit/www/data/Wildtrack/Wildtrack_dataset_full.zip
SIZE=6807496358
mkdir -p "$DIR" && cd "$DIR"
# the server drops long transfers; -C - resumes where the last attempt stopped
until [ "$(stat -c %s Wildtrack_dataset_full.zip 2>/dev/null || echo 0)" = "$SIZE" ]; do
  curl -L -C - -o Wildtrack_dataset_full.zip "$URL" || { echo "connection dropped, resuming"; sleep 5; }
done
7z x -y Wildtrack_dataset_full.zip -x'r!__MACOSX' -x'r!.DS_Store'
mv Wildtrack_dataset Wildtrack
echo "Unpacked to $DIR/Wildtrack. You can delete the zip: rm $DIR/Wildtrack_dataset_full.zip"
