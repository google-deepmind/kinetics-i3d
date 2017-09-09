#!/bin/bash

# Script to download the HMDB-51 data set.
#
# usage:
#  bash download_hmdb.sh [data dir]
set -e

if [ -z "$1" ]; then
  echo "usage download_and_preproces_hmdb.sh [data dir]"
  exit
fi

# Useful commands
UNRAR="unrar e"

# Create the output directories.
OUTPUT_DIR="${1%/}"
CURRENT_DIR=$(pwd)
if [ ! -f ${FILENAME} ]; then
    mkdir -p "${OUTPUT_DIR}"
  else
    cd ${OUTPUT_DIR}
  fi
REMOTE_URL="http://serre-lab.clps.brown.edu/wp-content/uploads/2013/10/hmdb51_org.rar"
HMDB_FILE="hmdb51_org.rar"


# Helper function to download and unpack a .unrar file.
function download_and_unrar() {
  local DOWNLOAD_URL=${1}
  local OUTPUT_DIR=${2}
  local FILENAME=${3}

  local WORKING_DIR=$(pwd)
  cd ${OUTPUT_DIR}

  if [ ! -f ${FILENAME} ]; then
    echo "Downloading ${FILENAME} to $(pwd)"
    wget -nd -c "${DOWNLOAD_URL}"
  else
    echo "Skipping download of ${FILENAME}"
  fi
  echo "Unrar ${FILENAME}"
  ${UNRAR} ${FILENAME}
  cd ${WORKING_DIR}
}


function extract_videos(){
  local OUTPUT_DIR=${1}

  local WORKING_DIR=$(pwd)
  cd ${OUTPUT_DIR}

  for FOLD in *.rar
  do
    local CLASS_FOLD="${FOLD%.*}"
    mkdir ${CLASS_FOLD}
    mv ${FOLD} ${CLASS_FOLD}
    cd ${CLASS_FOLD}
    ${UNRAR} ${FOLD}
    rm ${FOLD}
    cd ..    
  done
  cd ${WORKING_DIR}
}


# Download the videos
download_and_unrar ${REMOTE_URL} ${OUTPUT_DIR} ${HMDB_FILE}

# Extract the videos
extract_videos ${OUTPUT_DIR}
