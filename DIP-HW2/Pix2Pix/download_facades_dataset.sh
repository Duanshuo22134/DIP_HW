FILE=facades
URL=http://efrosgans.eecs.berkeley.edu/pix2pix/datasets/$FILE.tar.gz
TAR_FILE=./datasets/$FILE.tar.gz
TARGET_DIR=./datasets/$FILE/
mkdir -p $TARGET_DIR
echo "Downloading $URL dataset to $TARGET_DIR"

if command -v curl >/dev/null 2>&1; then
    curl -L "$URL" -o "$TAR_FILE"
elif command -v wget >/dev/null 2>&1; then
    wget -N "$URL" -O "$TAR_FILE"
else
    echo "Error: neither curl nor wget is installed. Please install one of them and retry."
    exit 1
fi

if [ ! -f "$TAR_FILE" ]; then
    echo "Error: dataset archive was not downloaded successfully."
    exit 1
fi

mkdir -p $TARGET_DIR
tar -zxvf $TAR_FILE -C ./datasets/
rm $TAR_FILE

find "${TARGET_DIR}train" -type f -name "*.jpg" |sort -V > ./train_list.txt
find "${TARGET_DIR}val" -type f -name "*.jpg" |sort -V > ./val_list.txt
