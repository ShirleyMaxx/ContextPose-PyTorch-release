# ONLY need to modify these two params to your own.
RAW_DATA_PATH="SOMEWHERE_ELSE/h36m_raw_data"  # modify the path to where you store the raw data
NUM_PROCESS=50  # modify this to the number of processes you want to use

# NO need to modify below!
ROOT_PATH="data/human36m"  # no need to modify, this is the path where you want to store the processed data under ${TransPose_ROOT}.
if [ -d $ROOT_PATH ]
then
    echo "Directory already exists"
else
    mkdir -p $ROOT_PATH
fi

PROCESS_PATH=$ROOT_PATH"/processed"
EXTRACT_PATH=$ROOT_PATH"/extracted"

python mvn/datasets/human36m_preprocessing/process_dataset/process_h36m.py --raw_path $RAW_DATA_PATH --extract_path $EXTRACT_PATH --process_path $PROCESS_PATH

python mvn/datasets/human36m_preprocessing/collect-bboxes.py $ROOT_PATH $NUM_PROCESS
echo "This will create $ROOT_PATH/data/human36m/extra/bboxes-Human36M-GT.npy"

# mkdir -p $ROOT_PATH/extra/una-dinosauria-data
echo "warning: you should first download h36m.zip from https://drive.google.com/file/d/1PIqzOfdIYUVJudV5f22JpbAqDCTNMU8E/view and put it under $ROOT_PATH/extra/una-dinosauria-data"
unzip -d $ROOT_PATH/extra/una-dinosauria-data/ $ROOT_PATH/extra/una-dinosauria-data/h36m.zip 

python mvn/datasets/human36m_preprocessing/generate-labels-npy-multiview.py $ROOT_PATH $ROOT_PATH/extra/una-dinosauria-data/h36m $ROOT_PATH/extra/bboxes-Human36M-GT.npy
echo "You should see only one warning saying camera 54138969 isn\'t present in S11/Directions-2. That\'s fine."

python mvn/datasets/human36m_preprocessing/undistort-h36m.py $ROOT_PATH $ROOT_PATH/extra/human36m-multiview-labels-GTbboxes.npy $NUM_PROCESS

python mvn/datasets/human36m_preprocessing/limb_length.py $ROOT_PATH