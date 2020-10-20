ARCH_LIST=("yolo" "mask_rcnn" "faster_rcnn")
CONF_LIST=("0.1" "0.2" "0.4" "0.6" "0.8")

for i in {0..14}
do
ARCH=${ARCH_LIST[$((i%3))]}
CONF=${CONF_LIST[$((i/3%5))]}

python test_smooth.py --bin location+label --sort center --denoise --smooth --model_type $ARCH --conf_thres $CONF
done