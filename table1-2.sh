SORTING_LIST=("object" "center")
BINNING_LIST=("single" "label" "location" "location+label")
DENOISE_LIST=("" "--denoise ")
CONF_LIST=("0.1" "0.2" "0.4" "0.6" "0.8")

for i in {0..79}
do
SORTING=${SORTING_LIST[$((i%2))]}
BINNING=${BINNING_LIST[$((i/2%4))]}
DENOISE=${DENOISE_LIST[$((i/8%2))]}
CONF=${CONF_LIST[$((i/16%5))]}

python test_smooth.py --bin $BINNING --sort $SORTING $DENOISE--smooth --conf_thres $CONF
done