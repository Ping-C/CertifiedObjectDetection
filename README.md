
# Certified Defense for Object Detection
The repository contains the code and models necessary to replicate the results of our recent paper:
**Detection as Regression: Certified Object Detection by Median Smoothing** <br>
*Ping-yeh Chiang, Michael J. Curry, Ahmed Abdelkader, Aounon Kumar, John Dickerson, Tom Goldstein* <br>
NIPS 2020

In this paper, we propose median smoothing and detection-to-regression techniques that allow us to 
provable defend object detectors against l2 adversaries in a blackbox manner.
Below are some examples of certificates generated for an l2 norm adversary. 
The dotted line represents the furthest the bounding box could move upon attack. 
If there is a cross at the corner of the bounding box, that means that the bounding box
could disappear or change label when attacked.

<p align="center">
<img src="images/img_detection_sample.png"  width="500" >
</p>


## Results
The AP is calculated by evaluating the precision-recall at 5 different
objectness thresholds (0.1, 0.2, 0.4, 0.6, 0.8). 

Performance of Smoothed YOLOv3 under various settings

| Sorting    | Binning        | Denoise? | Clean AP @ 50 | Certified AP @ 50 |
|------------|----------------|----------|---------------:|-------------------:|
| Objectness | None           | No       |         5.19% |             0.21% |
| Location   | None           | No       |         5.85% |             0.14% |
| Objectness | Label          | No       |         7.34% |             0.30% |
| Location   | Label          | No       |         8.58% |             0.32% |
| Objectness | Location       | No       |         7.89% |             0.42% |
| Location   | Location       | No       |         8.03% |             0.32% |
| Objectness | Location+Label | No       |         9.09% |             0.45% |
| Location   | Location+Label | No       |         9.49% |             0.44% |
| Objectness | None           | Yes      |        17.60% |             1.24% |
| Location   | None           | Yes      |        21.51% |             1.24% |
| Objectness | Label          | Yes      |        25.27% |             2.67% |
| Location   | Label          | Yes      |        29.75% |             3.32% |
| Objectness | Location       | Yes      |        27.48% |             3.23% |
| Location   | Location       | Yes      |        28.90% |             2.67% |
| Objectness | Location+Label | Yes      |        30.32% |             3.97% |
| Location   | Location+Label | Yes      |        32.04% |             4.18% |

Performance Comparison between Architectures

|             | Base Detector | Smoothed Detector |                   |
|-------------|---------------|-------------------|-------------------|
| Base Model  | AP @ 50       | AP @ 50           | Certified AP @ 50 |
| Yolo        |        48.66% |            31.93% |             4.21% |
| Mask RCNN   |        51.28% |            30.53% |             1.67% |
| Faster RCNN |        50.47% |            29.89% |             1.54% |

## Getting Started
1. Clone and install requirements 
    ```
    sudo pip3 install -r requirements.txt
   ```

2. Download COCO
    ```
    cd data/
    bash get_coco_dataset.sh
    ```
    
3. Download pretrained yolo weights
    ```
   cd weights/
    bash download_weights.sh
   ```
    
4. Download trained denoisers from [here](https://drive.google.com/open?id=1MCXSKlz8qYQOGqMhbqmwN4Y0YPNMxVj6). Then move the downloaded pretrained_models.tar.gz into the root directory of this repository. Run tar -xzvf pretrained_models.tar.gz to extract the models.
    
5. Reproduce results in the paper by running the two shell scripts below
    ```
   bash table1-2.sh
   bash table3.sh
   ``` 
   
## References
**Denoised Smoothing: A Provable Defense for Pretrained Classifiers** <br>
*Hadi Salman, Mingjie Sun, Greg Yang, Ashish Kapoor, J. Zico Kolter* <br>

**YOLOv3: An Incremental Improvement** <br>
*Joseph Redmon, Ali Farhadi* <br>

## Credit
Our code is based on the open source codes of the following repositories

* https://github.com/locuslab/smoothing [Cohen et al (2019)]
* https://github.com/Hadisalman/smoothing-adversarial [Salman et al. (2019)]
* https://github.com/microsoft/denoised-smoothing [Salman et al. (2020)]
* https://github.com/eriklindernoren/PyTorch-YOLOv3 <br>
