
# Certified Defense for Object Detection

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
* https://github.com/eriklindernoren/PyTorch-YOLOv3 <br>
* https://github.com/microsoft/denoised-smoothing
