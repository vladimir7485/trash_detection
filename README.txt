1. Installation
    a. See installation section from YOLOv9 github page (https://github.com/WongKinYiu/yolov9).
    b. Then install auxiliary packages from requirements.txt:
        pip install -r requirements.txt 

2. Download dataset
    a. According to downloading section from TACTO github page run the following commands to download TACO dataset:
        cd thirdparty/TACO
        python download.py
    b. These will put TACO dataset into "data" folder.

3. Convert dataset to darknet format
    a. Launch prepare_train_val_test.py script to convert downloaded TACO dataset into darknet format:
        python prepare_train_val_test.py \
            --path-to-annotations "<path-to-downloaded-TACO-dataset-root>/TACO/data/annotations" \
            --output-dir "<path-to-downloaded-TACO-dataset-root>/TACO/data"
    b. Then update "path" in data/taco.yaml file:
        path: <path-to-downloaded-TACO-dataset-root>/TACO/data

4. Launch training
    a. If your installation follows respective section of YOLOv9 github page, then launch docker container by calling launch_docker shell script:
        ./launch_docker
    Note: set your own paths inside!

    b. Create symbolic link to runs directory:
        cd ${HOME_DIR}/code/trash_detection
        ln -s ${HOME_DIR}/yolo9/runs ${HOME_DIR}/code/trash_detection/runs
        ln -s ${HOME_DIR}/yolo9/runs ${HOME_DIR}/code/trash_detection/thirdparty/yolov9/runs

    c. Download pretrained models from YOLOv9 github page.

    d. Launch training script:
        sh train_dual.sh
    Note: edit params inside!

5. Training artefacts (download them and put into ${HOME_DIR}/code/trash_detection/runs directory to use later):
    https://drive.google.com/drive/folders/1RmrcPJzu4-WQqTjBTgcbi52Rw3KBeMGr?usp=drive_link

6. To evaluate artefact run the following script (modify paths inside!):
    sh val.sh
    Note: if you use Pillow >= 10, then modify thirdparty/yolov9/utils/plots.py file as follows:
        86  w, h = self.font.getsize(label)  # text width, height -->
        86  _, _, w, h = self.font.getbbox(label)  # text width, height <--
