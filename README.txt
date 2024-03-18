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
        ln -s ${HOST_DIR}/yolo9/runs ${HOST_DIR}/code/trash_detection/thirdparty/yolov9/runs

    c. Download pretrained models from YOLOv9 github page.

    d. Then launch training script:
        sh train_dual.sh
    Note: edit params inside!

5. Training artefacts:
    https://drive.google.com/drive/folders/1RmrcPJzu4-WQqTjBTgcbi52Rw3KBeMGr?usp=drive_link
