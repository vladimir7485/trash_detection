1. Installation
    See installation section from YOLOv9 github page (https://github.com/WongKinYiu/yolov9).
    Then install auxiliary packages from requirements.txt:
        pip install -r requirements.txt 

2. Download dataset
    According to downloading section from TACTO github page run the following commands to download TACO dataset:
        cd thirdparty/TACO
        python download.py
    These will put TACO dataset into "data" folder.

3. Convert dataset to darknet format
    Launch prepare_train_val_test.py script to convert downloaded TACO dataset into darknet format:
        python prepare_train_val_test.py \
            --path-to-annotations "<path-to-downloaded-TACO-dataset-root>/TACO/data/annotations" \
            --output-dir "<path-to-downloaded-TACO-dataset-root>/TACO/data"
    Then update "path" in data/taco.yaml file:
        path: <path-to-downloaded-TACO-dataset-root>/TACO/data

4. Launch training
    If your installation follows respective section of YOLOv9 github page, then launch docker container by calling launch_docker shell script:
        ./launch_docker
    Note: set your own paths inside!
    Download pretrained models from YOLOv9 github page.
    Then launch training script:
        sh train_dual.sh
    Note: edit params inside!

5. Training artefacts:
    https://drive.google.com/drive/folders/1RmrcPJzu4-WQqTjBTgcbi52Rw3KBeMGr?usp=drive_link
