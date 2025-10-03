Install rpm2cpio, Json

    sudo apt update
    sudo apt install libjsoncpp-dev
    sudo apt install rpm2cpio -y


Build Instructions:

    mkdir build
    cd build
    cmake ..
    make

Usage:

./run_inference project_mpk_path images_dir_path / rtsp-url / videofile / 0,1,2 Web-cam or USB cameras

For Images inference:

    $ ./run_inference ./../project.mpk ./../../test_images

For IP-Cameras inference:

    $ ./run_inference ./../project.mpk rtsp://192.168.134.90/axis-media/media.amp

For Video file inference:

    $ ./run_inference ./../project.mpk video_file.mp4

For Web camera or USB cameras

    $ ./run_inference ./../project.mpk 0

