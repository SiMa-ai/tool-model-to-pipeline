#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>
#include <iostream>
#include <simacpp.hpp>
#include <sys/stat.h>
#include <sys/types.h>
#include <jsoncpp/json/json.h>
// using json = nlohmann::json;

typedef struct _DL_RESULT
{
    int classId;
    float confidence;
    cv::Rect box;
    std::vector<cv::Point2f> keyPoints;
} DL_RESULT;

cv::Mat preprocess(const cv::Mat& input_image, int target_width, int target_height) {
    // Define the desired size
    cv::Size target_size(target_width, target_height);
    // Resize the input image while maintaining the aspect ratio
    // Calculate aspect ratios
    double width_ratio = static_cast<double>(target_size.width) / input_image.cols;
    double height_ratio = static_cast<double>(target_size.height) / input_image.rows;

    // Resize the image while preserving aspect ratio
    cv::Mat resized_image;
    if (width_ratio == 1 && height_ratio == 1) {
        // No resizing needed, return the original image
        cv::cvtColor(input_image, input_image, cv::COLOR_BGR2RGB);
        return input_image;
    }
    if (width_ratio < height_ratio) {
        // Resize based on width
        int new_height = static_cast<int>(input_image.rows * width_ratio);
        cv::resize(input_image, resized_image, cv::Size(target_size.width, new_height));
        int top_padding = (target_size.height - new_height) / 2;
        int bottom_padding = target_size.height - new_height - top_padding;
        cv::copyMakeBorder(resized_image, resized_image, top_padding, bottom_padding, 0, 0, cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));
    } else {
        // Resize based on height
        int new_width = static_cast<int>(input_image.cols * height_ratio);
        cv::resize(input_image, resized_image, cv::Size(new_width, target_size.height));
        int left_padding = (target_size.width - new_width) / 2;
        int right_padding = target_size.width - new_width - left_padding;
        cv::copyMakeBorder(resized_image, resized_image, 0, 0, left_padding, right_padding, cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));
    }
    // Transpose image channels from BGR to RGB
    cv::cvtColor(resized_image, resized_image, cv::COLOR_BGR2RGB);

    return resized_image;
}

std::string extractFileName(const std::string& path, const std::string& outputFolder) {
    // Find the position of the last directory separator
    size_t lastSlashPos = path.find_last_of("/\\");

    // Extract the substring after the last directory separator
    if (lastSlashPos != std::string::npos) {
        return outputFolder + "/" + path.substr(lastSlashPos + 1);
    }

    // If no directory separator is found, return the entire path
    return outputFolder + "/" + path;
    // return path;
}


void writeTextToImage(cv::Mat& image, const std::string& imagePath, const std::string& predictedLabel, const std::string& outputFolder) {
    // Load the image
    // cv::Mat image = image; //cv::imread(imagePath);

    // Check if the image is loaded successfully
    if (image.empty()) {
        std::cerr << "Failed to load image: " << imagePath << std::endl;
        return;
    }

    // Font settings
    cv::Point textOrg((image.cols - predictedLabel.size() * 10) / 2, image.rows / 2);
    int fontFace = cv::FONT_HERSHEY_SIMPLEX;
    double fontScale = 1;
    cv::Scalar fontColor(255, 255, 255); // White color
    int thickness = 2;

    // Write the text on the image
    cv::putText(image, predictedLabel, textOrg, fontFace, fontScale, fontColor, thickness);

    // Extract the file name from the input image path
    std::string outputFileName = extractFileName(imagePath, outputFolder);
    // Save the image with the predicted label
    cv::imwrite(outputFileName, image);

    std::cout << "Image with predicted label saved to: " << outputFileName << std::endl;
}

bool createDirectory(const std::string& directory) {
    struct stat st;
    if (stat(directory.c_str(), &st) == 0) {
        // Directory already exists, just return true
        return true;
    }

    // Attempt to create the directory
    if (mkdir(directory.c_str(), 0777) == -1) {
        perror("Error creating directory");
        return false;
    }
    return true;
}

std::tuple<int, int, int, std::vector<std::string>> read_model_details(const std::string& file_path) {
    int width = 0, height = 0, topk=25;
    std::vector<std::string> labels;
    char buffer[200];
    ifstream source_mpk;
    ifstream mpk_file;
    Json::Reader reader;
    Json::Value mpk_json;

    if (access(file_path.c_str(), F_OK) == 0) {
        std::cout << "Loading the model from the file path: " << file_path << std::endl;

        // Create temporary directory
        sprintf(buffer, "rm -rf /tmp/sima_tmp");
        system(buffer);
        sprintf(buffer, "mkdir /tmp/sima_tmp");
        system(buffer);

        // Extract MPK file
        source_mpk.open(file_path, ios::binary);
        if(!source_mpk.is_open()) {
            std::cerr << "Failed to find mpk file" << std::endl;
            return {0, 0, topk, labels};
        }

        sprintf(buffer, "/tmp/sima_tmp/project.mpk");
        ofstream dest_mpk(buffer, ios::binary);
        dest_mpk << source_mpk.rdbuf();
        source_mpk.close();
        dest_mpk.close();

        // Extract contents
        system("cd /tmp/sima_tmp/; unzip /tmp/sima_tmp/project.mpk");
        system("cd /tmp/sima_tmp/*/resources/; rpm2cpio installer.rpm | cpio -idmv");
        system("cp /tmp/sima_tmp/*/resources/data/simaai/applications/*/etc/* /tmp/sima_tmp/");

        // Read JSON file
        mpk_file.open("/tmp/sima_tmp/boxdecoder.json");
        if(!mpk_file.is_open()) {
            std::cerr << "Failed to find boxdecoder.json file" << std::endl;
            return {0, 0, topk, labels};
        }
        reader.parse(mpk_file, mpk_json);
        // Get formatted labels
        // Read labels list
        std::vector<std::string> formatted_labels;
        if (mpk_json.isMember("labels")) {
            for (const auto& label : mpk_json["labels"]) {
                formatted_labels.push_back(label.asString());
            }
            labels = formatted_labels;
        }

        // Get dimensions
        if (mpk_json.isMember("original_width") && mpk_json.isMember("original_height")) {
            height = mpk_json["original_height"].asInt();
            width = mpk_json["original_width"].asInt();
            topk = mpk_json["topk"].asInt();
        }
        // Cleanup
        system("rm -rf /tmp/sima_tmp");
    }

    return {width, height, topk, labels};
}

std::tuple<float, float, float> calculate_padding_and_scale(
    int model_width, int model_height, 
    int frame_width, int frame_height) 
{
    float scale = std::min(static_cast<float>(model_width) / frame_width,
                            static_cast<float>(model_height) / frame_height);
    // Compute padding
    float new_w = frame_width * scale;
    float new_h = frame_height * scale;
    float pad_x = (model_width - new_w) / 2.0f;
    float pad_y = (model_height - new_h) / 2.0f;
    std::cout << "Padding: x=" << pad_x << ", y=" << pad_y << std::endl;
    return {pad_x, pad_y, scale};
}

cv::Mat process_yolo_output(char* data, cv::Mat& frame, const std::vector<std::string>& labels,
    double time_per_iteration_ms, float pad_x=0, float pad_y=0, float scale=1, bool image_directory_mode=false) {
                            
    int* intData = reinterpret_cast<int*>(data);
    int num_boxes = intData[0];
    std::cout << "Number of boxes: " << num_boxes << std::endl;
    int offset = 1;
    for (int i = 0; i < num_boxes; ++i) {
        int x1 = intData[offset];
        int y1 = intData[offset + 1];
        int width = intData[offset + 2];
        int height = intData[offset + 3];
        int x2 = x1 + width;
        int y2 = y1 + height;
        float confidence = *reinterpret_cast<float*>(&intData[offset + 4]);
        int classid = intData[offset + 5];

        int x1_orig = int((x1 - pad_x) / scale);
        int y1_orig = int((y1 - pad_y) / scale);
        int x2_orig = int((x2 - pad_x) / scale);
        int y2_orig = int((y2 - pad_y) / scale);

        std::cout << "Box " << i + 1 << ": "
                  << "x1=" << x1 << ", y1=" << y1
                  << ", x2=" << x2 << ", y2=" << y2
                  << ", confidence=" << confidence
                  << ", classid=" << classid 
                  << ", labels=" << labels[classid] << std::endl;

        cv::rectangle(frame, cv::Point(x1_orig, y1_orig), cv::Point(x2_orig, y2_orig), 
                        cv::Scalar(0, 255, 0), 2);

        if (classid >= 0 && classid < labels.size()) {
            std::string label_text = labels[classid] + " " + cv::format("%.2f", confidence);
            int baseline = 0;
            cv::Size label_size = cv::getTextSize(label_text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseline);
            int top = std::max(y1_orig, label_size.height);
            cv::rectangle(frame, cv::Point(x1_orig, top - label_size.height),
                            cv::Point(x1_orig + label_size.width, top + baseline),
                            cv::Scalar(0, 255, 0), cv::FILLED);
            cv::putText(frame, label_text, cv::Point(x1_orig, top),
                        cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0,0,0), 1);
        }

        offset += 6;
    }
    // Calculate and display FPS
    if (image_directory_mode) {
        std::cout << "Image directory mode: No FPS calculation" << std::endl;
        return frame;
    }

    float fps = 1000.0 / time_per_iteration_ms;
    std::string fps_text = cv::format("FPS: %.2f", fps);
    std::string latency_text = cv::format("Inference latency: %.2f", time_per_iteration_ms);
    cv::putText(frame, latency_text, cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 
                1.0, cv::Scalar(255, 255, 0), 2);
    
    return frame;
}

bool cleanupSimaDevice(shared_ptr<simaai::SimaMlsocApi>& simaDeviceInst, 
                      shared_ptr<simaai::SiMaBundle>& modelPtr,
                      shared_ptr<simaai::SiMaDevice>& SiMaDevicePtr,
                      const string& guid) {
    bool success = true;
    simaai::SiMaErrorCode ret;

    ret = simaDeviceInst->unload(modelPtr);
    if (ret != simaai::success) {
        cout << "unloadModel is failed for loaded modelPtr " << modelPtr << endl;
        success = false;
    } else {
        cout << "unloadModel for modelPtr : " << modelPtr << " successfully" << endl;
    }

    ret = simaDeviceInst->closeDevice(SiMaDevicePtr);
    if (ret != simaai::success) {
        cout << "closeDevice() is failed for GUID " << guid << endl;
        success = false;
    } else {
        cout << "Sima Device with SiMaDevicePtr:" << SiMaDevicePtr << " closed successfully" << endl;
    }

    return success;
}


bool runInferenceSynchronousloadModel_cppsdkpipeline(char *argv[]) { //const std::string& model_path) {

    std::string model_path = argv[1];
    vector <string> guids;
    bool null_flag = false;
    shared_ptr<simaai::SimaMlsocApi> simaDeviceInst = simaai::SimaMlsocApi::getInstance();
    guids = simaDeviceInst->enumerateDeviceGuids();
    simaDeviceInst->setLogVerbosity(simaai::SiMaLogLevel::debug);
    simaai::SiMaErrorCode ret = simaai::failure;
    std::vector<cv::String> image_paths;    

    int model_width = 0, model_height = 0, topk=25;
    float pad_x = 0, pad_y = 0, scale=1;

    cv::VideoCapture cap;

    std::vector<std::string> labels;
    std::tie(model_width, model_height, topk, labels) =  read_model_details(model_path);
    std::cout << "Model dimensions: " << model_width << "x" << model_height << std::endl;
    std::cout << "Labels (" << labels.size() << "): ";
    for (const auto& label : labels) {
        std::cout << label << " ";
    }
    std::cout << std::endl;
    
    std::string outputFolder = "./../output";
    if (createDirectory(outputFolder)) {
        std::cout << "Directory created or already exists: " << outputFolder << std::endl;
    }



    // for(int i = 0; i < guids.size(); i++) {

    if (!guids[0].empty()) {
        shared_ptr<simaai::SiMaDevice>  SiMaDevicePtr = simaDeviceInst->openDevice(guids[0]);
        cout << "SiMaDevicePtr for GUID : " << guids[0] << " is : " << SiMaDevicePtr << endl;

        simaai::SiMaBundle model;
        std::vector<uint32_t> in_shape{ int(model_width*model_height*3/4)}; // 640, 640, 3}; //640*640*3/4};
        std::vector<uint32_t> out_shape1{ (4 + topk*25)/4 };
        simaDeviceInst->setMaxOutstandingRequests(SiMaDevicePtr, 32);
        
        //Setting up Model structure
        int d_qid = 0;
        model.data_qid = d_qid;
        model.numInputTensors = 1;
        model.numOutputTensors = 1;
        model.outputBatchSize = 1;
        model.inputBatchSize = 1;
        model.inputShape.emplace_back(in_shape);
        model.outputShape.emplace_back(out_shape1);
       
        shared_ptr<simaai::SiMaBundle> modelPtr = NULL;

        modelPtr = simaDeviceInst->load(SiMaDevicePtr, model_path, model);
        // modelPtr = simaDeviceInst->loadModel(SiMaDevicePtr, model);
        if (NULL == modelPtr){
            cout << "load model failed" << endl;
            return false;
        } else {
            cout << "loadModel is successful with modelPtr: " << modelPtr << endl;
            cout << "modelPtr inputShape: " << modelPtr->inputShape.size() << endl;
            for (int i = 0; i < modelPtr->inputShape.size(); i++) {
                cout << "modelPtr inputShape[" << i << "]: ";
                for (const auto& dim : modelPtr->inputShape[i]) {
                    cout << dim << " ";
                }
                cout << endl;
            }
            cout << "modelPtr outputShape: " << modelPtr->outputShape.size() << endl;
            for (int i = 0; i < modelPtr->outputShape.size(); i++) {
                cout << "modelPtr outputShape[" << i << "]: ";
                for (const auto& dim : modelPtr->outputShape[i]) {
                    cout << dim << " ";
                }
                cout << endl;
            }
            cout << "modelPtr data_qid: " << modelPtr->data_qid << endl;
            cout << "modelPtr numInputTensors: " << modelPtr->numInputTensors << endl;
            cout << "modelPtr numOutputTensors: " << modelPtr->numOutputTensors << endl;
            cout << "modelPtr inputBatchSize: " << modelPtr->inputBatchSize << endl;
            cout << "modelPtr outputBatchSize: " << modelPtr->outputBatchSize << endl;  
        }

        // Setup cleanup handler
        auto cleanup_sima_device = [&]() {
            return cleanupSimaDevice(simaDeviceInst, modelPtr, SiMaDevicePtr, guids[0]);
        };
        
        simaai::SiMaTensorList inputTensorsList;
        simaai::SiMaTensorList outputTensorsList;
        simaai::SiMaMetaData metaData;
        inputTensorsList.emplace_back(simaai::SiMaTensor<simaai::SiMAType>(in_shape));
        inputTensorsList[0].setAppId(d_qid);
        
        outputTensorsList.emplace_back(simaai::SiMaTensor<simaai::SiMAType>(out_shape1));
        outputTensorsList[0].setAppId(d_qid);

    
        cout << "Total Input tensors: " << inputTensorsList.size() << endl;
        for(int i = 0; i < inputTensorsList.size(); i++) {
            cout << "Input Tensor "<< i << " of size " << inputTensorsList[i].getSizeInBytes() << endl;
            cout << "Input Tensor "<< i << " " << (uint64_t)(inputTensorsList[i].getPtr().get()) << endl;
        }

        cout << "Total Output tensors: " << outputTensorsList.size() << endl;
        for(int i = 0; i < outputTensorsList.size(); i++) {
            cout << "Output Tensor "<< i << " of size " << outputTensorsList[i].getSizeInBytes() << endl;
        cout << "Output Tensor "<< i << " " << (uint64_t)(outputTensorsList[i].getPtr().get()) << endl;
        }

        for(int i = 0; i < modelPtr->inputShape.size(); i++) {
            memset((void*)inputTensorsList[i].getPtr().get(), 0, inputTensorsList[i].getSizeInBytes());
        }
        
        double total_duration_ms = 0.0;
        int count = 0;

        std::string argument = argv[2]; //"rtsp://192.168.132.205/axis-media/media.amp"; // Default to RTSP URL if no images are provided
        // Check if the argument is a number (webcam), RTSP URL, or video file
        // Check if it's a directory containing images
        struct stat s;
        if (stat(argument.c_str(), &s) == 0 && S_ISDIR(s.st_mode)) {
            cout << "Provided is an image directory: " << argument << endl;
            // Patterns to match image files (both JPEG and JPG)
            std::vector<std::string> patterns = {argument + "*.jpg", argument + "*.jpeg"};
            // Iterate over each pattern and collect matching files
            for (const auto& pattern : patterns) {
                std::vector<cv::String> files;
                cv::glob(pattern, files);
                image_paths.insert(image_paths.end(), files.begin(), files.end());
            }
            double total_duration_ms = 0.0;
            int count = 0;
            cout<< "Total Images:"<< image_paths.size() << endl;
            bool image_directory_mode = true;
            // for(int index=1; index<=img_count; index++){
            for (const auto& image_path : image_paths) {
                std::cout << "Processing image: " << image_path << std::endl;
                cv::Mat image = cv::imread(image_path);
                std::cout << "Image size: " << image.cols << "x" << image.rows << std::endl;
                if(image.empty()) {
                    std::cerr << "Error: Could not read image file." << std::endl;
                    return -1;
                }
                cv::Mat preprocessed_image = preprocess(image, model_width, model_height);
                std::cout << "preprocessed_image size: " << preprocessed_image.cols << "x" << preprocessed_image.rows << std::endl;
                memcpy(inputTensorsList[0].getPtr().get(), preprocessed_image.data, inputTensorsList[0].getSizeInBytes());

                cout << "\nstarting the runInference \n";
                auto start = std::chrono::high_resolution_clock::now();
                simaai::SiMaErrorCode ret = simaDeviceInst->runSynchronous(modelPtr,
                    inputTensorsList,
                    metaData,
                    outputTensorsList);
                if (ret != simaai::success) {
                    cout << "runInference Failure\n";
                    break;
                } else {
                    cout << "runInferenceSynchronous is successful" << endl;
                }
                auto end = std::chrono::high_resolution_clock::now();
                auto duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
                double time_per_iteration_ms = duration_ms.count();
                total_duration_ms += time_per_iteration_ms;
                std::cout << "Time taken per inference: " << time_per_iteration_ms << " milliseconds" << std::endl;
                char* data = (char*)outputTensorsList[0].getPtr().get();
                // Process the YOLO output and draw bounding boxes
                std::tie(pad_x, pad_y, scale) =  calculate_padding_and_scale(
                                                        model_width, model_height, 
                                                        image.cols, image.rows);
                
                cv::Mat frame = process_yolo_output(data, image, labels, time_per_iteration_ms,
                                                    pad_x, pad_y, scale, image_directory_mode);
                std::string predicted_label = "";

                writeTextToImage(frame, image_path, predicted_label, outputFolder);
            }
            cleanupSimaDevice(simaDeviceInst, modelPtr, SiMaDevicePtr, guids[0]);
            return true;
        }

        if (argument == "0" || argument == "1" || argument == "2") {
            // Webcam index
            cap.open(std::stoi(argument));
        } else if (argument.find("rtsp://") == 0) {
            // video source is a video file
            cout << "Video source is rtsp url:" << argument << endl;
            cap.open(argument);
        } else {
            // Check if file exists and is a video file
            std::ifstream file(argument.c_str());
            if (!file.good()) {
                std::cerr << "Error: File does not exist: " << argument << std::endl;
                cleanupSimaDevice(simaDeviceInst, modelPtr, SiMaDevicePtr, guids[0]);
                return false;
            }
            file.close();
            // Try to open as video file
            cout << "Attempting to open video file: " << argument << endl;
            if (!cap.open(argument)) {
                std::cerr << "Error: Could not open as video file: " << argument << std::endl;
                cleanupSimaDevice(simaDeviceInst, modelPtr, SiMaDevicePtr, guids[0]);
                return false;
            }
            cout << "Successfully opened video file" << endl;
        }
        
        
        if (!cap.isOpened()) {
            std::cerr << "Error: Could not open video source: " << argument << std::endl;
            return false;
        }
        // Get frame dimensions from the video capture
        int frame_width = cap.get(cv::CAP_PROP_FRAME_WIDTH);
        int frame_height = cap.get(cv::CAP_PROP_FRAME_HEIGHT);
        std::cout << "Video dimensions: " << frame_width << "x" << frame_height << std::endl;
        
        if (model_width != frame_width || model_height != frame_height) {
            std::tie(pad_x, pad_y, scale) =  calculate_padding_and_scale(
                                            model_width, model_height, 
                                            frame_width, frame_height);
        }

        cv::Mat frame;
        while (cap.read(frame)) {
            try {
                if (frame.empty()) {
                    std::cerr << "Error: Empty frame captured." << std::endl;
                    break;
                }
                cv::Mat preprocessed_image = preprocess(frame, model_width, model_height);

                memcpy(inputTensorsList[0].getPtr().get(), preprocessed_image.data, inputTensorsList[0].getSizeInBytes());

                cout << "\nstarting the runInference \n";
                auto start = std::chrono::high_resolution_clock::now();
                simaai::SiMaErrorCode ret = simaDeviceInst->runSynchronous(modelPtr,
                    inputTensorsList,
                    metaData,
                    outputTensorsList);
                if (ret != simaai::success) {
                    cout << "runInference Failure\n";
                    break;
                } else {
                    cout << "runInferenceSynchronous is successful" << endl;
                }

                auto end = std::chrono::high_resolution_clock::now();
                auto duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
                double time_per_iteration_ms = duration_ms.count();
                total_duration_ms += time_per_iteration_ms;
                std::cout << "Time taken per inference: " << time_per_iteration_ms << " milliseconds" << std::endl;

                char* data = (char*)outputTensorsList[0].getPtr().get();

                // Process the YOLO output and draw bounding boxes
                frame = process_yolo_output(data, frame, labels, time_per_iteration_ms, pad_x, pad_y, scale);

                // Display the processed frame
                // cv::cvtColor(frame, frame, cv::COLOR_RGB2BGR);
                cv::imshow("Inference Result", frame);
                // Wait for 1 ms and break if 'q' is pressed
                if (cv::waitKey(1) == 'q') {
                    break;
                }

                count++;
            } catch (const cv::Exception& e) {
            std::cerr << "OpenCV error occurred while processing frame: " << e.what() << std::endl;
            cleanup_sima_device();
            continue;
            }
        }
        cv::destroyAllWindows();
        // Calculate average duration
        double average_duration_ms = total_duration_ms /(count-1); // ommiting the first inference time.
        std::cout << "Average time per iteration/ inference: " << average_duration_ms << " ms" << std::endl;
        
        cleanup_sima_device();
    }
    // }
    if (null_flag) {
        return false;
    } else {
        return true;
    }
}

int main(int argc, char *argv[]) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " project.mpk images_folder or videosrc 0 for the webcamera or rtsp-url " << std::endl;
        return 1;
    }
    std::string model_path = argv[1];
    runInferenceSynchronousloadModel_cppsdkpipeline(argv);
    return 0;
}

