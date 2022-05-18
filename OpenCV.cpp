

#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <iostream>

using namespace std;
using namespace cv;

CascadeClassifier face_cascade;

Mat faceDetection(Mat frame)
{
	Mat frame_gray;
	cvtColor(frame, frame_gray, COLOR_BGR2GRAY);


	std::vector<Rect> faces;
	face_cascade.detectMultiScale(frame_gray, faces);

	for (size_t i = 0; i < faces.size(); ++i)
	{
		Point center(faces[i].x + faces[i].width / 2, faces[i].y + faces[i].height / 2);
		ellipse(frame, center, Size(faces[i].width / 2, faces[i].height / 2), 0, 0, 360, Scalar(0, 0, 255), 6);
		Mat faceROI = frame_gray(faces[i]);
	}

	//namedWindow("Name", WINDOW_NORMAL); //create a window
	imshow("nn", frame);


	return frame;
}


int main()
{
	string faceClassifier = "data_face.xml";

	if (!face_cascade.load(faceClassifier))
	{
		cout << "Could not load Classifier";
		return -1;
	}

	cout << "Classifier loaded!\n";


	Mat image = imread("ImageTest1.jpg", IMREAD_COLOR);

	faceDetection(image);

	waitKey(0);

	//	//open the video file for reading
	//VideoCapture cap("Disciples_01.avi");
	//
	//// if not success, exit program
	//if (cap.isOpened() == false)
	//{
	//	cout << "Cannot open the video file" << endl;
	//	cin.get(); //wait for any key press
	//	return -1;
	//}
	//
	////Uncomment the following line if you want to start the video in the middle
	////cap.set(CAP_PROP_POS_MSEC, 300); 
	//
	////get the frames rate of the video
	//double fps = cap.get(CAP_PROP_FPS);
	//cout << "Frames per seconds : " << fps << endl;
	//
	//String window_name = "My First Video";
	//
	//namedWindow(window_name, WINDOW_NORMAL); //create a window
	//
	//while (true)
	//{
	//	Mat frame;
	//	bool bSuccess = cap.read(frame); // read a new frame from video 
	//
	//	//Breaking the while loop at the end of the video
	//	if (bSuccess == false)
	//	{
	//		cout << "Found the end of the video" << endl;
	//		break;
	//	}
	//
	//	//show the frame in the created window
	//	imshow(window_name, faceDetection(frame));
	//
	//	//wait for for 10 ms until any key is pressed.  
	//	//If the 'Esc' key is pressed, break the while loop.
	//	//If the any other key is pressed, continue the loop 
	//	//If any key is not pressed withing 10 ms, continue the loop
	//	if (waitKey(10) == 27)
	//	{
	//		cout << "Esc key is pressed by user. Stoppig the video" << endl;
	//		break;
	//	}
	//}

	return 0;
}





//#include <iostream>
//#include <fstream>
//#include <opencv2/opencv.hpp>
//#include <opencv2/dnn.hpp>
//#include <opencv2/dnn/all_layers.hpp>
//
//using namespace std;
//using namespace cv;
//using namespace dnn;
//
//
//int main(int, char**) {
//
//    string file_path = "C/Users/Administ/";
//    vector<string> class_names;
//    ifstream ifs(string(file_path + "object_detection_classes_coco.txt").c_str());
//    string line;
//
//    // Load in all the classes from the file
//    while (getline(ifs, line))
//    {
//        cout << line << endl;
//        class_names.push_back(line);
//    }
//
//
//    // Read in the neural network from the files
//    auto net = readNet(file_path + "frozen_inference_graph.pb",
//        file_path + "ssd_mobilenet_v2_coco_2018_03_29.pbtxt.txt", "TensorFlow");
//
//
//    // Open up the webcam
//    VideoCapture cap("Disciples_01.avi");
//
//
//    // Run on either CPU or GPU
//    //net.setPreferableBackend(DNN_BACKEND_CUDA);
//    //net.setPreferableTarget(DNN_TARGET_CUDA);
//
//
//    // Set a min confidence score for the detections
//    float min_confidence_score = 0.5;
//
//
//    // Loop running as long as webcam is open and "q" is not pressed
//    while (cap.isOpened()) {
//
//        // Load in an image
//        Mat image;
//        bool isSuccess = cap.read(image);
//
//        // Check if image is loaded in correctly
//        if (!isSuccess) {
//            cout << "Could not load the image!" << endl;
//            break;
//        }
//
//        int image_height = image.cols;
//        int image_width = image.rows;
//
//
//
//        auto start = getTickCount();
//
//        // Create a blob from the image
//        Mat blob = blobFromImage(image, 1.0, Size(300, 300), Scalar(127.5, 127.5, 127.5),
//            true, false);
//
//
//        // Set the blob to be input to the neural network
//        net.setInput(blob);
//
//        // Forward pass of the blob through the neural network to get the predictions
//        Mat output = net.forward();
//
//        auto end = getTickCount();
//
//
//
//        // Matrix with all the detections
//        Mat results(output.size[2], output.size[3], CV_32F, output.ptr<float>());
//
//        // Run through all the predictions
//        for (int i = 0; i < results.rows; i++) {
//            int class_id = int(results.at<float>(i, 1));
//            float confidence = results.at<float>(i, 2);
//
//            // Check if the detection is over the min threshold and then draw bbox
//            if (confidence > min_confidence_score) {
//                int bboxX = int(results.at<float>(i, 3) * image.cols);
//                int bboxY = int(results.at<float>(i, 4) * image.rows);
//                int bboxWidth = int(results.at<float>(i, 5) * image.cols - bboxX);
//                int bboxHeight = int(results.at<float>(i, 6) * image.rows - bboxY);
//                rectangle(image, Point(bboxX, bboxY), Point(bboxX + bboxWidth, bboxY + bboxHeight), Scalar(0, 0, 255), 2);
//                string class_name = class_names[class_id - 1];
//                putText(image, class_name + " " + to_string(int(confidence * 100)) + "%", Point(bboxX, bboxY - 10), FONT_HERSHEY_SIMPLEX, 1.5, Scalar(0, 255, 0), 2);
//            }
//        }
//
//
//        auto totalTime = (end - start) / getTickFrequency();
//
//
//        putText(image, "FPS: " + to_string(int(1 / totalTime)), Point(50, 50), FONT_HERSHEY_DUPLEX, 1, Scalar(0, 255, 0), 2, false);
//
//        imshow("image", image);
//
//
//        int k = waitKey(10);
//        if (k == 113) {
//            break;
//        }
//    }
//
//    cap.release();
//    destroyAllWindows();
//}
//





//#include <opencv2/imgcodecs.hpp>
//#include <opencv2/highgui.hpp>
//#include <opencv2/imgproc.hpp>
//#include <opencv2/objdetect.hpp>
//#include <iostream>
//
//using namespace cv;
//using namespace std;
//
//
//void main() 
//{
//
//	Mat img;
//	VideoCapture cap("Disciples_01.avi");
//
//	CascadeClassifier plateCascade;
//	//plateCascade.load("Resources/haarcascade_russian_plate_number.xml");
//
//	if (plateCascade.empty()) { cout << "XML file not loaded" << endl; }
//
//	vector<Rect> plates;
//
//	while (true) {
//
//		cap.read(img);
//		plateCascade.detectMultiScale(img, plates, 1.1, 10);
//
//		for (int i = 0; i < plates.size(); i++)
//		{
//			Mat imgCrop = img(plates[i]);
//			//imshow(to_string(i), imgCrop);
//			imwrite("Resources/Plates/" + to_string(i) + ".png", imgCrop);
//			rectangle(img, plates[i].tl(), plates[i].br(), Scalar(255, 0, 255), 3);
//		}
//
//		imshow("Image", img);
//		waitKey(1);
//	}
//}




//#include <opencv2/opencv.hpp>
//#include <iostream>
//
//using namespace cv;
//using namespace std;
//
//int main(int argc, char* argv[])
//{
//	//open the video file for reading
//	VideoCapture cap("Disciples_01.avi");
//
//	// if not success, exit program
//	if (cap.isOpened() == false)
//	{
//		cout << "Cannot open the video file" << endl;
//		cin.get(); //wait for any key press
//		return -1;
//	}
//
//	//Uncomment the following line if you want to start the video in the middle
//	//cap.set(CAP_PROP_POS_MSEC, 300); 
//
//	//get the frames rate of the video
//	double fps = cap.get(CAP_PROP_FPS);
//	cout << "Frames per seconds : " << fps << endl;
//
//	String window_name = "My First Video";
//
//	namedWindow(window_name, WINDOW_NORMAL); //create a window
//
//	while (true)
//	{
//		Mat frame;
//		bool bSuccess = cap.read(frame); // read a new frame from video 
//
//		//Breaking the while loop at the end of the video
//		if (bSuccess == false)
//		{
//			cout << "Found the end of the video" << endl;
//			break;
//		}
//
//		//show the frame in the created window
//		imshow(window_name, frame);
//
//		//wait for for 10 ms until any key is pressed.  
//		//If the 'Esc' key is pressed, break the while loop.
//		//If the any other key is pressed, continue the loop 
//		//If any key is not pressed withing 10 ms, continue the loop
//		if (waitKey(10) == 27)
//		{
//			cout << "Esc key is pressed by user. Stoppig the video" << endl;
//			break;
//		}
//	}
//
//	return 0;
//
//}