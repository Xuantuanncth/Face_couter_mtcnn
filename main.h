#include <string>
#include <sys/stat.h>
#include <errno.h>
#include <sys/types.h>
#include <bits/stdc++.h>

#include <fstream>

#define ImageDir "/home/pi/logs/FaceImg"
#define FileConfig "/home/pi/TuanDX/face-detection-MTCNN-ncnn/config/config.txt"
#define CpuInfo "/proc/cpuinfo"

#define THINGSPEAK_HOST "https://api.thingspeak.com"
#define API_KEY "J90J3CRBLCF9X3NG"
#define TIMEOUT_IN_SECS 15
#define SERVER_HOST_POST "http://192.168.6.125:3000/api/camera/postData"
#define SERVER_HOST_UPLOAD "http://192.168.6.125:3000/api/camera/uploadData"

char Token_Server[300] = "Authorization:Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhdXRob3IiOiJNUV9JQ1RfU09MVVRJT04iLCJ2ZXIiOiIyMDIwIiwidXNlciI6InBpIiwibmFtZSI6IkRpZ2l0YWwgRGV2aWNlIiwiaWF0IjoxNTk1NDE1MDYwfQ.6pMvhNs0k9t1tMU5nm5F5KW5S1i7N_zWVTK1BpoBCXY";

void __INIT();
void SaveImage(char dir[], int id, const cv::Mat &frame, cv::Rect_<float> tb);
bool CreateFolder(char path_file[]);
void SendToThingSpeak(int face_number, int people);