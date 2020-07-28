#include <iostream>

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "Mtcnn.h"
#include "main.h"

// Tracking
#include <thread>
#include <sys/time.h>
#include "KalmanTracker.h"
#include "Hungarian.h"
#include <set>
#include "queue/bounded_queue.hpp"
#include <curl/curl.h>
#include <curl/easy.h>
#include <ctime>
#include <sys/stat.h>
#include <fcntl.h>

//Create file

using namespace std;
using namespace cv;

static double iouThreshold = 0.1;
static int min_hits = 5;
static int max_age = 5;
int g_frame_count = 0;
bool g_is_update_detect_info_flag = true;
bool g_is_start_track = false;

int face_num = 0, current_id = 0;
time_t start_time;
int check_play = 0;
char cpu_id[20], last_day[3], last_month[3], last_year[5];

struct FrameInfo
{
    int channel_id;
    unsigned long frame_id;
    cv::Mat mat;
};
typedef struct TrackingBox
{
    int frame;
    int id;
    Rect_<float> box;
} TrackingBox;
vector<KalmanTracker> trackers;

// Computes IOU between two bounding boxes
double GetIOU(Rect_<float> bb_test, Rect_<float> bb_gt)
{
    float in = (bb_test & bb_gt).area();
    float un = bb_test.area() + bb_gt.area() - in;

    if (un < DBL_EPSILON)
        return 0;

    return (double)(in / un);
}

// using queue_t = BoundedQueue<FrameInfo>;

int PlotDetectionResult(const Mat &frame, const std::vector<SMtcnnFace> &bbox)
{
    int counter = 0;
    for (auto it = bbox.begin(); it != bbox.end(); it++)
    {
        //Plot bounding box
        counter++;
        rectangle(frame, Point(it->boundingBox[0], it->boundingBox[1]),
                  Point(it->boundingBox[2], it->boundingBox[3]), Scalar(0, 0, 255), 2, 8, 0);
        //cout << it->boundingBox[0]<<":"<< it->boundingBox[1]<<endl;
        //putText(frame,"Num: "+to_string(counter),Point(it->boundingBox[0],it->boundingBox[1]-5),cv::FONT_HERSHEY_DUPLEX, 1, cv::Scalar(0,0,255),2);
        // Plot facial landmark
        // for (int num = 0; num < 5; num++)
        // {
        //     circle(frame, Point(it->landmark[num], it->landmark[num + 5]), 3, Scalar(0, 255, 255), -1);
        // }
    }
    cout << "Counter: " << counter << endl;
    return counter;
}

int Tracking(char direction_path[], Mat frame, const std::vector<SMtcnnFace> &bboxes)
{
    int people_num = 0;
    vector<TrackingBox> detData;
    //cout << "Tracking 1 +++++: "<< endl;
    for (auto it = bboxes.begin(); it != bboxes.end(); it++)
    {
        TrackingBox tb;
        people_num++;
        tb.box = Rect_<float>(Point_<float>(it->boundingBox[0], it->boundingBox[1]), Point_<float>(it->boundingBox[2], it->boundingBox[3]));
        detData.push_back(tb);
    }
    //cout << "Tracking 2 +++++: "<< endl;
    KalmanTracker::kf_count = 0; // tracking id relies on this, so we have to reset it in each seq.

    // variables used in the for-loop
    vector<Rect_<float>> predictedBoxes;
    vector<vector<double>> iouMatrix;
    vector<int> assignment;
    set<int> unmatchedDetections;
    set<int> unmatchedTrajectories;
    set<int> allItems;
    set<int> matchedItems;
    vector<cv::Point> matchedPairs;
    vector<TrackingBox> frameTrackingResult;
    unsigned int trkNum = 0;
    unsigned int detNum = 0;
    predictedBoxes.clear();
    //cout << "Tracking 3 +++++: "<< endl;
    for (auto it = trackers.begin(); it != trackers.end();)
    {
        Rect_<float> pBox = (*it).predict();
        if (pBox.x >= 0 && pBox.y >= 0)
        {
            predictedBoxes.push_back(pBox);
            it++;
        }
        else
        {
            it = trackers.erase(it);
            //cerr << "Box invalid at frame: " << frame_count << endl;
        }
    }

    trkNum = predictedBoxes.size();
    detNum = detData.size();

    iouMatrix.clear();
    iouMatrix.resize(trkNum, vector<double>(detNum, 0));

    for (unsigned int i = 0; i < trkNum; i++) // compute iou matrix as a distance matrix
    {
        for (unsigned int j = 0; j < detNum; j++)
        {
            // use 1-iou because the hungarian algorithm computes a minimum-cost assignment.
            iouMatrix[i][j] = 1 - GetIOU(predictedBoxes[i], detData[j].box);
        }
    }

    // solve the assignment problem using hungarian algorithm.
    // the resulting assignment is [track(prediction) : detection], with len=preNum
    HungarianAlgorithm HungAlgo;
    assignment.clear();
    HungAlgo.Solve(iouMatrix, assignment);
    // find matches, unmatched_detections and unmatched_predictions
    unmatchedTrajectories.clear();
    unmatchedDetections.clear();
    allItems.clear();
    matchedItems.clear();

    if (detNum > trkNum) // there are unmatched detections
    {
        for (unsigned int n = 0; n < detNum; n++)
            allItems.insert(n);

        for (unsigned int i = 0; i < trkNum; ++i)
            matchedItems.insert(assignment[i]);

        set_difference(allItems.begin(), allItems.end(),
                       matchedItems.begin(), matchedItems.end(),
                       insert_iterator<set<int>>(unmatchedDetections, unmatchedDetections.begin()));
    }
    else if (detNum < trkNum) // there are unmatched trajectory/predictions
    {
        for (unsigned int i = 0; i < trkNum; ++i)
            if (assignment[i] == -1) // unassigned label will be set as -1 in the assignment algorithm
                unmatchedTrajectories.insert(i);
    }
    else
        ;
    // filter out matched with low IOU
    matchedPairs.clear();
    for (unsigned int i = 0; i < trkNum; ++i)
    {
        if (assignment[i] == -1) // pass over invalid values
            continue;
        if (1 - iouMatrix[i][assignment[i]] < iouThreshold)
        {
            unmatchedTrajectories.insert(i);
            unmatchedDetections.insert(assignment[i]);
        }
        else
            matchedPairs.push_back(cv::Point(i, assignment[i]));
    }

    ///////////////////////////////////////
    // 3.3. updating trackers

    // update matched trackers with assigned detections.
    // each prediction is corresponding to a tracker
    int detIdx, trkIdx;
    for (unsigned int i = 0; i < matchedPairs.size(); i++)
    {
        //cerr << "------MATCH" << endl;
        trkIdx = matchedPairs[i].x;
        detIdx = matchedPairs[i].y;
        //trackers[trkIdx].m_type = detData[detIdx].type;
        trackers[trkIdx].update(detData[detIdx].box);
    }
    // create and initialise new trackers for unmatched detections
    for (auto umd : unmatchedDetections)
    {
        //cerr << "*******UNMATCH" << endl;
        KalmanTracker tracker = KalmanTracker(detData[umd].box);
        trackers.push_back(tracker);
    }
    frameTrackingResult.clear();
    for (auto it = trackers.begin(); it != trackers.end();)
    {
        if ((((*it).m_time_since_update < 1) &&
             ((*it).m_hit_streak >= min_hits)))
        {

            TrackingBox res;
            res.box = (*it).get_state();
            if (!(*it).m_isCounted)
            {
                (*it).m_index = face_num++;
                (*it).m_isCounted = 1;
            }

            //res.id = (*it).m_id + 1;
            res.id = (*it).m_index;
            //res.frame = frame_count;
            frameTrackingResult.push_back(res);
            it++;
        }
        else
            it++;

        // remove dead tracklet
        if (it != trackers.end() && (*it).m_time_since_update > max_age)
            it = trackers.erase(it);
    }
    for (auto tb : frameTrackingResult)
    {
        //cout << "id: " << tb.id << endl;

        char text_buf[16];
        memset(text_buf, 0, 16);
        if (tb.id == -1)
            continue;
        sprintf(text_buf, "ID: %d", tb.id);
        if (current_id < tb.id)
        {
            cout << "ID: " << tb.id << " : " << current_id << endl;
            current_id = tb.id;
            SaveImage(direction_path, tb.id, frame, tb.box);
        }
        cv::rectangle(frame, tb.box, cv::Scalar(255, 0, 0), 1, 1, 0);
        putText(frame, text_buf, cv::Point(tb.box.x, tb.box.y - 10), FONT_HERSHEY_COMPLEX_SMALL, 0.5, cv::Scalar(255, 0, 0), 1, 2);
    }
    return people_num;
}

/*
 * Send request to server thingspeak
 * Used curl request send face_number and people in time
*/

void SendToThingSpeak(int face_number, int people)
{
    char url[200];
    CURLcode res = curl_global_init(CURL_GLOBAL_ALL);
    if (res != CURLE_OK)
    {
        cout << "Curl global initialization failed, exiting" << endl;
    }
    CURL *curl = curl_easy_init();
    if (curl)
    {

        curl_easy_setopt(curl, CURLOPT_SSL_VERIFYHOST, 0L);
        curl_easy_setopt(curl, CURLOPT_SSL_VERIFYPEER, 0L);
        sprintf(url, "%s/update?api_key=%s&field1=%d&field2=%d", THINGSPEAK_HOST, API_KEY, face_number, people);
        cout << "Url: " << url << endl;
        curl_easy_setopt(curl, CURLOPT_URL, url);

        res = curl_easy_perform(curl);
        if (res != CURLE_OK)
        {
            fprintf(stderr, "curl_easy_perform() failed: %s\n", curl_easy_strerror(res));
        }
        memset(url, 0, sizeof(url));
        curl_easy_cleanup(curl);
    }
    else
    {
        fprintf(stderr, "curl_easy_init() failed\n");
    }
    curl_global_cleanup();
}

/**
 * Function SendToServer
 * - mode = 1: Send number peoper viewer in time
 * - mode = 2: Send file image_face.zip
 * - fileName : 
*/

int SendToServer(int mode, int face_number, int people, char fileName[])
{
    char data[100];
    char file_location[100];
    char file_name[20];
    char16_t isSendToServer = 0;
    sprintf(file_location, "%s/%s.tar.gz", ImageDir, fileName);
    sprintf(file_name, "%s.tar.gz", fileName);
    CURLcode res = curl_global_init(CURL_GLOBAL_ALL);
    if (res != CURLE_OK)
    {
        cout << "Curl global initialization failed, exiting" << endl;
        isSendToServer = 0;
    }
    CURL *curl = curl_easy_init();
    if (curl)
    {
        struct curl_slist *headers = NULL;
        if (mode == 1)
        {
            headers = curl_slist_append(headers, Token_Server);
            headers = curl_slist_append(headers, "Accept: application/json");
            headers = curl_slist_append(headers, "Content-Type: application/json");

            snprintf(data, sizeof(data), "{\"cpuId\":\"%s\",\"count\":\"%d\"}", cpu_id, people);

            curl_easy_setopt(curl, CURLOPT_SSL_VERIFYHOST, 0L);
            curl_easy_setopt(curl, CURLOPT_SSL_VERIFYPEER, 0L);
            curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
            curl_easy_setopt(curl, CURLOPT_URL, SERVER_HOST_POST);
            curl_easy_setopt(curl, CURLOPT_POST, 1);
            curl_easy_setopt(curl, CURLOPT_POSTFIELDS, data);

            res = curl_easy_perform(curl);
            cout << "Res: " << res << endl;
            if (res != CURLE_OK)
            {
                fprintf(stderr, "curl_easy_perform() failed: %s\n", curl_easy_strerror(res));
                isSendToServer = 0;
            }
            else
            {
                isSendToServer = 1;
            }
        }
        else if (mode == 2)
        {
            struct stat file_info;
            curl_off_t speed_upload, total_time;
            struct curl_httppost *formpost = NULL;
            struct curl_httppost *lastptr = NULL;
            FILE *file;
            int file_length = 0;
            char *data_File = NULL;
            int read_byte = 0;
            file = fopen(file_location, "rb"); /* open file to upload */
            if (file == NULL)
            {
                cout << "Can't open file" << endl;
                isSendToServer = 0;
                return isSendToServer; /* can't continue */
            }
            fseek(file, 0, SEEK_END);
            file_length = ftell(file);
            fseek(file, 0, SEEK_SET);
            data_File = new char[file_length];
            memset(data_File, 0, file_length);
            read_byte = fread(data_File, sizeof(char), file_length, file);
            if (read_byte != file_length)
            {
                cout << "Can't read file, File_length: " << file_length << ": Read_byte: " << read_byte << endl;
                isSendToServer = 0;
                return isSendToServer;
            }
            curl_formadd(&formpost,
                         &lastptr,
                         CURLFORM_COPYNAME, "cache-control:",
                         CURLFORM_COPYCONTENTS, "no-cache",
                         CURLFORM_END);

            curl_formadd(&formpost,
                         &lastptr,
                         CURLFORM_COPYNAME, "content-type:",
                         CURLFORM_COPYCONTENTS, "multipart/form-data",
                         CURLFORM_END);

            curl_formadd(&formpost,
                         &lastptr,
                         CURLFORM_COPYNAME, "cpuId",
                         CURLFORM_COPYCONTENTS, cpu_id,
                         CURLFORM_END);

            curl_formadd(&formpost, &lastptr,
                         CURLFORM_COPYNAME, "images",
                         CURLFORM_BUFFER, file_name,
                         CURLFORM_BUFFERPTR, data_File,
                         CURLFORM_BUFFERLENGTH, file_length,
                         CURLFORM_END);

            headers = curl_slist_append(headers, Token_Server);

            curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
            curl_easy_setopt(curl, CURLOPT_URL, SERVER_HOST_UPLOAD);
            curl_easy_setopt(curl, CURLOPT_HTTPPOST, formpost);

            res = curl_easy_perform(curl);

            /* Check for errors */
            if (res != CURLE_OK)
            {
                fprintf(stderr, "curl_easy_perform() failed: %s\n", curl_easy_strerror(res));
                isSendToServer = 0;
            }
            else
            {
                /* now extract transfer info */
                curl_easy_getinfo(curl, CURLINFO_SPEED_UPLOAD_T, &speed_upload);
                curl_easy_getinfo(curl, CURLINFO_TOTAL_TIME_T, &total_time);

                fprintf(stderr, "Speed: %" CURL_FORMAT_CURL_OFF_T " bytes/sec during %" CURL_FORMAT_CURL_OFF_T ".%06ld seconds\n",
                        speed_upload,
                        (total_time / 1000000), (long)(total_time % 1000000));
                isSendToServer = 1;
            }
            delete[] data_File;
        }
        curl_easy_cleanup(curl);
    }
    else
    {
        fprintf(stderr, "curl_easy_init() failed\n");
        isSendToServer = 0;
    }
    curl_global_cleanup();
    return isSendToServer;
}

/*
 * Crop Face and save face image
 * file name: id.jpg
*/

void SaveImage(char dir[], int id, const Mat &frame, cv::Rect_<float> tb)
{
    bool isSave = false;
    char img_path[200];
    sprintf(img_path, "%s/%d.jpg", dir, id);
    cout << "SaveImg: " << img_path << endl;
    //Mat Roi(frame, Rect(x_crop, y_crop, width_crop, heigh_crop));
    Mat Roi(frame, tb);
    Mat cropImg;
    Roi.copyTo(cropImg);
    imwrite(img_path, cropImg);
}
/*
 * Check file/folder is exsit
 * path: local path
*/
bool CreateFolder(char folder_name[])
{
    char path_file[30];
    bool is_createFile = false;
    sprintf(path_file, "%s/%s", ImageDir, folder_name);
    if (mkdir(path_file, 0777) == -1)
    {
        cerr << "Error :  " << strerror(errno) << endl;
    }
    else
    {
        cout << " Create file oke" << endl;
        is_createFile = true;
    }
    return is_createFile;
}

/**
 * Using tar to compressTheFile
 * filename: yyyy_mm_dd
*/
void CompressTheFile(char filename[])
{
    char commnad[100];
    sprintf(commnad, "tar -zcvf %s/%s.tar.gz %s/%s", ImageDir, filename, ImageDir, filename);
    system(commnad);
}

void DeleteFile(char filename[])
{
    char deleteFile[100];
    sprintf(deleteFile, "rm -rf %s/%s %s/%s.tar.gz", ImageDir, filename, ImageDir, filename);
    cout << "File: " << deleteFile << endl;
    system(deleteFile);
}

/**
 * Init
 * - Read Config.txt
 * - Check create folder
 * 
*/
void __INIT()
{
    string line;
    char current_time[100];
    char date_temp[15];
    char imangeFileName[15];
    char imagePath[60];

    ifstream file(CpuInfo);
    time_t start_time = time(0);
    tm *ltm = localtime(&start_time);
    sprintf(current_time, "%d_%.2d_%.2d", (1900 + ltm->tm_year), 1 + ltm->tm_mon, (ltm->tm_mday));
    if (CreateFolder(current_time))
    {
    }
    if (file.is_open())
    {
        while (getline(file, line))
        {
            if ((line.compare(0, strlen("Serial"), "Serial")) == 0)
            {
                string data = line.substr(line.length() - 16);
                strcpy(cpu_id, data.c_str());
            }
        }
        file.close();
    }
    else
    {
        cout << "Can't open file" << endl;
    }
    if (cpu_id[0] == '1')
    {
        cpu_id[0] = '9';
    }
    ifstream file2(FileConfig);
    if (file2.is_open())
    {
        int key_length = strlen("lastupdate");

        while (getline(file2, line))
        {
            if (line.compare(0, key_length, "lastupdate") == 0)
            {
                uint16_t j = 0;
                string date = line.substr(line.length() - (key_length));
                strcpy(date_temp, date.c_str());
                for (uint16_t i = 0; i < strlen(date_temp); i++)
                {
                    if (i < 4)
                    {
                        last_year[j++] = date_temp[i];
                    }
                    else if (i == 4 || i == 7)
                    {
                        j = 0;
                    }
                    else if (i > 4 && i < 7)
                    {
                        last_month[j++] = date_temp[i];
                    }
                    else if (i > 7)
                    {
                        last_day[j++] = date_temp[i];
                    }
                }
                cout << "date_temp: " << date_temp << endl;
            }
            else if (line.compare(0, 6, "update") == 0)
            {
                string isupdate = line.substr(line.length() - 1);
                cout << "isupdate: " << isupdate << endl;
            }
        }
        file2.close();
    }
    else
    {
        cout << "Can't open file " << endl;
    }
    /*
    if ((1900 + ltm->tm_year) > atoi(last_year))
    {
        if ((1 + ltm->tm_mon) > atoi(last_month))
        {
        }
        else
        {
            if (ltm->tm_mday > atoi(last_day))
            {
            }
            else
            {
            }
        }
    }
    else
    {
        if ((1 + ltm->tm_mon) > atoi(last_month))
        {
        }
        else
        {
            if (ltm->tm_mday > atoi(last_day))
            {
                sprintf(imagePath, "%s/%s.tar.gz", ImageDir, date_temp);
                sprintf(imangeFileName, "%s.tar.gz", date_temp);
                Archives_file(date_temp);
                if (SendToServer(2, 0, 0, imagePath, imangeFileName))
                {
                    cout << "Send server oke" << endl;
                }
                else
                {
                    cout << "Send server error" << endl;
                }
            }
            else
            {
                cout << "Has updated already" << endl;
            }
        }
    }
    */
}

int main(int argc, char **argv)
{
    int camera = 0;
    int current_date = 0;
    char dir[100];
    char timeNow[10], lastTime[10];
    uint16_t isSendServer = 0;
    __INIT();
    if (argc > 1)
    {
        cout << "argv[1]: " << argv[1] << endl;
        camera = atoi(argv[1]);
    }
    VideoCapture cap(camera);

    if (!cap.isOpened())
    {
        cout << "video is not open" << endl;
        return -1;
    }
    cap.set(cv::CAP_PROP_FRAME_WIDTH, 640);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, 480);
    Mat frame;
    CMtcnn mtcnn;
    bool bSetParamToMtcnn = false;
    mtcnn.LoadModel("det1.param", "det1.bin", "det2.param", "det2.bin", "det3.param", "det3.bin");
    time_t start_time = time(0);
    tm *ltm = localtime(&start_time);
    sprintf(timeNow, "%d_%.2d_%.2d", (1900 + ltm->tm_year), (1 + ltm->tm_mon), ltm->tm_mday);
    strcpy(lastTime, timeNow);
    current_date = ltm->tm_mday;
    while (1)
    {
        cap >> frame;
        std::vector<SMtcnnFace> finalBbox;
        /*Update 28/7/2020 ================> Check and send file to server*/
        time_t now = time(0);
        tm *ltm = localtime(&now);

        if (current_date != ltm->tm_mday)
        {
            sprintf(timeNow, "%d_%.2d_%.2d", (1900 + ltm->tm_year), (1 + ltm->tm_mon), ltm->tm_mday);
            cout << "timeNow: " << timeNow << endl;
            if (CreateFolder(timeNow))
            {
                cout << "Create oke" << endl;
            }
            else
            {
                cout << "File is exist" << endl;
            }
            CompressTheFile(lastTime);
            isSendServer = SendToServer(2, 0, 0, lastTime);
            if (isSendServer)
            {
                DeleteFile(lastTime);
                strcpy(lastTime, timeNow);
            }
            current_date = ltm->tm_mday;
        }
        sprintf(dir, "%s/%s", ImageDir, timeNow);
        if (!bSetParamToMtcnn && frame.cols > 0)
        {
            SImageFormat format(frame.cols, frame.rows, eBGR888);
            cout << "rows:" << frame.rows << endl;
            cout << "cols:" << frame.cols << endl;
            const float faceScoreThreshold[3] = {0.6f, 0.6f, 0.6f};
            mtcnn.SetParam(format, 40, 0.709, 4, faceScoreThreshold);
            bSetParamToMtcnn = true;
        }

        if (!frame.empty())
        {
            mtcnn.Detect(frame.data, finalBbox);
            int people = Tracking(dir, frame, finalBbox);
            putText(frame, "People: " + to_string(face_num), cv::Point(10, 20), cv::FONT_HERSHEY_DUPLEX, 0.5, cv::Scalar(0, 0, 255), 2);

            if (now - start_time >= 600)
            {
                // tm *ltm = localtime(&time_person);
                cout << "10 minute, Check play: " << check_play << endl;
                SendToServer(1, face_num, people, " ");
                start_time = now;
                check_play++;
            }
            imshow("frame", frame);
        }
        if (waitKey(1) == 'q')
            break;
    }
    return 0;
}
