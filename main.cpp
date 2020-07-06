#include <iostream>

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "Mtcnn.h"

// Tracking
#include <thread>
#include <sys/time.h>
#include "KalmanTracker.h"
#include "Hungarian.h"
#include <set>
#include <queue> 

using namespace std;
using namespace cv;

static double iouThreshold = 0.1;
static int min_hits = 2;
static int max_age = 5;
int g_frame_count = 0;
bool g_is_update_detect_info_flag = true;
bool g_is_start_track = false;

int face_num = 0;

typedef struct TrackingBox
{
    int frame;
    int id;
    Rect_<float> box;
}TrackingBox;
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


int PlotDetectionResult(const Mat& frame, const std::vector<SMtcnnFace>& bbox)
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


void Tracking(Mat frame, const std::vector<SMtcnnFace>& bboxes){
    vector<TrackingBox> detData;
    //cout << "Tracking 1 +++++: "<< endl;
    for(auto it = bboxes.begin(); it != bboxes.end(); it++) {
        TrackingBox tb;
        tb.box = Rect_<float>(Point_<float>(it->boundingBox[0], it->boundingBox[1]), Point_<float>(it->boundingBox[2],it->boundingBox[3]));
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
        cerr << "*******UNMATCH" << endl;
        KalmanTracker tracker = KalmanTracker(detData[umd].box);
        trackers.push_back(tracker);
    }
    frameTrackingResult.clear();
    for (auto it = trackers.begin(); it != trackers.end();)
    {
        if ((((*it).m_time_since_update < 1) &&
            ((*it).m_hit_streak >= min_hits)) || ((*it).m_isCounted == 1))
        {

            TrackingBox res;
            res.box = (*it).get_state();
            if ( !(*it).m_isCounted) {
                (*it).m_index = face_num++;
                (*it).m_isCounted = 1;
            }
                
            //res.id = (*it).m_id + 1;
            res.id  = (*it).m_index;
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
    for (auto tb : frameTrackingResult) {
        //cout << "id: " << tb.id << endl;
        
        char text_buf[16];
        memset(text_buf, 0, 16);
        if (tb.id == -1)
            continue;
        sprintf(text_buf, "ID: %d", tb.id);
        cv::rectangle(frame, tb.box, cvScalar(255, 0, 0), 1, 1, 0);
        putText(frame, text_buf, cvPoint(tb.box.x, tb.box.y - 10), FONT_HERSHEY_COMPLEX_SMALL, 0.5, cvScalar(255, 0, 0), 1, CV_AA);  
        
    } 
    cout << "Face_num: " << face_num << endl;
                                                                             

}

int main(int argc, char** argv)
{
    int camera = 0;
    cout << "argc: " << argc << endl;
    if(argc > 1) {
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

    thread th1(Tracking)

    double sumMs = 0;
    int count = 0;

    while (1)
    {
        cap >> frame;
        std::vector<SMtcnnFace> finalBbox;
        queue <SMtcnnFace> buffer;

        if (!bSetParamToMtcnn && frame.cols > 0)
        {
            SImageFormat format(frame.cols, frame.rows, eBGR888);
            cout << "rows:" << frame.rows << endl;
            cout << "cols:" << frame.cols << endl;
            const float faceScoreThreshold[3] = { 0.6f, 0.6f, 0.6f };
            mtcnn.SetParam(format, 40, 0.709, 4, faceScoreThreshold);
            bSetParamToMtcnn = true;
        }

        mtcnn.Detect(frame.data, finalBbox);
        buffer.push(finalBbox);
        
        //int songuoi = PlotDetectionResult(frame, finalBbox);

       //cv::putText(frame, "People: "+to_string(songuoi), cv::Point(10,50), cv::FONT_HERSHEY_DUPLEX, 1, cv::Scalar(0,0,255),2);

        Tracking(frame, finalBbox);
        putText(frame, "People: "+to_string(face_num), cv::Point(10,20), cv::FONT_HERSHEY_DUPLEX, 0.5, cv::Scalar(0,0,255),2);
        imshow("frame", frame);
        if (waitKey(1) == 'q')
            break;
    }
    return 0;
}
