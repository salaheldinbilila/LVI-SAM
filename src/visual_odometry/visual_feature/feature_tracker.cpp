#include "feature_tracker.h"

int FeatureTracker::n_id = 0;

bool inBorder(const cv::Point2f &pt)
{
    const int BORDER_SIZE = 1;
    int img_x = cvRound(pt.x);
    int img_y = cvRound(pt.y);
    return BORDER_SIZE <= img_x && img_x < COL - BORDER_SIZE && BORDER_SIZE <= img_y && img_y < ROW - BORDER_SIZE;
}

void reduceVector(vector<cv::Point2f> &v, vector<uchar> status)
{
    int j = 0;
    for (int i = 0; i < int(v.size()); i++)
        if (status[i])
            v[j++] = v[i];
    v.resize(j);
}

void reduceVector(vector<int> &v, vector<uchar> status)
{
    int j = 0;
    for (int i = 0; i < int(v.size()); i++)
        if (status[i])
            v[j++] = v[i];
    v.resize(j);
}

void reduceVector(vector<uchar> &v, vector<uchar> status)
{
    int j = 0;
    for (int i = 0; i < int(v.size()); i++)
        if (status[i])
            v[j++] = v[i];
    v.resize(j);
}


FeatureTracker::FeatureTracker()
{
    seg_reject_flag.clear();
    det_reject_flag.clear();
    /*
    if (SEG)
    {
        seg_reject_flag.resize(MAX_CNT);
        fill(seg_reject_flag.begin(),seg_reject_flag.end(),0);
    }
    if (DET)
    {
        seg_reject_flag.resize(MAX_CNT);
        fill(seg_reject_flag.begin(),seg_reject_flag.end(),0);
    }
    */
}

void FeatureTracker::setMask()
{
    if(FISHEYE)
        mask = fisheye_mask.clone();
    else
        mask = cv::Mat(ROW, COL, CV_8UC1, cv::Scalar(255));
    

    // prefer to keep features that are tracked for long time
    vector<pair<int, pair<cv::Point2f, int>>> cnt_pts_id;

    for (unsigned int i = 0; i < forw_pts.size(); i++)
        cnt_pts_id.push_back(make_pair(track_cnt[i], make_pair(forw_pts[i], ids[i])));

    sort(cnt_pts_id.begin(), cnt_pts_id.end(), [](const pair<int, pair<cv::Point2f, int>> &a, const pair<int, pair<cv::Point2f, int>> &b)
         {
            return a.first > b.first;
         });

    forw_pts.clear();
    ids.clear();
    track_cnt.clear();

    for (auto &it : cnt_pts_id)
    {
        if (mask.at<uchar>(it.second.first) == 255)
        {
            forw_pts.push_back(it.second.first);
            ids.push_back(it.second.second);
            track_cnt.push_back(it.first);
            cv::circle(mask, it.second.first, MIN_DIST, 0, -1);
        }
    }
}

void FeatureTracker::setMaskMod()
{
    //mask = cv::Mat(row, col, CV_8UC1, cv::Scalar(255));
    if (DET)
        mask = car_mask.clone();

    // prefer to keep features that are tracked for long time
    vector<pair<pair<int, pair<cv::Point2f, int>>, pair<uchar,uchar>>> cnt_pts_id;
    for (unsigned int i = 0; i < forw_pts.size(); i++)
        cnt_pts_id.push_back(make_pair(make_pair(track_cnt[i], make_pair(forw_pts[i], ids[i])), make_pair(seg_reject_flag[i],det_reject_flag[i])));

    sort(cnt_pts_id.begin(), cnt_pts_id.end(), [](const pair<pair<int, pair<cv::Point2f, int>>, pair<uchar,uchar>> &a, const pair<pair<int, pair<cv::Point2f, int>>, pair<uchar,uchar>> &b)
         { return a.first.first > b.first.first; });

    forw_pts.clear();
    ids.clear();
    track_cnt.clear();
    seg_reject_flag.clear();
    det_reject_flag.clear();

    for (auto &it : cnt_pts_id)
    {
        if (mask.at<uchar>(it.first.second.first) == 255)
        {
            forw_pts.push_back(it.first.second.first);
            ids.push_back(it.first.second.second);
            track_cnt.push_back(it.first.first);
            seg_reject_flag.push_back(it.second.first);
            det_reject_flag.push_back(it.second.second);
            cv::circle(mask, it.first.second.first, MIN_DIST, 0, -1);
        }
    }
}


double FeatureTracker::distance(cv::Point2f &pt1, cv::Point2f &pt2)
{
    // printf("pt1: %f %f pt2: %f %f\n", pt1.x, pt1.y, pt2.x, pt2.y);
    double dx = pt1.x - pt2.x;
    double dy = pt1.y - pt2.y;
    return sqrt(dx * dx + dy * dy);
}


void FeatureTracker::addPoints()
{
    for (auto &p : n_pts)
    {
        forw_pts.push_back(p);
        ids.push_back(n_id++);
        track_cnt.push_back(1);
    }
}

void FeatureTracker::readImage(const cv::Mat &_img, double _cur_time)
{
    cv::Mat img;
    TicToc t_r;
    cur_time = _cur_time;

    if (EQUALIZE)
    {
        cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(3.0, cv::Size(8, 8));
        TicToc t_c;
        clahe->apply(_img, img);
        ROS_DEBUG("CLAHE costs: %fms", t_c.toc());
    }
    else
        img = _img;

    if (forw_img.empty())
    {
        prev_img = cur_img = forw_img = img;
    }
    else
    {
        forw_img = img;
    }

    forw_pts.clear();

    if (cur_pts.size() > 0)
    {
        TicToc t_o;
        vector<uchar> status;
        vector<float> err;
        cv::calcOpticalFlowPyrLK(cur_img, forw_img, cur_pts, forw_pts, status, err, cv::Size(21, 21), 3);

        
        // flow back
        vector<uchar> reverse_status;
        vector<cv::Point2f> reverse_pts = cur_pts;
        cv::calcOpticalFlowPyrLK(forw_img, cur_img, forw_pts, reverse_pts, reverse_status, err, cv::Size(21, 21), 1,
                                   cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 30, 0.01), cv::OPTFLOW_USE_INITIAL_FLOW);
    
        for (size_t i = 0; i < status.size(); i++)
        {
            if (status[i] && reverse_status[i] && distance(cur_pts[i], reverse_pts[i]) <= 0.5)
                status[i] = 1;
            else
                status[i] = 0;
        }
        
        for (int i = 0; i < int(forw_pts.size()); i++)
            if (status[i] && !inBorder(forw_pts[i]))
                status[i] = 0;
        reduceVector(prev_pts, status);
        reduceVector(cur_pts, status);
        reduceVector(forw_pts, status);
        reduceVector(ids, status);
        reduceVector(cur_un_pts, status);
        reduceVector(track_cnt, status);
        if(SEG || DET)
        {
            reduceVector(seg_reject_flag, status);
            reduceVector(det_reject_flag, status);
        }
        ROS_DEBUG("temporal optical flow costs: %fms", t_o.toc());
    }

    for (auto &n : track_cnt)
        n++;

    if (1)//(PUB_THIS_FRAME)
    {
        // rejectWithF();
        ROS_DEBUG("set mask begins");
        TicToc t_m;
        if (SEG || DET)
            setMaskMod();
        else
            setMask();
        ROS_DEBUG("set mask costs %fms", t_m.toc());

        ROS_DEBUG("detect feature begins");
        TicToc t_t;
        int n_max_cnt = MAX_CNT - static_cast<int>(forw_pts.size());
        if (n_max_cnt > 0)
        {
            if(mask.empty())
                cout << "mask is empty " << endl;
            if (mask.type() != CV_8UC1)
                cout << "mask type wrong " << endl;
            if (mask.size() != forw_img.size())
                cout << "wrong size " << endl;
            cv::goodFeaturesToTrack(forw_img, n_pts, MAX_CNT - forw_pts.size(), 0.01, MIN_DIST, mask);
        }
        else
            n_pts.clear();
        ROS_DEBUG("detect feature costs: %fms", t_t.toc());
        //  segmentation implementation
        //ROS_INFO("Reject Mask");
        if (SEG && !seg_img.empty())
        {
            reject_mask(seg_img,seg_classes,seg_reject_flag);
        }
        //  detection implementation
        if (DET && !det_img.empty())
        {
            reject_mask(det_img,det_classes,det_reject_flag);
        }

        ROS_DEBUG("add feature begins");
        TicToc t_a;
        addPoints();
        ROS_DEBUG("selectFeature costs: %fms", t_a.toc());
        if (SEG || DET)
        {
            if(seg_reject_flag.size() != forw_pts.size())
            {
                //ROS_INFO("fill up");
                seg_reject_flag.resize(forw_pts.size());
                fill(seg_reject_flag.begin(),seg_reject_flag.end(),0);
            }
        
            if(det_reject_flag.size() != forw_pts.size())
            {
                //ROS_INFO("fill up");
                det_reject_flag.resize(forw_pts.size());
                fill(det_reject_flag.begin(),det_reject_flag.end(),0);
            }
        }
    }
    prev_img = cur_img;
    prev_pts = cur_pts;
    prev_un_pts = cur_un_pts;
    cur_img = forw_img;
    cur_pts = forw_pts;
    undistortedPoints();
    prev_time = cur_time;

    if (SHOW_TRACK)
        drawTrack(cur_img);

    prev_pts_map.clear();
    for (size_t i = 0; i < cur_pts.size(); i++)
        prev_pts_map[ids[i]] = cur_pts[i];
}

void FeatureTracker::rejectWithF()
{
    if (forw_pts.size() >= 8)
    {
        ROS_DEBUG("FM ransac begins");
        TicToc t_f;
        vector<cv::Point2f> un_cur_pts(cur_pts.size()), un_forw_pts(forw_pts.size());
        for (unsigned int i = 0; i < cur_pts.size(); i++)
        {
            Eigen::Vector3d tmp_p;
            m_camera->liftProjective(Eigen::Vector2d(cur_pts[i].x, cur_pts[i].y), tmp_p);
            tmp_p.x() = FOCAL_LENGTH * tmp_p.x() / tmp_p.z() + COL / 2.0;
            tmp_p.y() = FOCAL_LENGTH * tmp_p.y() / tmp_p.z() + ROW / 2.0;
            un_cur_pts[i] = cv::Point2f(tmp_p.x(), tmp_p.y());

            m_camera->liftProjective(Eigen::Vector2d(forw_pts[i].x, forw_pts[i].y), tmp_p);
            tmp_p.x() = FOCAL_LENGTH * tmp_p.x() / tmp_p.z() + COL / 2.0;
            tmp_p.y() = FOCAL_LENGTH * tmp_p.y() / tmp_p.z() + ROW / 2.0;
            un_forw_pts[i] = cv::Point2f(tmp_p.x(), tmp_p.y());
        }

        vector<uchar> status;
        cv::findFundamentalMat(un_cur_pts, un_forw_pts, cv::FM_RANSAC, F_THRESHOLD, 0.99, status);
        int size_a = cur_pts.size();
        reduceVector(prev_pts, status);
        reduceVector(cur_pts, status);
        reduceVector(forw_pts, status);
        reduceVector(cur_un_pts, status);
        reduceVector(ids, status);
        reduceVector(track_cnt, status);
        ROS_DEBUG("FM ransac: %d -> %lu: %f", size_a, forw_pts.size(), 1.0 * forw_pts.size() / size_a);
        ROS_DEBUG("FM ransac costs: %fms", t_f.toc());
    }
}

// Segmentation & Detection method
void FeatureTracker::reject_mask(const cv::Mat _img, const vector<uchar> classes, vector<uchar>& reject_flag)
{
    uchar val,flag;
    //reject_flag.clear();
    // old points
    for (int i = 0; i < int(forw_pts.size()); i++)
    {
        if (reject_flag[i] == 0)
        {
            val = _img.at<uchar>(forw_pts[i]);
            for (auto it : classes)
            {
                if (val == it)
                {
                    reject_flag[i] = 1;
                    break;
                }
            }
        }
    }
    // new points
    for (int i = 0; i < int(n_pts.size()); i++)
    {
        flag = 0;
        //printf("current point: %f,%f\n",cur_pts[i].y,cur_pts[i].x);
        val = _img.at<uchar>(n_pts[i]);
        for (auto it : classes)
        {
            if (val == it)
            {
                flag = 1;
                break;
            }
        }
        reject_flag.push_back(flag);
    }
}

bool FeatureTracker::updateID(unsigned int i)
{
    if (i < ids.size())
    {
        if (ids[i] == -1)
            ids[i] = n_id++;
        return true;
    }
    else
        return false;
}

void FeatureTracker::readIntrinsicParameter(const string &calib_file)
{
    ROS_INFO("reading paramerter of camera %s", calib_file.c_str());
    m_camera = CameraFactory::instance()->generateCameraFromYamlFile(calib_file);
}

void FeatureTracker::showUndistortion(const string &name)
{
    cv::Mat undistortedImg(ROW + 600, COL + 600, CV_8UC1, cv::Scalar(0));
    vector<Eigen::Vector2d> distortedp, undistortedp;
    for (int i = 0; i < COL; i++)
        for (int j = 0; j < ROW; j++)
        {
            Eigen::Vector2d a(i, j);
            Eigen::Vector3d b;
            m_camera->liftProjective(a, b);
            distortedp.push_back(a);
            undistortedp.push_back(Eigen::Vector2d(b.x() / b.z(), b.y() / b.z()));
            //printf("%f,%f->%f,%f,%f\n)\n", a.x(), a.y(), b.x(), b.y(), b.z());
        }
    for (int i = 0; i < int(undistortedp.size()); i++)
    {
        cv::Mat pp(3, 1, CV_32FC1);
        pp.at<float>(0, 0) = undistortedp[i].x() * FOCAL_LENGTH + COL / 2;
        pp.at<float>(1, 0) = undistortedp[i].y() * FOCAL_LENGTH + ROW / 2;
        pp.at<float>(2, 0) = 1.0;
        //cout << trackerData[0].K << endl;
        //printf("%lf %lf\n", p.at<float>(1, 0), p.at<float>(0, 0));
        //printf("%lf %lf\n", pp.at<float>(1, 0), pp.at<float>(0, 0));
        if (pp.at<float>(1, 0) + 300 >= 0 && pp.at<float>(1, 0) + 300 < ROW + 600 && pp.at<float>(0, 0) + 300 >= 0 && pp.at<float>(0, 0) + 300 < COL + 600)
        {
            undistortedImg.at<uchar>(pp.at<float>(1, 0) + 300, pp.at<float>(0, 0) + 300) = cur_img.at<uchar>(distortedp[i].y(), distortedp[i].x());
        }
        else
        {
            //ROS_ERROR("(%f %f) -> (%f %f)", distortedp[i].y, distortedp[i].x, pp.at<float>(1, 0), pp.at<float>(0, 0));
        }
    }
    cv::imshow(name, undistortedImg);
    cv::waitKey(0);
}

void FeatureTracker::drawTrack(const cv::Mat &image)
{
    cv::cvtColor(image, imTrack, CV_GRAY2RGB);

    for (unsigned int j = 0; j < cur_pts.size(); j++)
    {
        
            // track count
            //double len = std::min(1.0, 1.0 * trackerData[i].track_cnt[j] / WINDOW_SIZE);
            //cv::circle(tmp_img, trackerData[i].cur_pts[j], 4, cv::Scalar(255 * (1 - len), 255 * len, 0), 4);

            if((SEG||DET) && (seg_reject_flag[j] || det_reject_flag[j]))
            {
                cv::circle(imTrack, cur_pts[j], 5, cv::Scalar(0, 255, 0), 5);
            }
            else
            {
                double len = std::min(1.0, 1.0 * track_cnt[j] / WINDOW_SIZE);
                cv::circle(imTrack, cur_pts[j], 3, cv::Scalar(255 * (1 - len), 0, 255 * len), 3);
            }
    }
    map<int, cv::Point2f>::iterator mapIt;
    for (size_t i = 0; i < ids.size(); i++)
    {
        mapIt = prev_pts_map.find(ids[i]);
        if (mapIt != prev_pts_map.end())
        {
            cv::arrowedLine(imTrack, cur_pts[i], mapIt->second, cv::Scalar(0, 255, 0), 1, 8, 0, 0.2);
        }
    }
}

void FeatureTracker::undistortedPoints()
{
    cur_un_pts.clear();
    cur_un_pts_map.clear();
    //cv::undistortPoints(cur_pts, un_pts, K, cv::Mat());
    for (unsigned int i = 0; i < cur_pts.size(); i++)
    {
        Eigen::Vector2d a(cur_pts[i].x, cur_pts[i].y);
        Eigen::Vector3d b;
        m_camera->liftProjective(a, b);
        cur_un_pts.push_back(cv::Point2f(b.x() / b.z(), b.y() / b.z()));
        cur_un_pts_map.insert(make_pair(ids[i], cv::Point2f(b.x() / b.z(), b.y() / b.z())));
        //printf("cur pts id %d %f %f", ids[i], cur_un_pts[i].x, cur_un_pts[i].y);
    }
    // caculate points velocity
    if (!prev_un_pts_map.empty())
    {
        double dt = cur_time - prev_time;
        pts_velocity.clear();
        for (unsigned int i = 0; i < cur_un_pts.size(); i++)
        {
            if (ids[i] != -1)
            {
                std::map<int, cv::Point2f>::iterator it;
                it = prev_un_pts_map.find(ids[i]);
                if (it != prev_un_pts_map.end())
                {
                    double v_x = (cur_un_pts[i].x - it->second.x) / dt;
                    double v_y = (cur_un_pts[i].y - it->second.y) / dt;
                    pts_velocity.push_back(cv::Point2f(v_x, v_y));
                }
                else
                    pts_velocity.push_back(cv::Point2f(0, 0));
            }
            else
            {
                pts_velocity.push_back(cv::Point2f(0, 0));
            }
        }
    }
    else
    {
        for (unsigned int i = 0; i < cur_pts.size(); i++)
        {
            pts_velocity.push_back(cv::Point2f(0, 0));
        }
    }
    prev_un_pts_map = cur_un_pts_map;
}

cv::Mat FeatureTracker::getTrackImage()
{
    return imTrack;
}