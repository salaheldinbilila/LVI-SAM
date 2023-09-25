#include "feature_tracker.h"

#define SHOW_UNDISTORTION 0

#define time_out_count 25

// mtx lock for two threads
std::mutex mtx_lidar;
std::mutex m_buf;

// global variable for saving the depthCloud shared between two threads
pcl::PointCloud<PointType>::Ptr depthCloud(new pcl::PointCloud<PointType>());
pcl::PointCloud<PointType>::Ptr depthCloudLocal(new pcl::PointCloud<PointType>());

// global variables saving the lidar point cloud
deque<pcl::PointCloud<PointType>> cloudQueue;
deque<double> timeQueue;

// queues for images
queue<sensor_msgs::ImageConstPtr> img_buf;
queue<sensor_msgs::ImageConstPtr> seg_buf;
queue<sensor_msgs::ImageConstPtr> det_buf;

// global depth register for obtaining depth of a feature
DepthRegister *depthRegister;

// feature publisher for VINS estimator
ros::Publisher pub_feature;
ros::Publisher pub_match;
ros::Publisher pub_restart;

// feature tracker variables
FeatureTracker trackerData;
double first_image_time;
int pub_count = 1;
bool first_image_flag = true;
double last_image_time = 0;
bool init_pub = 0;

// Image callback
void img_callback(const sensor_msgs::ImageConstPtr &img_msg)
{
    m_buf.lock();
    img_buf.push(img_msg);
    ROS_INFO("img callback successful");
    m_buf.unlock();
}

// Segmentation callback
void seg_callback(const sensor_msgs::ImageConstPtr &img_msg)
{
    m_buf.lock();
    seg_buf.push(img_msg);
    //ROS_INFO("seg callback successful");
    m_buf.unlock();
}

// Detection callback
void det_callback(const sensor_msgs::ImageConstPtr &img_msg)
{
    m_buf.lock();
    det_buf.push(img_msg);
    //ROS_INFO("det callback successful");
    m_buf.unlock();
}

cv::Mat getImageFromMsg(const sensor_msgs::ImageConstPtr &img_msg)
{
    cv_bridge::CvImageConstPtr ptr;
    if (img_msg->encoding == "8UC1")
    {
        sensor_msgs::Image img;
        img.header = img_msg->header;
        img.height = img_msg->height;
        img.width = img_msg->width;
        img.is_bigendian = img_msg->is_bigendian;
        img.step = img_msg->step;
        img.data = img_msg->data;
        img.encoding = "mono8";
        ptr = cv_bridge::toCvCopy(img, sensor_msgs::image_encodings::MONO8);
    }
    else
        ptr = cv_bridge::toCvCopy(img_msg, sensor_msgs::image_encodings::MONO8);
    cv::Mat img = ptr->image.clone();
    return img;
}

/*
void img_callback(const sensor_msgs::ImageConstPtr &img_msg)
{
    double cur_img_time = img_msg->header.stamp.toSec();

    if(first_image_flag)
    {
        first_image_flag = false;
        first_image_time = cur_img_time;
        last_image_time = cur_img_time;
        return;
    }
    // detect unstable camera stream
    if (cur_img_time - last_image_time > 1.0 || cur_img_time < last_image_time)
    {
        ROS_WARN("image discontinue! reset the feature tracker!");
        first_image_flag = true; 
        last_image_time = 0;
        pub_count = 1;
        std_msgs::Bool restart_flag;
        restart_flag.data = true;
        pub_restart.publish(restart_flag);
        return;
    }
    last_image_time = cur_img_time;
    // frequency control
    if (round(1.0 * pub_count / (cur_img_time - first_image_time)) <= FREQ)
    {
        PUB_THIS_FRAME = true;
        // reset the frequency control
        if (abs(1.0 * pub_count / (cur_img_time - first_image_time) - FREQ) < 0.01 * FREQ)
        {
            first_image_time = cur_img_time;
            pub_count = 0;
        }
    }
    else
    {
        PUB_THIS_FRAME = false;
    }

    cv_bridge::CvImageConstPtr ptr;
    if (img_msg->encoding == "8UC1")
    {
        sensor_msgs::Image img;
        img.header = img_msg->header;
        img.height = img_msg->height;
        img.width = img_msg->width;
        img.is_bigendian = img_msg->is_bigendian;
        img.step = img_msg->step;
        img.data = img_msg->data;
        img.encoding = "mono8";
        ptr = cv_bridge::toCvCopy(img, sensor_msgs::image_encodings::MONO8);
    }
    else
        ptr = cv_bridge::toCvCopy(img_msg, sensor_msgs::image_encodings::MONO8);

    cv::Mat show_img = ptr->image;
    cv::Mat image;
    std_msgs::Header header,seg_header,det_header;
    double time = 0 ,seg_time = 0 ,det_time = 0;
    static char empty_count = 0;
    TicToc t_r;
    for (int i = 0; i < NUM_OF_CAM; i++)
    {
        if (DET)
        {
            if(!det_buf.empty())
            {
                empty_count = 0;
                time = img_msg->header.stamp.toSec();
                header = img_msg->header;
                ROS_INFO("Image header: %f",time);
                det_time = det_buf.front()->header.stamp.toSec();
                det_header = det_buf.front()->header;
                ROS_INFO("detection time: %f",det_time);
                ROS_INFO("detection time diff: %f",det_time-time);
                if (det_time < time)
                {
                    ROS_INFO("The detection is of an old image, discard");
                    det_buf.pop();
                }
                else if (det_time > time)
                {
                    ROS_INFO("Image too late. Use image without detection");
                    image = getImageFromMsg(img_msg);
                }
                else //(det_time == time)
                {
                    ROS_INFO("Images match, process both");
                    image = getImageFromMsg(img_msg);
                    trackerData[i].det_img = getImageFromMsg(det_buf.front());
                    det_buf.pop();
                }
                
            }
            
            
            else
            {
                time = img_msg->header.stamp.toSec();
                header = img_msg->header;
                ROS_INFO("Image header: %f",time);
                empty_count++;
                ROS_INFO("Detection buffer empty, count: %d",empty_count);
                if (empty_count == time_out_count)
                {
                    empty_count = 0;
                    ROS_INFO("Buffer still empty. Use image without detection");
                    image = getImageFromMsg(img_msg);
                }
            }
            
        }
        ROS_DEBUG("processing camera %d", i);
        if (i != 1 || !STEREO_TRACK)
        {
            trackerData[i].readImage(ptr->image.rowRange(ROW * i, ROW * (i + 1)), cur_img_time);
        }
        else
        {
            if (EQUALIZE)
            {
                cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE();
                clahe->apply(ptr->image.rowRange(ROW * i, ROW * (i + 1)), trackerData[i].cur_img);
            }
            else
                trackerData[i].cur_img = ptr->image.rowRange(ROW * i, ROW * (i + 1));
        }

        #if SHOW_UNDISTORTION
            trackerData[i].showUndistortion("undistrotion_" + std::to_string(i));
        #endif
    }

    for (unsigned int i = 0;; i++)
    {
        bool completed = false;
        for (int j = 0; j < NUM_OF_CAM; j++)
            if (j != 1 || !STEREO_TRACK)
                completed |= trackerData[j].updateID(i);
        if (!completed)
            break;
    }

   if (PUB_THIS_FRAME)
   {
        pub_count++;
        sensor_msgs::PointCloudPtr feature_points(new sensor_msgs::PointCloud);
        sensor_msgs::ChannelFloat32 id_of_point;
        sensor_msgs::ChannelFloat32 u_of_point;
        sensor_msgs::ChannelFloat32 v_of_point;
        sensor_msgs::ChannelFloat32 velocity_x_of_point;
        sensor_msgs::ChannelFloat32 velocity_y_of_point;

        feature_points->header.stamp = img_msg->header.stamp;
        feature_points->header.frame_id = "vins_body";

        vector<set<int>> hash_ids(NUM_OF_CAM);
        for (int i = 0; i < NUM_OF_CAM; i++)
        {
            vector<uchar> reduce_flag(trackerData[i].cur_pts.size(),0);
            if (SEG || DET)
            {
                for (size_t j = 0; j < reduce_flag.size(); j++)
                {
                    reduce_flag[j] = (SEG && trackerData[i].seg_reject_flag[j]) || (DET && trackerData[i].det_reject_flag[j]);
                }
                //ROS_INFO("flags matching: %d",(reduce_flag == seg_reject_flag) || (reduce_flag == det_reject_flag));
            }
            auto &un_pts = trackerData[i].cur_un_pts;
            auto &cur_pts = trackerData[i].cur_pts;
            auto &ids = trackerData[i].ids;
            auto &pts_velocity = trackerData[i].pts_velocity;
            for (unsigned int j = 0; j < ids.size(); j++)
            {
                if(reduce_flag[j])
                    continue;
                if (trackerData[i].track_cnt[j] > 1)
                {
                    int p_id = ids[j];
                    hash_ids[i].insert(p_id);
                    geometry_msgs::Point32 p;
                    p.x = un_pts[j].x;
                    p.y = un_pts[j].y;
                    p.z = 1;

                    feature_points->points.push_back(p);
                    id_of_point.values.push_back(p_id * NUM_OF_CAM + i);
                    u_of_point.values.push_back(cur_pts[j].x);
                    v_of_point.values.push_back(cur_pts[j].y);
                    velocity_x_of_point.values.push_back(pts_velocity[j].x);
                    velocity_y_of_point.values.push_back(pts_velocity[j].y);
                }
            }
        }

        feature_points->channels.push_back(id_of_point);
        feature_points->channels.push_back(u_of_point);
        feature_points->channels.push_back(v_of_point);
        feature_points->channels.push_back(velocity_x_of_point);
        feature_points->channels.push_back(velocity_y_of_point);

        // get feature depth from lidar point cloud
        pcl::PointCloud<PointType>::Ptr depth_cloud_temp(new pcl::PointCloud<PointType>());
        mtx_lidar.lock();
	*depth_cloud_temp = *depthCloud;
	mtx_lidar.unlock();

        sensor_msgs::ChannelFloat32 depth_of_points = depthRegister->get_depth(img_msg->header.stamp, show_img, depth_cloud_temp, trackerData[0].m_camera, feature_points->points);
        feature_points->channels.push_back(depth_of_points);
        
        // skip the first image; since no optical speed on frist image
        if (!init_pub)
        {
            init_pub = 1;
        }
        else
            pub_feature.publish(feature_points);

        // publish features in image
        if (pub_match.getNumSubscribers() != 0)
        {
            ptr = cv_bridge::cvtColor(ptr, sensor_msgs::image_encodings::RGB8);
            //cv::Mat stereo_img(ROW * NUM_OF_CAM, COL, CV_8UC3);
            cv::Mat stereo_img = ptr->image;

            for (int i = 0; i < NUM_OF_CAM; i++)
            {
                cv::Mat tmp_img = stereo_img.rowRange(i * ROW, (i + 1) * ROW);
                cv::cvtColor(show_img, tmp_img, CV_GRAY2RGB);

                for (unsigned int j = 0; j < trackerData[i].cur_pts.size(); j++)
                {
                    if (SHOW_TRACK)
                    {
                        // track count
                        //double len = std::min(1.0, 1.0 * trackerData[i].track_cnt[j] / WINDOW_SIZE);
                        //cv::circle(tmp_img, trackerData[i].cur_pts[j], 4, cv::Scalar(255 * (1 - len), 255 * len, 0), 4);

                        if((SEG||DET) && (trackerData[i].seg_reject_flag[j] || trackerData[i].det_reject_flag[j]))
                        {
                            cv::circle(tmp_img, trackerData[i].cur_pts[j], 5, cv::Scalar(0, 255, 0), 5);
                        }
                        else
                        {
                            double len = std::min(1.0, 1.0 * trackerData[i].track_cnt[j] / WINDOW_SIZE);
                            cv::circle(tmp_img, trackerData[i].cur_pts[j], 3, cv::Scalar(255 * (1 - len), 0, 255 * len), 3);
                        }

                    } else {
                        // depth 
                        if(j < depth_of_points.values.size())
                        {
                            if (depth_of_points.values[j] > 0)
                                cv::circle(tmp_img, trackerData[i].cur_pts[j], 4, cv::Scalar(0, 255, 0), 4);
                            else
                                cv::circle(tmp_img, trackerData[i].cur_pts[j], 4, cv::Scalar(0, 0, 255), 4);
                        }
                    }
                }
            }

            pub_match.publish(ptr->toImageMsg());
        }
    }
}
*/

void sync_process()
{
    while(ros::ok())
    {
        cv::Mat image,seg_im,det_im;
        double time = 0 ,seg_time = 0 ,det_time = 0;
        std_msgs::Header img_header;
        static char empty_count = 0;
        m_buf.lock();
        // check if the segmentation buffer is filled instead of the image buffer
        if(!img_buf.empty())
        {
            time = img_buf.front()->header.stamp.toSec();
            img_header = img_buf.front()->header;
            //printf("Image time: %f\n",time);
            
            if(first_image_flag)
            {
                first_image_flag = false;
                first_image_time = time;
                last_image_time = time;
                m_buf.unlock();
                continue;
            }
            // detect unstable camera stream
            if (time - last_image_time > 1.0 || time < last_image_time)
            {
                ROS_WARN("image discontinue! reset the feature tracker!");
                first_image_flag = true; 
                last_image_time = 0;
                pub_count = 1;
                std_msgs::Bool restart_flag;
                restart_flag.data = true;
                pub_restart.publish(restart_flag);
                m_buf.unlock();
                continue;
            }
            last_image_time = time;
            /*
            // frequency control
            if (round(1.0 * pub_count / (time - first_image_time)) <= FREQ)
            {
                PUB_THIS_FRAME = true;
                // reset the frequency control
                if (abs(1.0 * pub_count / (time - first_image_time) - FREQ) < 0.01 * FREQ)
                {
                    first_image_time = time;
                    pub_count = 0;
                }
            }
            else
            {
                PUB_THIS_FRAME = false;
            }
            */
            if (SEG)
            {
                if(!seg_buf.empty())
                {
                    seg_time = seg_buf.front()->header.stamp.toSec();
                    ROS_INFO("segmentation time: %f",seg_time);
                    ROS_INFO("segmentation time diff: %f",seg_time-time);
                    if (seg_time < time)
                    {
                        ROS_INFO("The segmentation is of an old image");
                        seg_buf.pop();
                    }
                    else if (seg_time > time)
                    {
                        ROS_INFO("Image too late. Use image without segmentation");
                        image = getImageFromMsg(img_buf.front());
                        img_buf.pop();
                    }
                    else //(seg_time == time)
                    {
                        ROS_INFO("Images match");
                        image = getImageFromMsg(img_buf.front());
                        img_buf.pop();
                        trackerData.seg_img = getImageFromMsg(seg_buf.front());
                        seg_buf.pop();
                    }
                }
                else
                {
                    // TODO: Segmentation and detection seperate empty counters
                    empty_count++;
                    ROS_INFO("segmentation buffer empty, count: %d",empty_count);
                    if (empty_count == time_out_count)
                    {
                        empty_count = 0;
                        ROS_INFO("Buffer still empty. Use image without segmentation");
                        image = getImageFromMsg(img_buf.front());
                        img_buf.pop();
                    }
                }
            }
            else if (DET)
            {
                if(!det_buf.empty())
                {
                    empty_count = 0;
                    det_time = det_buf.front()->header.stamp.toSec();
                    //printf("detection time: %f\n",det_time);
                    //printf("detection time diff: %f\n",det_time-time);
                    if (det_time < time)
                    {
                        //printf("The detection is of an old image, discard\n");
                        det_buf.pop();
                    }
                    else if (det_time > time)
                    {
                        //printf("Image too late. Use image without detection\n");
                        image = getImageFromMsg(img_buf.front());
                        img_buf.pop();
                    }
                    else //(det_time == time)
                    {
                        //printf("Images match, process both\n");
                        image = getImageFromMsg(img_buf.front());
                        img_buf.pop();
                        trackerData.det_img = getImageFromMsg(det_buf.front());
                        det_buf.pop();
                    }
                }
                else
                {
                    empty_count++;
                    //printf("Detection buffer empty, count: %d\n",empty_count);
                    
                    if (empty_count == time_out_count)
                    {
                        empty_count = 0;
                        //printf("Buffer still empty. Use image without detection\n");
                        image = getImageFromMsg(img_buf.front());
                        img_buf.pop();
                    }
                    
                }
            }
            else
            {
                image = getImageFromMsg(img_buf.front());
                img_buf.pop();
            }
        }
        m_buf.unlock();
        if(!image.empty())
        {
            trackerData.readImage(image, time);
            if (1)//(PUB_THIS_FRAME)
            {
                pub_count++;
                sensor_msgs::PointCloudPtr feature_points(new sensor_msgs::PointCloud);
                sensor_msgs::ChannelFloat32 id_of_point;
                sensor_msgs::ChannelFloat32 u_of_point;
                sensor_msgs::ChannelFloat32 v_of_point;
                sensor_msgs::ChannelFloat32 velocity_x_of_point;
                sensor_msgs::ChannelFloat32 velocity_y_of_point;

                feature_points->header.stamp = img_header.stamp;
                feature_points->header.frame_id = "vins_body";

                vector<set<int>> hash_ids(NUM_OF_CAM);
                vector<uchar> reduce_flag(trackerData.cur_pts.size(),0);
                if (SEG || DET)
                {
                    for (size_t j = 0; j < reduce_flag.size(); j++)
                    {
                        reduce_flag[j] = (SEG && trackerData.seg_reject_flag[j]) || (DET && trackerData.det_reject_flag[j]);
                    }
                    //ROS_INFO("flags matching: %d\n",(reduce_flag == trackerData.seg_reject_flag) || (reduce_flag == trackerData.det_reject_flag));
                }
                
                auto &un_pts = trackerData.cur_un_pts;
                auto &cur_pts = trackerData.cur_pts;
                auto &ids = trackerData.ids;
                auto &pts_velocity = trackerData.pts_velocity;
                for (unsigned int j = 0; j < ids.size(); j++)
                {
                    if(reduce_flag[j])
                        continue;
                    if (trackerData.track_cnt[j] > 1)
                    {
                        int p_id = ids[j];
                        hash_ids[0].insert(p_id);
                        geometry_msgs::Point32 p;
                        p.x = un_pts[j].x;
                        p.y = un_pts[j].y;
                        p.z = 1;

                        feature_points->points.push_back(p);
                        id_of_point.values.push_back(p_id);
                        u_of_point.values.push_back(cur_pts[j].x);
                        v_of_point.values.push_back(cur_pts[j].y);
                        velocity_x_of_point.values.push_back(pts_velocity[j].x);
                        velocity_y_of_point.values.push_back(pts_velocity[j].y);
                    }
                }
                feature_points->channels.push_back(id_of_point);
                feature_points->channels.push_back(u_of_point);
                feature_points->channels.push_back(v_of_point);
                feature_points->channels.push_back(velocity_x_of_point);
                feature_points->channels.push_back(velocity_y_of_point);

                // get feature depth from lidar point cloud
                pcl::PointCloud<PointType>::Ptr depth_cloud_temp(new pcl::PointCloud<PointType>());
                mtx_lidar.lock();
                *depth_cloud_temp = *depthCloud;
                mtx_lidar.unlock();
                sensor_msgs::ChannelFloat32 depth_of_points = depthRegister->get_depth(img_header.stamp, image, depth_cloud_temp, trackerData.m_camera, feature_points->points);
                feature_points->channels.push_back(depth_of_points);
                // skip the first image; since no optical speed on frist image
                if (!init_pub)
                {
                    init_pub = 1;
                }
                else
                    pub_feature.publish(feature_points);

                //printf("initial features: %d, after reduction: %d\n",ids.size(),id_of_point.values.size());

                // publish features in image
                if (pub_match.getNumSubscribers() != 0)
                {
                    sensor_msgs::ImagePtr ptr = cv_bridge::CvImage(img_header, "bgr8", trackerData.getTrackImage()).toImageMsg();
                    pub_match.publish(ptr);
                }
            }
        }
        std::chrono::milliseconds dura(2);
        std::this_thread::sleep_for(dura);
    }
}

void lidar_raw_callback(const sensor_msgs::PointCloud2ConstPtr& laser_msg)
{
    static int lidar_count = -1;
    if (++lidar_count % (LIDAR_SKIP+1) != 0)
        return;

   //TODO: 1. convert Lidar to camera frame using configuration T_CL
   //1.1 convert point cloud to PCL
   pcl::PointCloud<PointType>::Ptr laser_cloud_in(new pcl::PointCloud<PointType>());
   pcl::fromROSMsg(*laser_msg, *laser_cloud_in);
   pcl::PointCloud<PointType>::Ptr laser_cloud_in_ds(new pcl::PointCloud<PointType>());
   
   //1.2 Downsample point cloud
   static pcl::VoxelGrid<PointType> downSizeFilter;
   downSizeFilter.setLeafSize(0.2, 0.2, 0.2);
   downSizeFilter.setInputCloud(laser_cloud_in);
   downSizeFilter.filter(*laser_cloud_in_ds);
   *laser_cloud_in = *laser_cloud_in_ds;
   
   //1.3. filter lidar points (only keep points in camera view)
    pcl::PointCloud<PointType>::Ptr laser_cloud_in_filter(new pcl::PointCloud<PointType>());
    for (int i = 0; i < (int)laser_cloud_in->size(); ++i)
    {
        PointType p = laser_cloud_in->points[i];
        if (p.x >= 0 && abs(p.y / p.x) <= 10 && abs(p.z / p.x) <= 10)
            laser_cloud_in_filter->push_back(p);
    }
    *laser_cloud_in = *laser_cloud_in_filter;
    
    // Transform to Camera frame
    pcl::PointCloud<PointType>::Ptr laser_cloud_offset(new pcl::PointCloud<PointType>());
   // ROS_INFO_STREAM("Extrinsic_R in depth image: " << std::endl << RCL[0]);
    //ROS_INFO_STREAM("Extrinsic_t in depth image: " << std::endl << TCL[0].transpose());

    //Eigen::Affine3f transOffset = pcl::getTransformation(L_C_TX, L_C_TY, L_C_TZ, L_C_RX, L_C_RY, L_C_RZ);
    //pcl::transformPointCloud(*laser_cloud_in, *laser_cloud_offset, transOffset);
    //*laser_cloud_in = *laser_cloud_offset;

   //TODO: 2. save it to a global cloud

   //TODO: in image call back project the points to the image using the camera model and publish an image    


}

void lidar_callback(const sensor_msgs::PointCloud2ConstPtr& laser_msg)
{
    static int lidar_count = -1;
    if (++lidar_count % (LIDAR_SKIP+1) != 0)
        return;

    // 0. listen to transform
    static tf::TransformListener listener;
    static tf::StampedTransform transform;
    try{
        listener.waitForTransform("vins_world", "vins_body_ros", laser_msg->header.stamp, ros::Duration(0.01));
        listener.lookupTransform("vins_world", "vins_body_ros", laser_msg->header.stamp, transform);
    } 
    catch (tf::TransformException ex){
        // ROS_ERROR("lidar no tf");
        return;
    }

    double xCur, yCur, zCur, rollCur, pitchCur, yawCur;
    xCur = transform.getOrigin().x();
    yCur = transform.getOrigin().y();
    zCur = transform.getOrigin().z();
    tf::Matrix3x3 m(transform.getRotation());
    m.getRPY(rollCur, pitchCur, yawCur);
    Eigen::Affine3f transNow = pcl::getTransformation(xCur, yCur, zCur, rollCur, pitchCur, yawCur);

    // 1. convert laser cloud message to pcl
    pcl::PointCloud<PointType>::Ptr laser_cloud_in(new pcl::PointCloud<PointType>());
    pcl::fromROSMsg(*laser_msg, *laser_cloud_in);

    // 2. downsample new cloud (save memory)
    pcl::PointCloud<PointType>::Ptr laser_cloud_in_ds(new pcl::PointCloud<PointType>());
    static pcl::VoxelGrid<PointType> downSizeFilter;
    downSizeFilter.setLeafSize(0.2, 0.2, 0.2);
    downSizeFilter.setInputCloud(laser_cloud_in);
    downSizeFilter.filter(*laser_cloud_in_ds);
    *laser_cloud_in = *laser_cloud_in_ds;

    // 3. filter lidar points (only keep points in camera view)
    pcl::PointCloud<PointType>::Ptr laser_cloud_in_filter(new pcl::PointCloud<PointType>());
    for (int i = 0; i < (int)laser_cloud_in->size(); ++i)
    {
        PointType p = laser_cloud_in->points[i];
        if (p.x >= 0 && abs(p.y / p.x) <= 10 && abs(p.z / p.x) <= 10)
            laser_cloud_in_filter->push_back(p);
    }
    *laser_cloud_in = *laser_cloud_in_filter;

    // TODO: transform to IMU body frame
    // 4. offset T_lidar -> T_camera 
    pcl::PointCloud<PointType>::Ptr laser_cloud_offset(new pcl::PointCloud<PointType>());
    Eigen::Affine3f transOffset = pcl::getTransformation(L_C_TX, L_C_TY, L_C_TZ, L_C_RX, L_C_RY, L_C_RZ);
    pcl::transformPointCloud(*laser_cloud_in, *laser_cloud_offset, transOffset);
    *laser_cloud_in = *laser_cloud_offset;

    // 5. transform new cloud into global odom frame
    pcl::PointCloud<PointType>::Ptr laser_cloud_global(new pcl::PointCloud<PointType>());
    pcl::transformPointCloud(*laser_cloud_in, *laser_cloud_global, transNow);

    // 6. save new cloud
    double timeScanCur = laser_msg->header.stamp.toSec();
    cloudQueue.push_back(*laser_cloud_global);
    timeQueue.push_back(timeScanCur);
    *depthCloudLocal = *laser_cloud_in;

    // 7. pop old cloud
    while (!timeQueue.empty())
    {
        if (timeScanCur - timeQueue.front() > 5.0)
        {
            cloudQueue.pop_front();
            timeQueue.pop_front();
        } else {
            break;
        }
    }
    
    std::lock_guard<std::mutex> lock(mtx_lidar);
    
    // 8. fuse global cloud
    depthCloud->clear();
    if (USE_DENSE_CLOUD == 0){
    	*depthCloud += cloudQueue.back();
    }else {
    	for (int i = 0; i < (int)cloudQueue.size(); ++i)
        	*depthCloud += cloudQueue[i];
    }

    // 9. downsample global cloud
    pcl::PointCloud<PointType>::Ptr depthCloudDS(new pcl::PointCloud<PointType>());
    downSizeFilter.setLeafSize(0.2, 0.2, 0.2);
    downSizeFilter.setInputCloud(depthCloud);
    downSizeFilter.filter(*depthCloudDS);
    *depthCloud = *depthCloudDS;
}

int main(int argc, char **argv)
{
    // initialize ROS node
    ros::init(argc, argv, "vins");
    ros::NodeHandle n;
    ROS_INFO("\033[1;32m----> Visual Feature Tracker Started.\033[0m");
    ros::console::set_logger_level(ROSCONSOLE_DEFAULT_NAME, ros::console::levels::Warn);
    readParameters(n);

    // read camera params
    trackerData.readIntrinsicParameter(CAM_NAMES[0]);

    // load fisheye mask to remove features on the boundry
    if(FISHEYE)
    {
        trackerData.fisheye_mask = cv::imread(FISHEYE_MASK, 0);
        if(!trackerData.fisheye_mask.data)
        {
            ROS_ERROR("load fisheye mask fail");
            ROS_BREAK();
        }
        else
            ROS_INFO("load mask success");
    }
    if (DET)
    {
        trackerData.car_mask = cv::imread(CAR_MASK, 0);
        trackerData.bus_mask = cv::imread(BUS_MASK, 0);
    }

    // initialize depthRegister (after readParameters())
    depthRegister = new DepthRegister(n);
    
    // subscriber to image and lidar
    ros::Subscriber sub_img   = n.subscribe(IMAGE_TOPIC,       100,    img_callback);
    ros::Subscriber sub_lidar = n.subscribe(POINT_CLOUD_TOPIC, 100,    lidar_callback);
    ros::Subscriber sub_seg;
    ros::Subscriber sub_det;
    if(SEG)
        sub_seg = n.subscribe("/semantic", 100, seg_callback);      // segmentation subscriber
    if(DET)
        sub_det = n.subscribe("/detect", 100, det_callback);      // detection subscriber
    //ros::Subscriber sub_lidar_raw = n.subscribe("/velodyne_points", 5,    lidar_raw_callback);
    if (!USE_LIDAR)
        sub_lidar.shutdown();

    // messages to vins estimator
    pub_feature = n.advertise<sensor_msgs::PointCloud>(PROJECT_NAME + "/vins/feature/feature",     5);
    pub_match   = n.advertise<sensor_msgs::Image>     (PROJECT_NAME + "/vins/feature/feature_img", 5);
    pub_restart = n.advertise<std_msgs::Bool>         (PROJECT_NAME + "/vins/feature/restart",     5);

    std::thread sync_thread{sync_process};
    // four ROS spinners for parallel processing (segmentation, detection, image and lidar)
    ros::MultiThreadedSpinner spinner(4);
    spinner.spin();

    return 0;
}
