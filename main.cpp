#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/flann.hpp>
#include <fstream>
#include <filesystem>

namespace fs = std::experimental::filesystem;

void convert(const cv::Vec3b& in, cv::Vec3b& out, int code)
{
    thread_local static std::vector<cv::Vec3b> v0(1);
    thread_local static std::vector<cv::Vec3b> v1(1);
    v0[0] = in;
    cvtColor(v0, v1, code);
    out = v1[0];
}

struct Cluster
{
    cv::Vec3b color;
    int count;
};

void sortCentersByColor(std::vector<Cluster>& clusters)
{
    std::sort(clusters.begin(), clusters.end(), [](const Cluster& l, const Cluster& r) {
        cv::Vec3b l_rgb, r_rgb;
        cv::Vec3b l_hsv, r_hsv;
        
        convert(l.color, l_rgb, CV_Lab2RGB);
        convert(r.color, r_rgb, CV_Lab2RGB);

        convert(l_rgb, l_hsv, CV_RGB2HSV);
        convert(r_rgb, r_hsv, CV_RGB2HSV);

        return l_hsv[0] < r_hsv[0] ||
               (l_hsv[0] == r_hsv[0] && l_hsv[1] < r_hsv[1]) ||
               (l_hsv[0] == r_hsv[0] && l_hsv[1] == r_hsv[1] && l_hsv[2] < r_hsv[2]);
    });
}

void sortCentersByCount(std::vector<Cluster>& clusters)
{
    std::sort(clusters.begin(), clusters.end(), [](const Cluster& l, const Cluster& r) {
        return l.count >= r.count;
    });
}

void exportCenters(const std::vector<Cluster>& clusters, const std::string& imgFilename, const std::string& csvFilename)
{
    const int k = clusters.size();

    int w = (512 + k - 1) / k;
    
    std::ofstream f(csvFilename);
    cv::Mat img = cv::Mat::zeros(32, w*k, CV_8UC3);

    std::cout << "k=" << k << "\n";

    int i = 0;
    for(auto it : clusters) {
        cv::Vec3b Lab(it.color[0], it.color[1], it.color[2]);
        cv::Vec3b rgb;
        convert(Lab, rgb, CV_Lab2RGB);
        auto r = (int)rgb[0];
        auto g = (int)rgb[1];
        auto b = (int)rgb[2];

        std::cout << r << ", " << g << ", " << b << "\n";

        f << it.count << ";" << r << ";" << g << ";" << b << ";\n";

        cv::rectangle(img, cv::Rect(i * w, 0, w, 32), cv::Scalar(b, g, r), cv::FILLED);
        ++i;
    }

    cv::imwrite(imgFilename, img);
}

float cluster(cv::Mat Z, int k, std::vector<Cluster>& clusters)
{
    cv::Mat centers;
    std::vector<int> bestLabels;
    float compactness = cv::kmeans(Z, k, bestLabels, cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 10, 1.0), 10, cv::KMEANS_PP_CENTERS, centers);

    std::vector<int> count(k, 0);
    for (int i = 0; i < bestLabels.size(); ++i) {
        count[bestLabels[i]]++;
    }

    for (int i = 0; i < centers.rows; ++i) {
        Cluster c = { centers.row(i), count[i] };
        clusters.push_back(c);
    }

    return compactness;
}

void cluster(cv::Mat Z, float threshold, std::vector<Cluster>& clusters)
{
    auto last_compactness = std::numeric_limits<double>::max();
    for (auto k = 1; k < 100; ++k) {
        float compactness = cluster(Z, k, clusters);
        
        std::cout << compactness / last_compactness << "\n";

        if (compactness / last_compactness > threshold) {
            break;
        }

        last_compactness = compactness;
    }
}

cv::Mat loadImage(const std::string& filename)
{
    auto img = cv::imread(filename);
    cv::imshow("img", img);

    std::vector<cv::Mat> bgr;
    cv::split(img, bgr);

    auto n = img.rows * img.cols;

    cv::Mat Z(n, 3, CV_8U);
    bgr[0].reshape(1, n).copyTo(Z.col(0));
    bgr[1].reshape(1, n).copyTo(Z.col(1));
    bgr[2].reshape(1, n).copyTo(Z.col(2));

    Z.convertTo(Z, CV_32F);
    return Z;
}

cv::Mat loadThumbnails(const std::string& path)
{
    std::vector<fs::path> files;
    for (fs::directory_iterator it{ fs::path(path) }, end; it != end; ++it) {
        if (it->path().extension() == ".png" || it->path().extension() == ".jpg") {
            files.push_back(it->path());
        }
    }

    auto frameCount = files.size();
    auto w = 16;
    auto h = 16;
    auto n = frameCount * (w*h);

    std::cout << "Allocating memory... " << n * 3 * sizeof(float) / 1024 / 1024 << " MB\n";
    cv::Mat Z(n, 3, CV_32F);
    auto Zframe = 0;

    std::cout << "Reading frames...\n";
    auto frameIdx = 0;
    for (auto f : files) {
        std::cout << "  " << frameIdx << "/" << frameCount << "\n";

        auto frame = cv::imread(f.string());
        cv::resize(frame, frame, cv::Size(w, h));

        assert(frame.isContinuous());

        for (int i = 0; i < frame.size().height; ++i) {
            const cv::Vec3b* row = frame.ptr<cv::Vec3b>(i);
            for (int j = 0; j < frame.size().width; ++j) {
                cv::Vec3b v;
                convert(row[j], v, CV_BGR2Lab);
                Z.at<float>(Zframe*w*h + i*w + j, 0) = v[0];
                Z.at<float>(Zframe*w*h + i*w + j, 1) = v[1];
                Z.at<float>(Zframe*w*h + i*w + j, 2) = v[2];
            }
        }
        ++frameIdx;
        ++Zframe;
    }

    return Z;
}

cv::Mat loadVideo(const std::string& filename)
{
    cv::VideoCapture cap(filename);
    if (!cap.isOpened()) {
        std::cout << "Could not open " << filename << "\n";
        return cv::Mat();
    }

    int frameCount;
    frameCount = (int)cap.get(CV_CAP_PROP_FRAME_COUNT);
    int frameStep = 30;

    auto w = 16;
    auto h = 16;
    auto n = (frameCount/frameStep+1) * (w*h);

    std::cout << "Allocating memory... " << n * 3 * sizeof(float) / 1024 / 1024 << " MB\n";
        
    cv::Mat Z(n, 3, CV_32F);
    auto Zframe = 0;

    std::cout << "Reading frames...\n";
    
    auto frameIdx = 0;
    for (;;) {
        cv::Mat frame;
        cap.set(CV_CAP_PROP_POS_FRAMES, (double)frameIdx);
        if (!cap.read(frame)) {
            break;
        }
        if (frameIdx > frameCount) {
            break;
        }

        std::cout << "  " << frameIdx / frameStep << "/" << frameCount / frameStep << "\n";

        cv::resize(frame, frame, cv::Size(w, h));

        assert(frame.isContinuous());

        for (int i = 0; i < frame.size().height; ++i) {
            const cv::Vec3b* row = frame.ptr<cv::Vec3b>(i);
            for (int j = 0; j < frame.size().width; ++j) {
                cv::Vec3b v;
                convert(row[j], v, CV_BGR2Lab);
                Z.at<float>(Zframe*w*h + i*w + j, 0) = v[0];
                Z.at<float>(Zframe*w*h + i*w + j, 1) = v[1];
                Z.at<float>(Zframe*w*h + i*w + j, 2) = v[2];
            }
        }

        frameIdx += frameStep;
        ++Zframe;
    }

    return Z;
}

int main(int argc, char* argv[])
{
    if (argc < 6) {
        std::cout << "Usage: " << argv[0] << " in out-image out-csv threshold [sort-mode]\n";
        std::cout << "  sort-mode: sort-by-color, sort-by-count\n";
        return 0;
    }

    cv::Mat Z;
    std::string f = argv[1];
    if (fs::is_directory(f)) {
        Z = loadThumbnails(f);
    } else if (fs::path(f).extension() == ".png" || fs::path(f).extension() == ".jpg") {
        Z = loadImage(f);
    } else {
        Z = loadVideo(f);
    }

    std::cout << "Clustering...\n";
    std::vector<Cluster> centers;
    if (std::string(argv[4]).find(".") == std::string::npos) {
        cluster(Z, (int)atoi(argv[4]), centers);
    } else {
        cluster(Z, (float)atof(argv[4]), centers);
    }

    std::string sort_mode = argc >= 6 ? argv[5] : "sort-by-count";
    if (sort_mode == "sort-by-color") {
        sortCentersByColor(centers);
    } else {
        sortCentersByCount(centers);
    }
    exportCenters(centers, argv[2], argv[3]);
}
