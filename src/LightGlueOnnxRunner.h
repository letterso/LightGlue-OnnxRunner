/**
 * @file main.cpp
 * @author letterso
 * @brief modified form OroChippw/LightGlue-OnnxRunner
 * @version 0.5
 * @date 2023-11-20
 *
 * @copyright Copyright (c) 2023
 *
 */

#pragma once
#pragma warning(disable : 4819)

#ifndef LIGHTGLUE_ONNX_RUNNER_H
#define LIGHTGLUE_ONNX_RUNNER_H

#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>
// #include <cuda_provider_factory.h>  // 若在GPU环境下运行可以使用cuda进行加速

#include "transform.h"
#include "BaseOnnxRunner.h"
#include "Configuration.h"

class LightGlueOnnxRunner : public BaseFeatureMatchOnnxRunner
{
private:
    const unsigned int num_threads;

    Ort::Env env;
    Ort::SessionOptions session_options;
    std::unique_ptr<Ort::Session> session;
    Ort::AllocatorWithDefaultOptions allocator;

    std::vector<char *> InputNodeNames;
    std::vector<std::vector<int64_t>> InputNodeShapes;

    std::vector<char *> OutputNodeNames;
    std::vector<std::vector<int64_t>> OutputNodeShapes;

    float matchThresh = 0.0f;
    long long timer = 0.0f;

    std::vector<float> scales = {1.0f, 1.0f};

    std::vector<Ort::Value> output_tensors;
    std::pair<std::vector<cv::Point2f>, std::vector<cv::Point2f>> keypoints_result;

private:
    cv::Mat PreProcess(Configuration cfg, const cv::Mat &srcImage, float &scale);
    int Inference(Configuration cfg, const cv::Mat &src, const cv::Mat &dest);
    int PostProcess(Configuration cfg);

public:
    explicit LightGlueOnnxRunner(unsigned int num_threads = 1);
    ~LightGlueOnnxRunner();

    float GetMatchThresh();
    void SetMatchThresh(float thresh);
    double GetTimer(std::string name);

    std::pair<std::vector<cv::Point2f>, std::vector<cv::Point2f>> GetKeypointsResult();

    int InitOrtEnv(Configuration cfg);

    std::pair<std::vector<cv::Point2f>, std::vector<cv::Point2f>> InferenceImage(Configuration cfg,
                                                                                 const cv::Mat &srcImage, const cv::Mat &destImage);
};

#endif // LIGHTGLUE_ONNX_RUNNER_H