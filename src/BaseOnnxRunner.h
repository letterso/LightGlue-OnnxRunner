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

#ifndef BASEONNXRUNNER_H
#define BASEONNXRUNNER_H

#include <iostream>
#include <algorithm>
#include <chrono>
#include <string.h>
#include <stdlib.h>
#include <thread>
#include <memory>

#include "Configuration.h"

class BaseFeatureMatchOnnxRunner
{
public:
    virtual int InitOrtEnv(Configuration cfg)
    {
        return EXIT_SUCCESS;
    }

    virtual float GetMatchThresh()
    {
        return 0.0f;
    }

    virtual void SetMatchThresh(float thresh) {}

    virtual double GetTimer(std::string name = "matcher")
    {
        return 0.0f;
    }

    virtual std::pair<std::vector<cv::Point2f>, std::vector<cv::Point2f>> InferenceImage(Configuration cfg,
                                                                                         const cv::Mat &srcImage, const cv::Mat &destImage)
    {
        return std::pair<std::vector<cv::Point2f>, std::vector<cv::Point2f>>();
    };

    virtual std::pair<std::vector<cv::Point2f>, std::vector<cv::Point2f>> GetKeypointsResult()
    {
        return std::pair<std::vector<cv::Point2f>, std::vector<cv::Point2f>>();
    };
};

#endif // BASEONNXRUNNER_H