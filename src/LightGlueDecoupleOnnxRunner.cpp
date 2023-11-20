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

#include "LightGlueDecoupleOnnxRunner.h"

int LightGlueDecoupleOnnxRunner::InitOrtEnv(Configuration cfg)
{
    std::cout << "< - * -------- INITIAL ONNXRUNTIME ENV START -------- * ->" << std::endl;
    try
    {
        env0 = Ort::Env(ORT_LOGGING_LEVEL_WARNING, "LightGlueDecoupleOnnxRunner Extractor");
        env1 = Ort::Env(ORT_LOGGING_LEVEL_WARNING, "LightGlueDecoupleOnnxRunner Matcher");

        session_options0 = Ort::SessionOptions();
        session_options0.SetInterOpNumThreads(std::thread::hardware_concurrency());
        session_options0.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

        session_options1 = Ort::SessionOptions();
        session_options1.SetInterOpNumThreads(std::thread::hardware_concurrency());
        session_options1.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

        if (cfg.device == "cuda")
        {
            std::cout << "[INFO] OrtSessionOptions Append CUDAExecutionProvider" << std::endl;
            OrtCUDAProviderOptions cuda_options{};

            cuda_options.device_id = 0;
            cuda_options.cudnn_conv_algo_search = OrtCudnnConvAlgoSearchDefault;
            cuda_options.gpu_mem_limit = 0;
            cuda_options.arena_extend_strategy = 1;     // 设置GPU内存管理中的Arena扩展策略
            cuda_options.do_copy_in_default_stream = 1; // 是否在默认CUDA流中执行数据复制
            cuda_options.has_user_compute_stream = 0;
            cuda_options.default_memory_arena_cfg = nullptr;

            session_options0.AppendExecutionProvider_CUDA(cuda_options);
            session_options0.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
            session_options1.AppendExecutionProvider_CUDA(cuda_options);
            session_options1.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
        }

        ExtractorSession = std::make_unique<Ort::Session>(env0, cfg.extractorPath.c_str(), session_options0);
        MatcherSession = std::make_unique<Ort::Session>(env1, cfg.lightgluePath.c_str(), session_options1);

        // Initial Extractor
        size_t numInputNodes = ExtractorSession->GetInputCount();
        ExtractorInputNodeNames.reserve(numInputNodes);
        for (size_t i = 0; i < numInputNodes; i++)
        {
            ExtractorInputNodeNames.emplace_back(strdup(ExtractorSession->GetInputNameAllocated(i, allocator).get()));
            ExtractorInputNodeShapes.emplace_back(ExtractorSession->GetInputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape());
        }

        size_t numOutputNodes = ExtractorSession->GetOutputCount();
        ExtractorOutputNodeNames.reserve(numOutputNodes);
        for (size_t i = 0; i < numOutputNodes; i++)
        {
            ExtractorOutputNodeNames.emplace_back(strdup(ExtractorSession->GetOutputNameAllocated(i, allocator).get()));
            ExtractorOutputNodeShapes.emplace_back(ExtractorSession->GetOutputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape());
        }

        numInputNodes = 0;
        numOutputNodes = 0;

        // Initial Matcher
        numInputNodes = MatcherSession->GetInputCount();
        ExtractorInputNodeNames.reserve(numInputNodes);
        for (size_t i = 0; i < numInputNodes; i++)
        {
            MatcherInputNodeNames.emplace_back(strdup(MatcherSession->GetInputNameAllocated(i, allocator).get()));
            MatcherInputNodeShapes.emplace_back(MatcherSession->GetInputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape());
        }

        numOutputNodes = MatcherSession->GetOutputCount();
        ExtractorOutputNodeNames.reserve(numOutputNodes);
        for (size_t i = 0; i < numOutputNodes; i++)
        {
            MatcherOutputNodeNames.emplace_back(strdup(MatcherSession->GetOutputNameAllocated(i, allocator).get()));
            MatcherOutputNodeShapes.emplace_back(MatcherSession->GetOutputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape());
        }

        std::cout << "[INFO] ONNXRuntime environment created successfully." << std::endl;
    }
    catch (const std::exception &ex)
    {
        std::cerr << "[ERROR] ONNXRuntime environment created failed : " << ex.what() << '\n';
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}

cv::Mat LightGlueDecoupleOnnxRunner::Extractor_PreProcess(Configuration cfg, const cv::Mat &Image, float &scale)
{
    float temp_scale = scale;
    cv::Mat tempImage = Image.clone();
    std::cout << "[INFO] Image info :  width : " << Image.cols << " height :  " << Image.rows << std::endl;

    std::string fn = "max";
    std::string interp = "area";
    cv::Mat resize_img = ResizeImage(tempImage, cfg.image_size, scale, fn, interp);
    cv::Mat resultImage = NormalizeImage(resize_img);
    if (cfg.extractorType == "superpoint")
    {
        std::cout << "[INFO] ExtractorType Superpoint turn RGB to Grayscale" << std::endl;
        resultImage = RGB2Grayscale(resultImage);
    }
    std::cout << "[INFO] Scale from " << temp_scale << " to " << scale << std::endl;

    return resultImage;
}

int LightGlueDecoupleOnnxRunner::Extractor_Inference(Configuration cfg, const cv::Mat &image)
{
    std::cout << "< - * -------- Extractor Inference START -------- * ->" << std::endl;
    try
    {
        // Dynamic InputNodeShapes is [1,3,-1,-1] or [1,1,-1,-1]
        std::cout << "[INFO] Image Size : " << image.size() << " Channels : " << image.channels() << std::endl;

        // Build src input node shape and destImage input node shape
        int srcInputTensorSize, destInputTensorSize;
        if (cfg.extractorType == "superpoint")
        {
            ExtractorInputNodeShapes[0] = {1, 1, image.size().height, image.size().width};
        }
        else if (cfg.extractorType == "disk")
        {
            ExtractorInputNodeShapes[0] = {1, 3, image.size().height, image.size().width};
        }
        srcInputTensorSize = ExtractorInputNodeShapes[0][0] * ExtractorInputNodeShapes[0][1] * ExtractorInputNodeShapes[0][2] * ExtractorInputNodeShapes[0][3];

        std::vector<float> srcInputTensorValues(srcInputTensorSize);

        if (cfg.extractorType == "superpoint")
        {
            srcInputTensorValues.assign(image.begin<float>(), image.end<float>());
        }
        else
        {
            int height = image.rows;
            int width = image.cols;
            for (int y = 0; y < height; y++)
            {
                for (int x = 0; x < width; x++)
                {
                    cv::Vec3f pixel = image.at<cv::Vec3f>(y, x); // RGB
                    srcInputTensorValues[y * width + x] = pixel[2];
                    srcInputTensorValues[height * width + y * width + x] = pixel[1];
                    srcInputTensorValues[2 * height * width + y * width + x] = pixel[0];
                }
            }
        }

        auto memory_info_handler = Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtDeviceAllocator,
                                                              OrtMemType::OrtMemTypeCPU);

        std::vector<Ort::Value> input_tensors;
        input_tensors.push_back(Ort::Value::CreateTensor<float>(
            memory_info_handler, srcInputTensorValues.data(), srcInputTensorValues.size(),
            ExtractorInputNodeShapes[0].data(), ExtractorInputNodeShapes[0].size()));

        auto time_start = std::chrono::high_resolution_clock::now();

        auto output_tensor = ExtractorSession->Run(Ort::RunOptions{nullptr}, ExtractorInputNodeNames.data(), input_tensors.data(),
                                                   input_tensors.size(), ExtractorOutputNodeNames.data(), ExtractorOutputNodeNames.size());

        auto time_end = std::chrono::high_resolution_clock::now();
        auto diff = std::chrono::duration_cast<std::chrono::milliseconds>(time_end - time_start).count();
        extractor_timer += diff;

        for (auto &tensor : output_tensor)
        {
            if (!tensor.IsTensor() || !tensor.HasValue())
            {
                std::cerr << "[ERROR] Inference output tensor is not a tensor or don't have value" << std::endl;
            }
        }

        extractor_outputtensors.emplace_back(std::move(output_tensor));

        std::cout << "[INFO] LightGlueDecoupleOnnxRunner Extractor inference finish ..." << std::endl;
        std::cout << "[INFO] Extractor inference cost time : " << diff << "ms" << std::endl;
    }
    catch (const std::exception &ex)
    {
        std::cerr << "[ERROR] LightGlueDecoupleOnnxRunner Extractor inference failed : " << ex.what() << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}

std::pair<std::vector<cv::Point2f>, float *> LightGlueDecoupleOnnxRunner::Extractor_PostProcess(Configuration cfg, std::vector<Ort::Value> tensor)
{
    std::pair<std::vector<cv::Point2f>, float *> extractor_result;
    try
    {
        std::vector<int64_t> kpts_Shape = tensor[0].GetTensorTypeAndShapeInfo().GetShape();
        int64_t *kpts = (int64_t *)tensor[0].GetTensorMutableData<void>();
        // for (int i = 0 ; i < kpts_Shape[1] ; i++)
        // {
        //     std::cout << kpts[i] << " ";
        // }
        printf("[RESULT INFO] kpts Shape : (%ld , %ld , %ld)\n", kpts_Shape[0], kpts_Shape[1], kpts_Shape[2]);

        std::vector<int64_t> score_Shape = tensor[1].GetTensorTypeAndShapeInfo().GetShape();
        float *scores = (float *)tensor[1].GetTensorMutableData<void>();

        std::vector<int64_t> descriptors_Shape = tensor[2].GetTensorTypeAndShapeInfo().GetShape();
        float *desc = (float *)tensor[2].GetTensorMutableData<void>();
        printf("[RESULT INFO] desc Shape : (%ld , %ld , %ld)\n", descriptors_Shape[0], descriptors_Shape[1], descriptors_Shape[2]);

        // Process kpts and descriptors
        std::vector<cv::Point2f> kpts_f;
        for (int i = 0; i < kpts_Shape[1] * 2; i += 2)
        {
            kpts_f.emplace_back(cv::Point2f(kpts[i], kpts[i + 1]));
        }

        extractor_result.first = kpts_f;
        extractor_result.second = desc;

        std::cout << "[INFO] Extractor postprocessing operation completed successfully" << std::endl;
    }
    catch (const std::exception &ex)
    {
        std::cerr << "[ERROR] Extractor postprocess failed : " << ex.what() << std::endl;
    }

    return extractor_result;
}

std::vector<cv::Point2f> LightGlueDecoupleOnnxRunner::Matcher_PreProcess(std::vector<cv::Point2f> kpts, int h, int w)
{
    return NormalizeKeypoints(kpts, h, w);
}

int LightGlueDecoupleOnnxRunner::Matcher_Inference(Configuration cfg, std::vector<cv::Point2f> kpts0,
                                                   std::vector<cv::Point2f> kpts1, float *desc0, float *desc1)
{
    std::cout << "< - * -------- Matcher Inference START -------- * ->" << std::endl;
    try
    {
        MatcherInputNodeShapes[0] = {1, static_cast<int>(kpts0.size()), 2};
        MatcherInputNodeShapes[1] = {1, static_cast<int>(kpts1.size()), 2};
        if (cfg.extractorType == "superpoint")
        {
            MatcherInputNodeShapes[2] = {1, static_cast<int>(kpts0.size()), 256};
            MatcherInputNodeShapes[3] = {1, static_cast<int>(kpts1.size()), 256};
        }
        else
        {
            MatcherInputNodeShapes[2] = {1, static_cast<int>(kpts0.size()), 128};
            MatcherInputNodeShapes[3] = {1, static_cast<int>(kpts1.size()), 128};
        }

        auto memory_info_handler = Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtDeviceAllocator, OrtMemType::OrtMemTypeCPU);

        float *kpts0_data = new float[kpts0.size() * 2];
        float *kpts1_data = new float[kpts1.size() * 2];

        for (size_t i = 0; i < kpts0.size(); ++i)
        {
            kpts0_data[i * 2] = kpts0[i].x;
            kpts0_data[i * 2 + 1] = kpts0[i].y;
        }
        for (size_t i = 0; i < kpts1.size(); ++i)
        {
            kpts1_data[i * 2] = kpts1[i].x;
            kpts1_data[i * 2 + 1] = kpts1[i].y;
        }

        std::vector<Ort::Value> input_tensors;
        input_tensors.push_back(Ort::Value::CreateTensor<float>(
            memory_info_handler, kpts0_data, kpts0.size() * 2 * sizeof(float),
            MatcherInputNodeShapes[0].data(), MatcherInputNodeShapes[0].size()));
        input_tensors.push_back(Ort::Value::CreateTensor<float>(
            memory_info_handler, kpts1_data, kpts1.size() * 2 * sizeof(float),
            MatcherInputNodeShapes[1].data(), MatcherInputNodeShapes[1].size()));
        input_tensors.push_back(Ort::Value::CreateTensor<float>(
            memory_info_handler, desc0, kpts0.size() * 256 * sizeof(float),
            MatcherInputNodeShapes[2].data(), MatcherInputNodeShapes[2].size()));
        input_tensors.push_back(Ort::Value::CreateTensor<float>(
            memory_info_handler, desc1, kpts1.size() * 256 * sizeof(float),
            MatcherInputNodeShapes[3].data(), MatcherInputNodeShapes[3].size()));

        auto time_start = std::chrono::high_resolution_clock::now();

        auto output_tensor = MatcherSession->Run(Ort::RunOptions{nullptr}, MatcherInputNodeNames.data(), input_tensors.data(),
                                                 input_tensors.size(), MatcherOutputNodeNames.data(), MatcherOutputNodeNames.size());

        auto time_end = std::chrono::high_resolution_clock::now();
        auto diff = std::chrono::duration_cast<std::chrono::milliseconds>(time_end - time_start).count();
        matcher_timer += diff;

        for (auto &tensor : output_tensor)
        {
            if (!tensor.IsTensor() || !tensor.HasValue())
            {
                std::cerr << "[ERROR] Inference output tensor is not a tensor or don't have value" << std::endl;
            }
        }
        matcher_outputtensors = std::move(output_tensor);

        std::cout << "[INFO] LightGlueDecoupleOnnxRunner Matcher inference finish ..." << std::endl;
        std::cout << "[INFO] Matcher inference cost time : " << diff << "ms" << std::endl;
    }
    catch (const std::exception &ex)
    {
        std::cerr << "[ERROR] LightGlueDecoupleOnnxRunner Matcher inference failed : " << ex.what() << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}

int LightGlueDecoupleOnnxRunner::Matcher_PostProcess(Configuration cfg, std::vector<cv::Point2f> kpts0, std::vector<cv::Point2f> kpts1)
{
    try
    {
        // load date from tensor
        std::vector<int64_t> matches_Shape = matcher_outputtensors[0].GetTensorTypeAndShapeInfo().GetShape();
        int64_t *matches = (int64_t *)matcher_outputtensors[0].GetTensorMutableData<void>();
        printf("[RESULT INFO] matches0 Shape : (%ld , %ld)\n", matches_Shape[0], matches_Shape[1]);

        std::vector<int64_t> mscore_Shape = matcher_outputtensors[1].GetTensorTypeAndShapeInfo().GetShape();
        float *mscores = (float *)matcher_outputtensors[1].GetTensorMutableData<void>();

        // Process kpts0 and kpts1
        std::vector<cv::Point2f> kpts0_f, kpts1_f;
        kpts0_f.reserve(kpts0.size());
        kpts1_f.reserve(kpts1.size());
        for (int i = 0; i < kpts0.size(); i++)
        {
            kpts0_f.emplace_back(cv::Point2f(
                (kpts0[i].x + 0.5) / scales[0] - 0.5, (kpts0[i].y + 0.5) / scales[0] - 0.5));
        }
        for (int i = 0; i < kpts1.size(); i++)
        {
            kpts1_f.emplace_back(cv::Point2f(
                (kpts1[i].x + 0.5) / scales[1] - 0.5, (kpts1[i].y + 0.5) / scales[1] - 0.5));
        }

        // get the good match
        std::vector<cv::Point2f> m_kpts0, m_kpts1;
        m_kpts0.reserve(matches_Shape[0]);
        m_kpts1.reserve(matches_Shape[0]);
        for (int i = 0; i < matches_Shape[0]; i++)
        {
            if (mscores[i] > this->matchThresh)
            {
                m_kpts0.emplace_back(kpts0_f[matches[i * 2]]);
                m_kpts1.emplace_back(kpts1_f[matches[i * 2 + 1]]);
            }
        }
        std::cout << "[RESULT INFO] matches Size : " << m_kpts1.size() << std::endl;

        keypoints_result.first = m_kpts0;
        keypoints_result.second = m_kpts1;

        std::cout << "[INFO] Postprocessing operation completed successfully" << std::endl;
    }
    catch (const std::exception &ex)
    {
        std::cerr << "[ERROR] PostProcess failed : " << ex.what() << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}

std::pair<std::vector<cv::Point2f>, float *> LightGlueDecoupleOnnxRunner::Extractor(Configuration cfg, const cv::Mat &srcImage, float scale)
{
    std::cout << "< - * -------- Extractor START -------- * ->" << std::endl;
    if (srcImage.empty())
    {
        throw "[ERROR] ImageEmptyError ";
    }
    cv::Mat srcImage_copy = cv::Mat(srcImage);

    // Extract Keypoints
    std::cout << "[INFO] => PreProcess srcImage" << std::endl;
    cv::Mat src = Extractor_PreProcess(cfg, srcImage_copy, scale);
    Extractor_Inference(cfg, src);

    std::pair<std::vector<cv::Point2f>, float *> src_extract = Extractor_PostProcess(cfg, std::move(extractor_outputtensors[0]));
    std::vector<cv::Point2f> normal_kpts = Matcher_PreProcess(src_extract.first, src.rows, src.cols);
    src_extract.first = normal_kpts;
    return src_extract;
}

std::pair<std::vector<cv::Point2f>, std::vector<cv::Point2f>> LightGlueDecoupleOnnxRunner::Matcher(Configuration cfg, std::vector<cv::Point2f> kpts0,
                                                                                                   std::vector<cv::Point2f> kpts1, float *desc0, float *desc1)
{
    Matcher_Inference(cfg, kpts0, kpts1, desc0, desc1);
    Matcher_PostProcess(cfg, kpts0, kpts1);
    return GetKeypointsResult();
}

std::pair<std::vector<cv::Point2f>, std::vector<cv::Point2f>> LightGlueDecoupleOnnxRunner::InferenceImage(Configuration cfg,
                                                                                                          const cv::Mat &srcImage, const cv::Mat &destImage)
{
    std::cout << "< - * -------- INFERENCEIMAGE START -------- * ->" << std::endl;

    if (srcImage.empty() || destImage.empty())
    {
        throw "[ERROR] ImageEmptyError ";
    }
    cv::Mat srcImage_copy = cv::Mat(srcImage);
    cv::Mat destImage_copy = cv::Mat(destImage);

    // Extract Keypoints
    std::cout << "[INFO] => PreProcess srcImage" << std::endl;
    cv::Mat src = Extractor_PreProcess(cfg, srcImage_copy, scales[0]);
    std::cout << "[INFO] => PreProcess destImage" << std::endl;
    cv::Mat dest = Extractor_PreProcess(cfg, destImage_copy, scales[1]);

    Extractor_Inference(cfg, src);
    Extractor_Inference(cfg, dest);

    auto src_extract = Extractor_PostProcess(cfg, std::move(extractor_outputtensors[0]));
    auto dest_extract = Extractor_PostProcess(cfg, std::move(extractor_outputtensors[1]));

    // Build Matches
    auto normal_kpts0 = Matcher_PreProcess(src_extract.first, src.rows, src.cols);
    auto normal_kpts1 = Matcher_PreProcess(dest_extract.first, dest.rows, dest.cols);

    Matcher_Inference(cfg, normal_kpts0, normal_kpts1, src_extract.second, dest_extract.second);

    Matcher_PostProcess(cfg, src_extract.first, dest_extract.first);

    extractor_outputtensors.clear();
    matcher_outputtensors.clear();

    return GetKeypointsResult();
}

float LightGlueDecoupleOnnxRunner::GetMatchThresh()
{
    return this->matchThresh;
}

void LightGlueDecoupleOnnxRunner::SetMatchThresh(float thresh)
{
    this->matchThresh = thresh;
}

double LightGlueDecoupleOnnxRunner::GetTimer(std::string name)
{
    if (name == "extractor")
    {
        return this->extractor_timer;
    }
    else
    {
        return this->matcher_timer;
    }
}

std::pair<std::vector<cv::Point2f>, std::vector<cv::Point2f>> LightGlueDecoupleOnnxRunner::GetKeypointsResult()
{
    return this->keypoints_result;
}

LightGlueDecoupleOnnxRunner::LightGlueDecoupleOnnxRunner(unsigned int threads) : num_threads(threads)
{
}

LightGlueDecoupleOnnxRunner::~LightGlueDecoupleOnnxRunner()
{
}
