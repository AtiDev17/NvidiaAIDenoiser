#include <optix.h>
#include <optix_stubs.h>
#include <optix_function_table_definition.h>
#include <cuda_runtime.h>
#include <iostream>
#include <iomanip>
#include <OpenImageIO/imageio.h>
#include <OpenImageIO/imagebuf.h>
#include <stdio.h>
#include <exception>
#include <algorithm>

#include <time.h>
#include <thread>
#include <chrono>
#include <vector>
#include <string>
#include <unordered_map>
#include <memory>

#include "CudaCheck.h"
#include "CudaBuffer.h"

#define DENOISER_MAJOR_VERSION 3
#define DENOISER_MINOR_VERSION 1

// Image information structure
struct ImageInfo
{
    std::string filename;
    std::string output_filename;
    std::unique_ptr<OIIO::ImageBuf> data;
};

class DenoiserApp
{
public:
    DenoiserApp() 
    {
        m_app_start_time = std::chrono::high_resolution_clock::now();
    }

    ~DenoiserApp()
    {
        if (m_optix_denoiser) optixDenoiserDestroy(m_optix_denoiser);
        if (m_optix_context) optixDeviceContextDestroy(m_optix_context);
        if (m_cuda_stream) cudaStreamDestroy(m_cuda_stream);
    }

    void run(int argc, char *argv[]);

private:
    void printParams();
    bool discoverDevices();
    std::string getTime();
    
    template<typename... Args>
    void logInfo(const char *c, Args... args);
    template<typename... Args>
    void logError(const char *c, Args... args);

    void loadImages(int argc, char *argv[]);
    void setupDenoiser();
    void executeDenoiser();
    void saveImages();
    void uploadLayer(const ImageInfo& img, CudaBuffer& buffer, int target_channels);

    // Helper to format string
    std::string format(const char* fmt, ...);

    int m_verbosity = 2;
    std::chrono::high_resolution_clock::time_point m_app_start_time;
    std::vector<cudaDeviceProp> m_device_props;
    unsigned int m_selected_device_id = 0;
    
    // OptiX / CUDA
    OptixDeviceContext m_optix_context = nullptr;
    OptixDenoiser m_optix_denoiser = nullptr;
    cudaStream_t m_cuda_stream = nullptr;
    
    CudaBuffer m_denoiser_state;
    CudaBuffer m_denoiser_scratch;
    CudaBuffer m_hdr_intensity;

    OptixDenoiserSizes m_denoiser_sizes = {};
    OptixDenoiserParams m_denoiser_params = {};

    // Images
    ImageInfo m_beauty;
    ImageInfo m_prev_denoised;
    ImageInfo m_albedo;
    ImageInfo m_normal;
    ImageInfo m_motion_vectors;
    std::unordered_map<int, ImageInfo> m_aovs;

    bool m_b_loaded = false;
    bool m_n_loaded = false;
    bool m_a_loaded = false;
    bool m_mv_loaded = false;
    bool m_pi_loaded = false;

    // Config
    float m_blend = 0.f;
    unsigned int m_hdr = 1;
    unsigned int m_num_runs = 1;
    std::string m_out_suffix;
    bool m_denoise_aovs = false;

    // Layer management
    std::vector<OptixDenoiserLayer> m_layers;
    std::vector<CudaBuffer> m_layer_buffers; // Holds input/output/prev buffers
    OptixDenoiserGuideLayer m_guide_layer = {};
    CudaBuffer m_albedo_buffer;
    CudaBuffer m_normal_buffer;
    CudaBuffer m_flow_buffer;
    CudaBuffer m_internal_guide_layer_buffer;
    CudaBuffer m_internal_guide_layer_prev_buffer;

    int m_width = 0;
    int m_height = 0;

    // Tiling config
    unsigned int m_tile_width = 2048; // Max tile width (excluding overlap)
    unsigned int m_tile_height = 2048; // Max tile height (excluding overlap)
    unsigned int m_overlap = 64;       // Overlap size

    // Tile buffers (reused)
    std::vector<CudaBuffer> m_tile_layer_buffers;
    CudaBuffer m_tile_albedo_buffer;
    CudaBuffer m_tile_normal_buffer;
    CudaBuffer m_tile_flow_buffer;
};

std::string DenoiserApp::getTime()
{
    auto now = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> time_span = now - m_app_start_time;
    double milliseconds = time_span.count();
    int seconds = (int)(milliseconds / 1000.0);
    int minutes = seconds / 60;
    milliseconds -= seconds * 1000.0;
    seconds %= 60;
    char s[32];
    snprintf(s, sizeof(s), "%02d:%02d:%03d", minutes, seconds, (int)milliseconds);
    return std::string(s);
}

template<typename... Args>
void DenoiserApp::logInfo(const char *c, Args... args)
{
    if (m_verbosity < 2) return;
    char buffer[1024];
    snprintf(buffer, sizeof(buffer), c, args...);
    std::cout << getTime() << "         | " << buffer << std::endl;
}

template<typename... Args>
void DenoiserApp::logError(const char *c, Args... args)
{
    char buffer[1024];
    snprintf(buffer, sizeof(buffer), c, args...);
    std::cerr << getTime() << " ERROR   | " << buffer << std::endl;
}

void DenoiserApp::printParams()
{
    std::cout << "Command line parameters" << std::endl;
    std::cout << "-v [int]         : log verbosity level 0:disabled 1:simple 2:full (default 2)" << std::endl;
    std::cout << "-i [string]      : path to input image" << std::endl;
    std::cout << "-pi [string]     : path previous denoised result (optional, required for temporal denoising)" << std::endl;
    std::cout << "-aov%d [string]  : path to additional input AOV image to denoise" << std::endl;
    std::cout << "-oaov%d [string] : path to additional AOV output image to denoise" << std::endl;
    std::cout << "-o [string]      : path to output image" << std::endl;
    std::cout << "-os [string]     : output suffix appended to input filename to create output image filename" << std::endl;
    std::cout << "-a [string]      : path to input albedo AOV (optional)" << std::endl;
    std::cout << "-n [string]      : path to input normal AOV (optional, requires albedo AOV)" << std::endl;
    std::cout << "-mv [string]     : path to motion vector AOV (optional, required for temporal denoising)" << std::endl;
    std::cout << "-b [float]       : blend amount (default 0)" << std::endl;
    std::cout << "-hdr [int]       : Use HDR training data (default 1)" << std::endl;
    std::cout << "-gpu [int]       : Select which GPU to use for denoising (default 0)" << std::endl;
    std::cout << "-repeat [int]    : Execute the denoiser N times. Useful for profiling." << std::endl;
}

bool DenoiserApp::discoverDevices()
{
    int device_count = 0;
    cudaGetDeviceCount(&device_count);
    if (device_count == 0) {
        logError("No Nvidia GPUs found");
        return false;
    }
    
    logInfo("Found %d CUDA device(s)", device_count);
    for (int i=0; i < device_count; i++)
    {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        logInfo("GPU %d: %s (compute %d.%d) with %dMB memory", i, prop.name, prop.major, prop.minor, (int)(prop.totalGlobalMem / 1024 / 1024));
        m_device_props.push_back(prop);
    }
    return true;
}

void DenoiserApp::loadImages(int argc, char *argv[])
{
    // Parse arguments
    for (int i=1; i<argc; i++)
    {
        std::string arg(argv[i]);
        if (arg == "-i" && i+1 < argc) {
            m_beauty.filename = argv[++i];
            m_beauty.data = std::make_unique<OIIO::ImageBuf>(m_beauty.filename);
            if (m_beauty.data->init_spec(m_beauty.filename, 0, 0)) {
                m_b_loaded = true;
                logInfo("Input image loaded: %s", m_beauty.filename.c_str());
            } else {
                logError("Failed to load input image: %s", m_beauty.data->geterror().c_str());
            }
        }
        else if (arg == "-pi" && i+1 < argc) {
            m_prev_denoised.filename = argv[++i];
            m_prev_denoised.data = std::make_unique<OIIO::ImageBuf>(m_prev_denoised.filename);
            if (m_prev_denoised.data->init_spec(m_prev_denoised.filename, 0, 0)) {
                m_pi_loaded = true;
                logInfo("Prev denoised frame loaded: %s", m_prev_denoised.filename.c_str());
            }
        }
        else if (arg == "-n" && i+1 < argc) {
            m_normal.filename = argv[++i];
            m_normal.data = std::make_unique<OIIO::ImageBuf>(m_normal.filename);
            if (m_normal.data->init_spec(m_normal.filename, 0, 0)) {
                m_n_loaded = true;
                logInfo("Normal loaded: %s", m_normal.filename.c_str());
            }
        }
        else if (arg == "-a" && i+1 < argc) {
            m_albedo.filename = argv[++i];
            m_albedo.data = std::make_unique<OIIO::ImageBuf>(m_albedo.filename);
            if (m_albedo.data->init_spec(m_albedo.filename, 0, 0)) {
                m_a_loaded = true;
                logInfo("Albedo loaded: %s", m_albedo.filename.c_str());
            }
        }
        else if (arg == "-mv" && i+1 < argc) {
            m_motion_vectors.filename = argv[++i];
            m_motion_vectors.data = std::make_unique<OIIO::ImageBuf>(m_motion_vectors.filename);
            if (m_motion_vectors.data->init_spec(m_motion_vectors.filename, 0, 0)) {
                m_mv_loaded = true;
                logInfo("Motion vectors loaded: %s", m_motion_vectors.filename.c_str());
            }
        }
        else if (arg == "-o" && i+1 < argc) {
            m_beauty.output_filename = argv[++i];
        }
        else if (arg == "-os" && i+1 < argc) {
            m_out_suffix = argv[++i];
        }
        else if (arg == "-b" && i+1 < argc) {
            m_blend = std::stof(argv[++i]);
        }
        else if (arg == "-hdr" && i+1 < argc) {
            m_hdr = std::stoi(argv[++i]);
        }
        else if (arg == "-gpu" && i+1 < argc) {
            m_selected_device_id = std::stoi(argv[++i]);
        }
        else if (arg == "-repeat" && i+1 < argc) {
            m_num_runs = std::max(std::stoi(argv[++i]), 1);
        }
        else if (arg.find("-aov") == 0 && arg.size() > 4) {
             int id = std::stoi(arg.substr(4));
             if (i+1 < argc) {
                 std::string fname = argv[++i];
                 if (m_aovs.find(id) == m_aovs.end()) m_aovs[id] = ImageInfo();
                 m_aovs[id].filename = fname;
                 m_aovs[id].data = std::make_unique<OIIO::ImageBuf>(fname);
                 if (m_aovs[id].data->init_spec(fname, 0, 0)) {
                     m_denoise_aovs = true;
                     logInfo("AOV %d loaded: %s", id, fname.c_str());
                 }
             }
        }
        else if (arg.find("-oaov") == 0 && arg.size() > 5) {
             int id = std::stoi(arg.substr(5));
             if (i+1 < argc) {
                 if (m_aovs.find(id) == m_aovs.end()) m_aovs[id] = ImageInfo();
                 m_aovs[id].output_filename = argv[++i];
             }
        }
    }
}

void DenoiserApp::setupDenoiser()
{
    // Check requirements
    if (!m_b_loaded) throw std::runtime_error("No input image loaded");
    if (m_n_loaded && !m_a_loaded) throw std::runtime_error("Normal map requires Albedo map");

    OIIO::ROI roi = OIIO::get_roi_full(m_beauty.data->spec());
    m_width = roi.width();
    m_height = roi.height();

    CU_CHECK(cudaSetDevice(m_selected_device_id));
    CU_CHECK(cudaStreamCreate(&m_cuda_stream));

    // OptiX Init
    OptixResult res = optixInit();
    if (res != OPTIX_SUCCESS) throw std::runtime_error("OptiX Init failed");

    OPTIX_CHECK(optixDeviceContextCreate(0, nullptr, &m_optix_context));

    OptixDenoiserOptions options = {};
    options.guideAlbedo = m_a_loaded;
    options.guideNormal = m_n_loaded;
    options.denoiseAlpha = OPTIX_DENOISER_ALPHA_MODE_COPY;

    OptixDenoiserModelKind model = (m_hdr) ? OPTIX_DENOISER_MODEL_KIND_HDR : OPTIX_DENOISER_MODEL_KIND_LDR;
    if (m_denoise_aovs) model = OPTIX_DENOISER_MODEL_KIND_AOV;
    if (m_pi_loaded) model = OPTIX_DENOISER_MODEL_KIND_TEMPORAL;
    
    if (m_denoise_aovs && m_pi_loaded)
        model = OPTIX_DENOISER_MODEL_KIND_TEMPORAL_AOV;

    OPTIX_CHECK(optixDenoiserCreate(m_optix_context, model, &options, &m_optix_denoiser));

    // Determine if tiling is needed and get overlap
    OPTIX_CHECK(optixDenoiserComputeMemoryResources(m_optix_denoiser, 512, 512, &m_denoiser_sizes));
    m_overlap = m_denoiser_sizes.overlapWindowSizeInPixels;

    unsigned int compute_width = m_width;
    unsigned int compute_height = m_height;

    if (m_width > m_tile_width || m_height > m_tile_height) {
        logInfo("Image too large for single pass, using tiling (%dx%d tiles with %d overlap)", m_tile_width, m_tile_height, m_overlap);
        compute_width = std::min((unsigned int)m_width, m_tile_width + 2 * m_overlap);
        compute_height = std::min((unsigned int)m_height, m_tile_height + 2 * m_overlap);
    }

    OPTIX_CHECK(optixDenoiserComputeMemoryResources(m_optix_denoiser, compute_width, compute_height, &m_denoiser_sizes));

    m_denoiser_state.alloc(m_denoiser_sizes.stateSizeInBytes);
    m_denoiser_scratch.alloc(m_denoiser_sizes.withoutOverlapScratchSizeInBytes);

    if (model == OPTIX_DENOISER_MODEL_KIND_TEMPORAL_AOV) {
        size_t internalGuideLayerSize = (size_t)compute_width * compute_height * m_denoiser_sizes.internalGuideLayerPixelSizeInBytes;
        m_internal_guide_layer_buffer.alloc(internalGuideLayerSize);
        m_internal_guide_layer_prev_buffer.alloc(internalGuideLayerSize);
        
        m_guide_layer.outputInternalGuideLayer.data = m_internal_guide_layer_buffer.d_ptr();
        m_guide_layer.outputInternalGuideLayer.width = compute_width;
        m_guide_layer.outputInternalGuideLayer.height = compute_height;
        m_guide_layer.outputInternalGuideLayer.pixelStrideInBytes = m_denoiser_sizes.internalGuideLayerPixelSizeInBytes;
        m_guide_layer.outputInternalGuideLayer.rowStrideInBytes = m_guide_layer.outputInternalGuideLayer.width * m_denoiser_sizes.internalGuideLayerPixelSizeInBytes;
        m_guide_layer.outputInternalGuideLayer.format = OPTIX_PIXEL_FORMAT_INTERNAL_GUIDE_LAYER;

        m_guide_layer.previousOutputInternalGuideLayer.data = m_internal_guide_layer_prev_buffer.d_ptr();
        m_guide_layer.previousOutputInternalGuideLayer.width = compute_width;
        m_guide_layer.previousOutputInternalGuideLayer.height = compute_height;
        m_guide_layer.previousOutputInternalGuideLayer.pixelStrideInBytes = m_denoiser_sizes.internalGuideLayerPixelSizeInBytes;
        m_guide_layer.previousOutputInternalGuideLayer.rowStrideInBytes = m_guide_layer.previousOutputInternalGuideLayer.width * m_denoiser_sizes.internalGuideLayerPixelSizeInBytes;
        m_guide_layer.previousOutputInternalGuideLayer.format = OPTIX_PIXEL_FORMAT_INTERNAL_GUIDE_LAYER;
    }

    OPTIX_CHECK(optixDenoiserSetup(m_optix_denoiser, m_cuda_stream, compute_width, compute_height, 
        m_denoiser_state.d_ptr(), m_denoiser_sizes.stateSizeInBytes,
        m_denoiser_scratch.d_ptr(), m_denoiser_sizes.withoutOverlapScratchSizeInBytes));

    // Params
    m_denoiser_params.blendFactor = m_blend;
    
    m_hdr_intensity.alloc(sizeof(float));
    m_denoiser_params.hdrIntensity = m_hdr_intensity.d_ptr();

    // Setup Layers
    m_layers.resize(1 + m_aovs.size());
    memset(m_layers.data(), 0, sizeof(OptixDenoiserLayer) * m_layers.size());

    // Allocate GPU buffers for layers
    // Each layer needs input and output, and potentially previousOutput
    unsigned int pixel_size_4 = sizeof(float) * 4;
    size_t img_size_bytes_4 = (size_t)m_width * m_height * pixel_size_4;

    for (auto& layer : m_layers)
    {
        CudaBuffer inBuf, outBuf;
        inBuf.alloc(img_size_bytes_4);
        outBuf.alloc(img_size_bytes_4);
        
        layer.input.data = inBuf.d_ptr();
        layer.input.width = m_width;
        layer.input.height = m_height;
        layer.input.rowStrideInBytes = m_width * pixel_size_4;
        layer.input.pixelStrideInBytes = pixel_size_4;
        layer.input.format = OPTIX_PIXEL_FORMAT_FLOAT4;

        layer.output.data = outBuf.d_ptr();
        layer.output.width = m_width;
        layer.output.height = m_height;
        layer.output.rowStrideInBytes = m_width * pixel_size_4;
        layer.output.pixelStrideInBytes = pixel_size_4;
        layer.output.format = OPTIX_PIXEL_FORMAT_FLOAT4;

        m_layer_buffers.push_back(std::move(inBuf));
        m_layer_buffers.push_back(std::move(outBuf));
    }

    if (m_pi_loaded)
    {
        CudaBuffer prevBuf;
        prevBuf.alloc(img_size_bytes_4);
        m_layers[0].previousOutput.data = prevBuf.d_ptr();
        m_layers[0].previousOutput.width = m_width;
        m_layers[0].previousOutput.height = m_height;
        m_layers[0].previousOutput.rowStrideInBytes = m_width * pixel_size_4;
        m_layers[0].previousOutput.pixelStrideInBytes = pixel_size_4;
        m_layers[0].previousOutput.format = OPTIX_PIXEL_FORMAT_FLOAT4;
        m_layer_buffers.push_back(std::move(prevBuf));
    }

    // Guide layers
    if (m_a_loaded) {
        m_albedo_buffer.alloc(img_size_bytes_4);
        m_guide_layer.albedo.data = m_albedo_buffer.d_ptr();
        m_guide_layer.albedo.width = m_width;
        m_guide_layer.albedo.height = m_height;
        m_guide_layer.albedo.rowStrideInBytes = m_width * pixel_size_4;
        m_guide_layer.albedo.pixelStrideInBytes = pixel_size_4;
        m_guide_layer.albedo.format = OPTIX_PIXEL_FORMAT_FLOAT4;
    }
    if (m_n_loaded) {
        m_normal_buffer.alloc(img_size_bytes_4);
        m_guide_layer.normal.data = m_normal_buffer.d_ptr();
        m_guide_layer.normal.width = m_width;
        m_guide_layer.normal.height = m_height;
        m_guide_layer.normal.rowStrideInBytes = m_width * pixel_size_4;
        m_guide_layer.normal.pixelStrideInBytes = pixel_size_4;
        m_guide_layer.normal.format = OPTIX_PIXEL_FORMAT_FLOAT4;
    }
    if (m_mv_loaded) {
        unsigned int pixel_size_2 = sizeof(float) * 2;
        m_flow_buffer.alloc((size_t)m_width * m_height * pixel_size_2);
        m_guide_layer.flow.data = m_flow_buffer.d_ptr();
        m_guide_layer.flow.width = m_width;
        m_guide_layer.flow.height = m_height;
        m_guide_layer.flow.rowStrideInBytes = m_width * pixel_size_2;
        m_guide_layer.flow.pixelStrideInBytes = pixel_size_2;
        m_guide_layer.flow.format = OPTIX_PIXEL_FORMAT_FLOAT2;
    }
    
    m_denoiser_params.temporalModeUsePreviousLayers = 0;
}

// Helper to convert pixels to format for OptiX using OIIO stride
// Optimized: Avoids manual loop by using get_pixels with stride
void DenoiserApp::uploadLayer(const ImageInfo& img, CudaBuffer& buffer, int target_channels)
{
    // Ensure we only read up to the configured resolution
    OIIO::ROI roi(0, m_width, 0, m_height, 0, 1, 0, img.data->nchannels());
    
    std::vector<float> scratch;
    scratch.resize((size_t)m_width * m_height * target_channels);
    
    // Fill with 0
    std::fill(scratch.begin(), scratch.end(), 0.0f);
    
    // Read directly into scratch with stride
    // xstride: bytes to next pixel
    // ystride: bytes to next row
    OIIO::stride_t xstride = target_channels * sizeof(float);
    OIIO::stride_t ystride = (OIIO::stride_t)m_width * xstride;

    img.data->get_pixels(roi, OIIO::TypeDesc::FLOAT, scratch.data(), xstride, ystride);
    
    buffer.copyToDevice(scratch.data(), scratch.size() * sizeof(float));
}

void DenoiserApp::executeDenoiser()
{
    // Upload Beauty
    uploadLayer(m_beauty, m_layer_buffers[0], 4);

    // Upload Previous
    if (m_pi_loaded) {
        // The previous output buffer is the last one in the list
        uploadLayer(m_prev_denoised, m_layer_buffers[m_layer_buffers.size() - 1], 4);
    }

    // Guides
    if (m_a_loaded) uploadLayer(m_albedo, m_albedo_buffer, 4);
    if (m_n_loaded) uploadLayer(m_normal, m_normal_buffer, 4);
    if (m_mv_loaded) uploadLayer(m_motion_vectors, m_flow_buffer, 2);

    // AOVs
    int aov_idx = 1;
    for(auto& pair : m_aovs) {
        uploadLayer(pair.second, m_layer_buffers[aov_idx * 2], 4);
        aov_idx++;
    }

    // Compute Intensity on full image
    logInfo("Computing HDR intensity...");
    size_t intensity_scratch_size = sizeof(int) * (2 + (size_t)m_width * m_height);
    CudaBuffer intensity_scratch;
    intensity_scratch.alloc(intensity_scratch_size);
    OPTIX_CHECK(optixDenoiserComputeIntensity(m_optix_denoiser, m_cuda_stream, &m_layers[0].input, m_denoiser_params.hdrIntensity,
        intensity_scratch.d_ptr(), intensity_scratch.size()));

    // Execute
    bool use_tiling = (m_width > m_tile_width || m_height > m_tile_height);
    long long total_time = 0;

    for (unsigned int run = 0; run < m_num_runs; run++)
    {
        auto start = std::chrono::high_resolution_clock::now();

        if (run > 0 || m_pi_loaded) {
             m_denoiser_params.temporalModeUsePreviousLayers = 1;
        }

        if (!use_tiling)
        {
            logInfo("Denoising (single pass)...");
            OPTIX_CHECK(optixDenoiserInvoke(m_optix_denoiser, m_cuda_stream, &m_denoiser_params,
                m_denoiser_state.d_ptr(), m_denoiser_sizes.stateSizeInBytes,
                &m_guide_layer, m_layers.data(), (unsigned int)m_layers.size(), 0, 0,
                m_denoiser_scratch.d_ptr(), m_denoiser_sizes.withoutOverlapScratchSizeInBytes));
        }
        else
        {
            logInfo("Denoising (tiled)...");
            for (unsigned int ty = 0; ty < (unsigned int)m_height; ty += m_tile_height)
            {
                for (unsigned int tx = 0; tx < (unsigned int)m_width; tx += m_tile_width)
                {
                    // Valid region for this tile
                    unsigned int tw = std::min(m_tile_width, (unsigned int)m_width - tx);
                    unsigned int th = std::min(m_tile_height, (unsigned int)m_height - ty);

                    // Input region with overlap
                    int x0 = std::max(0, (int)tx - (int)m_overlap);
                    int y0 = std::max(0, (int)ty - (int)m_overlap);
                    int x1 = std::min(m_width, (int)(tx + tw + m_overlap));
                    int y1 = std::min(m_height, (int)(ty + th + m_overlap));

                    unsigned int sub_w = x1 - x0;
                    unsigned int sub_h = y1 - y0;

                    // Offset of the valid region within the input tile
                    unsigned int inputOffsetX = tx - x0;
                    unsigned int inputOffsetY = ty - y0;

                    // Set up tile-specific layers and guide
                    OptixDenoiserGuideLayer tile_guide = m_guide_layer;
                    std::vector<OptixDenoiserLayer> tile_layers = m_layers;

                    // Helper to map input images (includes overlap)
                    auto set_input_tile = [&](OptixImage2D& img, CUdeviceptr base_ptr, int pixel_stride) {
                        img.data = base_ptr + (size_t)(y0 * m_width + x0) * pixel_stride;
                        img.width = sub_w;
                        img.height = sub_h;
                        img.rowStrideInBytes = m_width * pixel_stride;
                    };

                    // Helper to map output images (valid region only)
                    auto set_output_tile = [&](OptixImage2D& img, CUdeviceptr base_ptr, int pixel_stride) {
                        img.data = base_ptr + (size_t)(ty * m_width + tx) * pixel_stride;
                        img.width = tw;
                        img.height = th;
                        img.rowStrideInBytes = m_width * pixel_stride;
                    };

                    if (m_a_loaded) set_input_tile(tile_guide.albedo, m_albedo_buffer.d_ptr(), sizeof(float) * 4);
                    if (m_n_loaded) set_input_tile(tile_guide.normal, m_normal_buffer.d_ptr(), sizeof(float) * 4);
                    if (m_mv_loaded) set_input_tile(tile_guide.flow, m_flow_buffer.d_ptr(), sizeof(float) * 2);
                    
                    // Handle tile mapping for internal guide layers if present
                    if (m_guide_layer.previousOutputInternalGuideLayer.data) {
                         set_input_tile(tile_guide.previousOutputInternalGuideLayer, m_internal_guide_layer_prev_buffer.d_ptr(), m_denoiser_sizes.internalGuideLayerPixelSizeInBytes);
                         set_output_tile(tile_guide.outputInternalGuideLayer, m_internal_guide_layer_buffer.d_ptr(), m_denoiser_sizes.internalGuideLayerPixelSizeInBytes);
                    }

                    for (size_t i = 0; i < tile_layers.size(); ++i)
                    {
                        // Input reads from overlapped region
                        set_input_tile(tile_layers[i].input, m_layer_buffers[i * 2].d_ptr(), sizeof(float) * 4);
                        
                        // Output writes to valid region only
                        set_output_tile(tile_layers[i].output, m_layer_buffers[i * 2 + 1].d_ptr(), sizeof(float) * 4);
                        
                        if (i == 0 && m_pi_loaded)
                            set_input_tile(tile_layers[i].previousOutput, m_layer_buffers[tile_layers.size() * 2].d_ptr(), sizeof(float) * 4);
                    }

                    OPTIX_CHECK(optixDenoiserInvoke(m_optix_denoiser, m_cuda_stream, &m_denoiser_params,
                        m_denoiser_state.d_ptr(), m_denoiser_sizes.stateSizeInBytes,
                        &tile_guide, tile_layers.data(), (unsigned int)tile_layers.size(), 
                        inputOffsetX, inputOffsetY,
                        m_denoiser_scratch.d_ptr(), m_denoiser_sizes.withoutOverlapScratchSizeInBytes));
                }
            }
        }
        
        // If we have internal guide layers (Temporal AOV), swap them for the next run
        if (m_guide_layer.previousOutputInternalGuideLayer.data) {
             std::swap(m_guide_layer.previousOutputInternalGuideLayer.data, m_guide_layer.outputInternalGuideLayer.data);
        }

        CU_CHECK(cudaStreamSynchronize(m_cuda_stream)); // Wait for completion

        auto end = std::chrono::high_resolution_clock::now();
        long long msec = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        total_time += msec;
        
        if (m_num_runs > 1) logInfo("Run %u: %lld ms", run, msec);
        else logInfo("Complete in %lld ms", msec);
    }
    if (m_num_runs > 1) logInfo("Average: %lld ms", total_time / m_num_runs);
}

void DenoiserApp::saveImages()
{
    // Download and Save
    std::vector<float> host_pixels(m_width * m_height * 4);
    
    // Save Beauty
    CU_CHECK(cudaMemcpy(host_pixels.data(), reinterpret_cast<const void*>(m_layers[0].output.data), host_pixels.size() * sizeof(float), cudaMemcpyDeviceToHost));
    
    // Convert back to original channels
    int channels = OIIO::get_roi_full(m_beauty.data->spec()).nchannels();
    OIIO::TypeDesc original_format = m_beauty.data->spec().format;

    std::vector<float> final_pixels((size_t)m_width * m_height * channels);
    for(int i=0; i<m_width*m_height; ++i) {
        for(int c=0; c<channels; ++c) {
            final_pixels[(size_t)i*channels + c] = host_pixels[(size_t)i*4 + c];
        }
    }
    m_beauty.data->set_pixels(OIIO::get_roi_full(m_beauty.data->spec()), OIIO::TypeDesc::FLOAT, final_pixels.data());
    
    std::string out_path = m_beauty.output_filename;
    if (out_path.empty()) {
        if (!m_out_suffix.empty()) {
             // Basic suffix logic
             std::string base = m_beauty.filename.substr(0, m_beauty.filename.find_last_of("."));
             std::string ext = m_beauty.filename.substr(m_beauty.filename.find_last_of("."));
             out_path = base + m_out_suffix + ext;
        } else {
            logError("No output filename or suffix specified");
            return;
        }
    }
    m_beauty.data->write(out_path, original_format);
    logInfo("Saved: %s", out_path.c_str());

    // Save AOVs
    int aov_idx = 1;
    for(auto& pair : m_aovs) {
        CU_CHECK(cudaMemcpy(host_pixels.data(), reinterpret_cast<const void*>(m_layers[aov_idx].output.data), host_pixels.size() * sizeof(float), cudaMemcpyDeviceToHost));
        
        channels = OIIO::get_roi_full(pair.second.data->spec()).nchannels();
        OIIO::TypeDesc aov_fmt = pair.second.data->spec().format;
        
        final_pixels.resize((size_t)m_width * m_height * channels);
        for(int i=0; i<m_width*m_height; ++i) {
             for(int c=0; c<channels; ++c) {
                 final_pixels[(size_t)i*channels + c] = host_pixels[(size_t)i*4 + c];
             }
        }
        pair.second.data->set_pixels(OIIO::get_roi_full(pair.second.data->spec()), OIIO::TypeDesc::FLOAT, final_pixels.data());

        std::string aov_out = pair.second.output_filename;
        if (aov_out.empty() && !m_out_suffix.empty()) {
             std::string base = pair.second.filename.substr(0, pair.second.filename.find_last_of("."));
             std::string ext = pair.second.filename.substr(pair.second.filename.find_last_of("."));
             aov_out = base + m_out_suffix + ext;
        }
        if (!aov_out.empty()) {
            pair.second.data->write(aov_out, aov_fmt);
            logInfo("Saved AOV %d: %s", pair.first, aov_out.c_str());
        }
        aov_idx++;
    }
}

void DenoiserApp::run(int argc, char *argv[])
{
    logInfo("Launching Nvidia AI Denoiser command line app v%d.%d", DENOISER_MAJOR_VERSION, DENOISER_MINOR_VERSION);
    if (argc <= 1) {
        printParams();
        return;
    }

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "-h" || arg == "--help") {
            printParams();
            return;
        }
    }

    try {
        if (!discoverDevices()) return;
        loadImages(argc, argv);
        setupDenoiser();
        executeDenoiser();
        saveImages();
        logInfo("Done!");
    } catch (const std::exception& e) {
        logError("Exception: %s", e.what());
        exit(EXIT_FAILURE);
    }
}

int main(int argc, char *argv[])
{
    DenoiserApp app;
    app.run(argc, argv);
    return 0;
}