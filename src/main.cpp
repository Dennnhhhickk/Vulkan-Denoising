#include <vulkan/vulkan.h>

#include <vector>
#define STB_IMAGE_IMPLEMENTATION 
#include "Bitmap1.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "Bitmap2.h"
#include <fstream>
#include <string.h>
#include <assert.h>
#include <stdexcept>
#include <cmath>
#include <array>
#include <iomanip>

#define sqr(a) ((a)*(a))

#include "Bitmap.h" // Save bmp file

#include <iostream>
#include <chrono>

int WIDTH = 4;  // Size of rendered mandelbrot set.
int HEIGHT = 4;  // Size of renderered mandelbrot set.
int R = 2;
float DISPER = 1, eps = 1e-6;
int stepH = 1, stepW = 1;
const int WORKGROUP_SIZE = 16;    // Workgroup size in compute shader.

// #ifdef NDEBUG
const bool enableValidationLayers = false;
// #else
// const bool enableValidationLayers = true;
// #endif

#include "vk_utils.h"


/*
The application launches a compute shader that renders the mandelbrot set,
by rendering it into a storage buffer.
The storage buffer is then read from the GPU, and saved as .png. 
*/
class ComputeApplication
{
private:
    // The pixels of the rendered mandelbrot set are in this format:
    struct Pixel {
        float r, g, b, a;
    };
    
    /*
    In order to use Vulkan, you must create an instance. 
    */
    VkInstance instance;

    VkDebugReportCallbackEXT debugReportCallback;
    /*
    The physical device is some device on the system that supports usage of Vulkan.
    Often, it is simply a graphics card that supports Vulkan. 
    */
    VkPhysicalDevice physicalDevice;
    /*
    Then we have the logical device VkDevice, which basically allows 
    us to interact with the physical device. 
    */
    VkDevice device;

    /*
    The pipeline specifies the pipeline that all graphics and compute commands pass though in Vulkan.

    We will be creating a simple compute pipeline in this application. 
    */
    VkPipeline       pipeline;
    VkPipelineLayout pipelineLayout;
    VkShaderModule   computeShaderModule;

    /*
    The command buffer is used to record commands, that will be submitted to a queue.

    To allocate such command buffers, we use a command pool.
    */
    VkCommandPool   commandPool;
    VkCommandBuffer commandBuffer;

    /*

    Descriptors represent resources in shaders. They allow us to use things like
    uniform buffers, storage buffers and images in GLSL. 

    A single descriptor represents a single resource, and several descriptors are organized
    into descriptor sets, which are basically just collections of descriptors.
    */
    VkDescriptorPool      descriptorPool;
    VkDescriptorSet       descriptorSet;
    VkDescriptorSetLayout descriptorSetLayout;

    /*
    The mandelbrot set will be rendered to this buffer.

    The memory that backs the buffer is bufferMemory. 
    */
    VkBuffer       buffer;
    VkDeviceMemory bufferMemory;

    std::vector<const char *> enabledLayers;

    /*
    In order to execute commands on a device(GPU), the commands must be submitted
    to a queue. The commands are stored in a command buffer, and this command buffer
    is given to the queue. 

    There will be different kinds of queues on the device. Not all queues support
    graphics operations, for instance. For this application, we at least want a queue
    that supports compute operations. 
    */
    VkQueue queue; // a queue supporting compute operations.

public:

    void run()
    {
        std::cout << "Image path: ";
        std::string s;
        std::cin >> s;
        std::cout << "Modes:\n0 - CPU bilateral filter\n1 - GPU Non Local Means filter\n2 - GPU Bilateral filter\nMode: ";
        int mode1;
        std::cin >> mode1;
        std::cout << "read image ..." << std::endl;
        int a_height, a_width;
        std::vector<std::vector<Pixel> > tmp = readBMP(s.c_str(), a_width, a_height);
        srand(time(NULL));
        // std::cout << "saving image       ... " << std::endl;
        // saveRenderedImage(tmp, a_width, a_height, "bb/Source.png");
        if (mode1 == 0) {
            std::cout << "CPU BILATERAL" << std::endl;
            std::cout << "Name of processed image: ";
            std::string nam;
            std::cin >> nam;
            std::vector<std::vector<Pixel> > FinalImage2 = CPU(tmp, a_width, a_height);
            saveRenderedImage(FinalImage2, a_width, a_height, nam.c_str());
        }
        if (mode1 == 1) {
            std::cout << "NLM" << std::endl;
            std::cout << "Name of processed image: ";
            std::string nam;
            std::cin >> nam; 
            std::cout << "Number of adjacent blocks to be processed(less - faster, more-better quality, minimum 1, maximum photo resolution / 11):" << std::endl;
            std::cout << "Width:";
            std::cin >> stepW;
            std::cout << "Height:";
            std::cin >> stepH;
            std::vector<std::vector<Pixel> > FinalImage = GPU_NON_LOCAL_MEANS(tmp, a_width, a_height);
            saveRenderedImage(FinalImage, a_width, a_height, (nam.c_str()));
        }
        if (mode1 == 2) {
            std::cout << "GPU BILATERAL" << std::endl; 
            std::cout << "Name of processed image: ";
            std::string nam;
            std::cin >> nam;
            std::vector<std::vector<Pixel> > FinalImage1 = GPU_BILATERAL(tmp, a_width, a_height);
            saveRenderedImage(FinalImage1, a_width, a_height, nam.c_str());
        }
        // std::cout << pnsr(FinalImage, tmp) << ' ' << pnsr(FinalImage1, tmp) << ' ' << pnsr(FinalImage2, tmp) << std::endl;
    }

    static float ddistance(Pixel a, Pixel b)
    {
        Pixel c;
        c.r = 256 * (a.r - b.r);
        c.g = 256 * (a.g - b.g);
        c.b = 256 * (a.b - b.b);
        c.a = 256 * (a.a - b.a);
        return sqrt(sqr(c.r) + sqr(c.g) + sqr(c.b) + sqr(c.a));
    }

    static float w(std::vector<std::vector<Pixel> > &tmp, uint posx, uint posy, uint x, uint y, float S1, float S2)
    {
        return std::exp(-((((posx - x) * (posx - x) + (posy - y) * (posy - y)) / (2 * S1 * S1)) + (ddistance(tmp[y][x], tmp[posy][posx]) / (2 * S2 * S2))));
    }

    static std::vector<std::vector<Pixel> > CPU(std::vector<std::vector<Pixel> > &tmp, int a_width, int a_height)
    {
        unsigned int start =  clock();
        std::vector<std::vector<Pixel> > FinalImage(a_height);
        float ans, a2;
        Pixel a1{0};
        WIDTH = 32;
        HEIGHT = 32;
        float S1 = 3;
        float S2 = 3;
        std::cout << std::setprecision(5) << std::fixed;
        for (int y = 0; y < a_height; ++y) {
            FinalImage[y].resize(a_width);
            for (int x = 0; x < a_width; ++x) {
                a2 = 0;
                a1.r = 0;
                a1.g = 0;
                a1.b = 0;
                a1.a = 0;
                for (int i = 0; i < HEIGHT; ++i) {
                    if (y + i < a_height) {
                        for (int j = 0; j < WIDTH; ++j) {
                            if (x + j < a_width) {
                                ans = w(tmp, x, y, x + j, y + i, S1, S2);
                                a1.r += tmp[y + i][x + j].r * ans;
                                a1.g += tmp[y + i][x + j].g * ans;
                                a1.b += tmp[y + i][x + j].b * ans;
                                a1.a += tmp[y + i][x + j].a * ans;
                                a2 += ans;
                            }
                            if (x - j >= 0) {
                                ans = w(tmp, x, y, x - j, y + i, S1, S2);
                                a1.r += tmp[y + i][x - j].r * ans;
                                a1.g += tmp[y + i][x - j].g * ans;
                                a1.b += tmp[y + i][x - j].b * ans;
                                a1.a += tmp[y + i][x - j].a * ans;
                                a2 += ans;
                            }
                        }
                    }
                    if (y - i >= 0) {
                        for (int j = 0; j < WIDTH; ++j) {
                            if (x + j < a_width) {
                                ans = w(tmp, x, y, x + j, y - i, S1, S2);
                                a1.r += tmp[y - i][x + j].r * ans;
                                a1.g += tmp[y - i][x + j].g * ans;
                                a1.b += tmp[y - i][x + j].b * ans;
                                a1.a += tmp[y - i][x + j].a * ans;
                                a2 += ans;
                            }
                            if (x - j >= 0) {
                                ans = w(tmp, x, y, x - j, y - i, S1, S2);
                                a1.r += tmp[y - i][x - j].r * ans;
                                a1.g += tmp[y - i][x - j].g * ans;
                                a1.b += tmp[y - i][x - j].b * ans;
                                a1.a += tmp[y - i][x - j].a * ans;
                                a2 += ans;
                            }
                        }
                    }
                }
                FinalImage[y][x].r = a1.r / a2;
                FinalImage[y][x].g = a1.g / a2;
                FinalImage[y][x].b = a1.b / a2;
                FinalImage[y][x].a = a1.a / a2;
                std::cout << "processing " << (x + a_width * y) * 100.0 / (a_width * a_height) << "% ... " << std::endl;
            }
        }
        std::cout << std::setprecision(2) << std::fixed;
        unsigned int finish = clock();
        std::cout << "Time: " << (finish-start) / (CLOCKS_PER_SEC * 1.0) << "s" << std::endl;
        return FinalImage;
    }

    std::vector<std::vector<Pixel> > GPU_NON_LOCAL_MEANS(std::vector<std::vector<Pixel> > &tmp, int a_width, int a_height)
    {
        auto start2 = std::chrono::high_resolution_clock::now();
        const int deviceId = 0;

        std::cout << "init vulkan for device " << deviceId << " ... " << std::endl;
        instance = vk_utils::CreateInstance(enableValidationLayers, enabledLayers);
        if(enableValidationLayers) {
            vk_utils::InitDebugReportCallback(instance, &debugReportCallbackFn, &debugReportCallback);
        }
        physicalDevice = vk_utils::FindPhysicalDevice(instance, true, deviceId);
        uint32_t queueFamilyIndex = vk_utils::GetComputeQueueFamilyIndex(physicalDevice);
        device = vk_utils::CreateLogicalDevice(queueFamilyIndex, physicalDevice, enabledLayers);
        vkGetDeviceQueue(device, queueFamilyIndex, 0, &queue);
        size_t bufferSize = sizeof(Pixel) * (2 + 3 * a_width * a_height);
        std::cout << "creating resources ... " << std::endl;
        createBuffer(device, physicalDevice, bufferSize, &buffer, &bufferMemory);
        createDescriptorSetLayout(device, &descriptorSetLayout);
        createDescriptorSetForOurBuffer(device, buffer, bufferSize, &descriptorSetLayout, &descriptorPool, &descriptorSet);
        clearMemory(device, bufferMemory, 0, 1, 1);
        clearMemory(device, bufferMemory, sizeof(Pixel), 1, 1);
        clearMemory(device, bufferMemory, sizeof(Pixel) * 2, a_width, a_height);
        clearMemory(device, bufferMemory, sizeof(Pixel) * (2 + a_height * a_width), a_width, a_height);
        clearMemory(device, bufferMemory, sizeof(Pixel) * (2 + 2 * a_height * a_width), a_width, a_height);
        int h_offset, w_offset;
        Pixel tmp1, tmp2;
        tmp1.b = a_height;
        tmp1.a = a_width;
        tmp1.r = 0;
        tmp1.g = 0;
        tmp2.r = WIDTH;
        tmp2.g = HEIGHT;
        tmp2.b = R;
        tmp2.a = DISPER;
        writeImageToDeviceMemory(device, bufferMemory, 2 * sizeof(Pixel), a_width, a_height, tmp);
        writeImageToDeviceMemory(device, bufferMemory, sizeof(Pixel), tmp1);
        {
            float l = 0, r = 30, m1, m2;
            tmp2.a = 30;
            std::vector<std::vector<Pixel> > im1, im2;
            writeImageToDeviceMemory(device, bufferMemory, 0, tmp2);
            clearMemory(device, bufferMemory, sizeof(Pixel) * (2 + a_height * a_width), a_width, a_height);
            clearMemory(device, bufferMemory, sizeof(Pixel) * (2 + 2 * a_height * a_width), a_width, a_height);
            createComputePipeline(device, descriptorSetLayout, &computeShaderModule, &pipeline, &pipelineLayout, "shaders/nlm.spv");
            createCommandBuffer(device, queueFamilyIndex, pipeline, pipelineLayout, &commandPool, &commandBuffer);
            recordCommandsTo(commandBuffer, pipeline, pipelineLayout, descriptorSet, (uint32_t)(ceil(a_width / float(WORKGROUP_SIZE))), (uint32_t)(ceil(a_height / float(WORKGROUP_SIZE))));
            runCommandBuffer(commandBuffer, queue, device);
            vkDestroyShaderModule(device, computeShaderModule, NULL);
            vkDestroyPipelineLayout(device, pipelineLayout, NULL);
            vkDestroyPipeline(device, pipeline, NULL);
            vkDestroyCommandPool(device, commandPool, NULL);
            im1 = getImage(device, bufferMemory, a_width, a_height, tmp);
            eps = pnsr(im1, tmp) * sqr(1.1);
            while (r - l > DISPER) {
                std::cout << "preparatory calculations ... " << std::endl;
                m1 = (r + l) / 2;
                auto start = std::chrono::high_resolution_clock::now();
                tmp2.a = m1;
                writeImageToDeviceMemory(device, bufferMemory, 0, tmp2);
                clearMemory(device, bufferMemory, sizeof(Pixel) * (2 + a_height * a_width), a_width, a_height);
                clearMemory(device, bufferMemory, sizeof(Pixel) * (2 + 2 * a_height * a_width), a_width, a_height);
                createComputePipeline(device, descriptorSetLayout, &computeShaderModule, &pipeline, &pipelineLayout, "shaders/nlm.spv");
                createCommandBuffer(device, queueFamilyIndex, pipeline, pipelineLayout, &commandPool, &commandBuffer);
                recordCommandsTo(commandBuffer, pipeline, pipelineLayout, descriptorSet, (uint32_t)(ceil(a_width / float(WORKGROUP_SIZE))), (uint32_t)(ceil(a_height / float(WORKGROUP_SIZE))));
                runCommandBuffer(commandBuffer, queue, device);
                vkDestroyShaderModule(device, computeShaderModule, NULL);
                vkDestroyPipelineLayout(device, pipelineLayout, NULL);
                vkDestroyPipeline(device, pipeline, NULL);
                vkDestroyCommandPool(device, commandPool, NULL);
                im1 = getImage(device, bufferMemory, a_width, a_height, tmp);
                float rr = pnsr(im1, tmp);
                if (std::abs(rr) - eps > 0) {
                    l = m1;
                } else {
                    r = m1;
                }
            }
            tmp2.a = (l + r) / 2;
            clearMemory(device, bufferMemory, sizeof(Pixel) * (2 + a_height * a_width), a_width, a_height);
            clearMemory(device, bufferMemory, sizeof(Pixel) * (2 + 2 * a_height * a_width), a_width, a_height);
        }
        writeImageToDeviceMemory(device, bufferMemory, 0, tmp2);
        auto ttime = 0;
        // writeImageToDeviceMemory(device, bufferMemory, 2 * sizeof(Pixel), a_width, a_height, tmp);
        for (int h = 0; h < stepH; ++h) {
            for (int w = 0; w < stepW; ++w) {
                auto start1 = std::chrono::high_resolution_clock::now();
                tmp1.r = h;
                tmp1.g = w;
                writeImageToDeviceMemory(device, bufferMemory, sizeof(Pixel), tmp1);
                std::cout << "processing " << int((w + stepW * h) * 100.0 / (stepH * stepW)) << "% ... ";
                createComputePipeline(device, descriptorSetLayout, &computeShaderModule, &pipeline, &pipelineLayout, "shaders/nlm.spv");
                createCommandBuffer(device, queueFamilyIndex, pipeline, pipelineLayout, &commandPool, &commandBuffer);
                recordCommandsTo(commandBuffer, pipeline, pipelineLayout, descriptorSet, (uint32_t)(ceil(a_width / float(WORKGROUP_SIZE))), (uint32_t)(ceil(a_height / float(WORKGROUP_SIZE))));
                auto start = std::chrono::high_resolution_clock::now();
                // std::cout << "doing computations ... " << std::endl;
                runCommandBuffer(commandBuffer, queue, device);
                auto finish = std::chrono::high_resolution_clock::now();
                std::cout << std::chrono::duration_cast<std::chrono::nanoseconds>(finish-start).count() << "ns " << std::endl;

                vkDestroyShaderModule(device, computeShaderModule, NULL);
                vkDestroyPipelineLayout(device, pipelineLayout, NULL);
                vkDestroyPipeline(device, pipeline, NULL);
                vkDestroyCommandPool(device, commandPool, NULL);
                auto finish1 = std::chrono::high_resolution_clock::now();
                ttime += std::chrono::duration_cast<std::chrono::nanoseconds>(finish-start).count();
                // std::cout << std::chrono::duration_cast<std::chrono::nanoseconds>(finish1-start1).count() << "ns" << std::endl;
            }
        }
        std::cout << "processing 100% " << std::endl;
        std::vector<std::vector<Pixel> > FinalImage = getImage(device, bufferMemory, a_width, a_height, tmp);
        std::cout << "destroying all     ... " << std::endl;
        cleanup();
        auto finish2 = std::chrono::high_resolution_clock::now();
        std::cout << "Time with copy:" << std::chrono::duration_cast<std::chrono::nanoseconds>(finish2-start2).count() << "ns" << std::endl;
        std::cout << "Time without copy:" << ttime << "ns" << std::endl;
        std::cout << "Time copy:" << std::chrono::duration_cast<std::chrono::nanoseconds>(finish2-start2).count() - ttime << "ns" << std::endl;
        return FinalImage;
    }

    std::vector<std::vector<Pixel> > GPU_BILATERAL(std::vector<std::vector<Pixel> > &tmp, int a_width, int a_height)
    {
        auto start2 = std::chrono::high_resolution_clock::now();
        const int deviceId = 0;

        std::cout << "init vulkan for device " << deviceId << " ... " << std::endl;
        instance = vk_utils::CreateInstance(enableValidationLayers, enabledLayers);
        if(enableValidationLayers) {
            vk_utils::InitDebugReportCallback(instance, &debugReportCallbackFn, &debugReportCallback);
        }
        physicalDevice = vk_utils::FindPhysicalDevice(instance, true, deviceId);
        uint32_t queueFamilyIndex = vk_utils::GetComputeQueueFamilyIndex(physicalDevice);
        device = vk_utils::CreateLogicalDevice(queueFamilyIndex, physicalDevice, enabledLayers);
        vkGetDeviceQueue(device, queueFamilyIndex, 0, &queue);
        size_t bufferSize = sizeof(Pixel) * (2 + 2 * a_width * a_height);
        std::cout << "creating resources ... " << std::endl;
        createBuffer(device, physicalDevice, bufferSize, &buffer, &bufferMemory);
        createDescriptorSetLayout(device, &descriptorSetLayout);
        createDescriptorSetForOurBuffer(device, buffer, bufferSize, &descriptorSetLayout, &descriptorPool, &descriptorSet);
        clearMemory(device, bufferMemory, 0, 1, 1);
        clearMemory(device, bufferMemory, sizeof(Pixel), 1, 1);
        clearMemory(device, bufferMemory, sizeof(Pixel) * 2, a_width, a_height);
        clearMemory(device, bufferMemory, sizeof(Pixel) * (2 + a_height * a_width), a_width, a_height);
        int h_offset, w_offset;
        Pixel tmp1, tmp2;
        tmp1.b = a_height;
        tmp1.a = a_width;
        tmp1.r = 0;
        tmp1.g = 0;
        tmp2.r = 32;
        tmp2.g = 32;
        tmp2.b = R;
        tmp2.a = DISPER;
        writeImageToDeviceMemory(device, bufferMemory, 2 * sizeof(Pixel), a_width, a_height, tmp);
        writeImageToDeviceMemory(device, bufferMemory, 0, tmp2);
        writeImageToDeviceMemory(device, bufferMemory, sizeof(Pixel), tmp1);
        {
            float l = 0, r = 20, m1, m2;
            tmp2.a = r;
            tmp2.b = r;
            std::vector<std::vector<Pixel> > im1, im2;
            writeImageToDeviceMemory(device, bufferMemory, 0, tmp2);
            createComputePipeline(device, descriptorSetLayout, &computeShaderModule, &pipeline, &pipelineLayout, "shaders/bilateral.spv");
            createCommandBuffer(device, queueFamilyIndex, pipeline, pipelineLayout, &commandPool, &commandBuffer);
            recordCommandsTo(commandBuffer, pipeline, pipelineLayout, descriptorSet, (uint32_t)(ceil(a_width / float(WORKGROUP_SIZE))), (uint32_t)(ceil(a_height / float(WORKGROUP_SIZE))));
            runCommandBuffer(commandBuffer, queue, device);
            vkDestroyShaderModule(device, computeShaderModule, NULL);
            vkDestroyPipelineLayout(device, pipelineLayout, NULL);
            vkDestroyPipeline(device, pipeline, NULL);
            vkDestroyCommandPool(device, commandPool, NULL);
            im1 = COPY(device, bufferMemory, (2 + a_width * a_height) * sizeof(Pixel), a_width, a_height);
            eps = pnsr(im1, tmp) * sqr(1.16);
            while (r - l > DISPER) {
                m1 = (r + l) / 2;
                std::cout << "preparatory calculations ";
                auto start = std::chrono::high_resolution_clock::now();
                tmp2.a = m1;
                tmp2.b = m1;
                writeImageToDeviceMemory(device, bufferMemory, 0, tmp2);
                clearMemory(device, bufferMemory, sizeof(Pixel) * (2 + a_height * a_width), a_width, a_height);
                createComputePipeline(device, descriptorSetLayout, &computeShaderModule, &pipeline, &pipelineLayout, "shaders/bilateral.spv");
                createCommandBuffer(device, queueFamilyIndex, pipeline, pipelineLayout, &commandPool, &commandBuffer);
                recordCommandsTo(commandBuffer, pipeline, pipelineLayout, descriptorSet, (uint32_t)(ceil(a_width / float(WORKGROUP_SIZE))), (uint32_t)(ceil(a_height / float(WORKGROUP_SIZE))));
                runCommandBuffer(commandBuffer, queue, device);
                vkDestroyShaderModule(device, computeShaderModule, NULL);
                vkDestroyPipelineLayout(device, pipelineLayout, NULL);
                vkDestroyPipeline(device, pipeline, NULL);
                vkDestroyCommandPool(device, commandPool, NULL);
                im1 = COPY(device, bufferMemory, (2 + a_width * a_height) * sizeof(Pixel), a_width, a_height);
                auto finish = std::chrono::high_resolution_clock::now();
                std::cout << std::chrono::duration_cast<std::chrono::nanoseconds>(finish-start).count() << "ns " << std::endl;
                float rr = pnsr(im1, tmp);
                if (std::abs(rr) - eps > 0) {
                    l = m1;
                } else {
                    r = m1;
                }
            }
            tmp2.a = (l + r) / 2;
            tmp2.b = (l + r) / 2;
            clearMemory(device, bufferMemory, sizeof(Pixel) * (2 + a_height * a_width), a_width, a_height);
        }
        writeImageToDeviceMemory(device, bufferMemory, 0, tmp2);
        // writeImageToDeviceMemory(device, bufferMemory, 2 * sizeof(Pixel), a_width, a_height, tmp);
        auto start1 = std::chrono::high_resolution_clock::now();
        writeImageToDeviceMemory(device, bufferMemory, sizeof(Pixel), tmp1);
        std::cout << "start processing ... " << std::endl;
        createComputePipeline(device, descriptorSetLayout, &computeShaderModule, &pipeline, &pipelineLayout, "shaders/bilateral.spv");
        createCommandBuffer(device, queueFamilyIndex, pipeline, pipelineLayout, &commandPool, &commandBuffer);
        recordCommandsTo(commandBuffer, pipeline, pipelineLayout, descriptorSet, (uint32_t)(ceil(a_width / float(WORKGROUP_SIZE))), (uint32_t)(ceil(a_height / float(WORKGROUP_SIZE))));
        auto start = std::chrono::high_resolution_clock::now();
        std::cout << "doing computations ... " << std::endl;
        runCommandBuffer(commandBuffer, queue, device);
        auto finish = std::chrono::high_resolution_clock::now();
        std::cout << std::chrono::duration_cast<std::chrono::nanoseconds>(finish-start).count() << "ns ";
        vkDestroyShaderModule(device, computeShaderModule, NULL);
        vkDestroyPipeline(device, pipeline, NULL);
        vkDestroyPipelineLayout(device, pipelineLayout, NULL);
        vkDestroyCommandPool(device, commandPool, NULL);
        auto finish1 = std::chrono::high_resolution_clock::now();
        std::cout << std::chrono::duration_cast<std::chrono::nanoseconds>(finish1-start1).count() << "ns" << std::endl;

        std::vector<std::vector<Pixel> > FinalImage = COPY(device, bufferMemory, (2 + a_width * a_height) * sizeof(Pixel), a_width, a_height);
        std::cout << "destroying all     ... " << std::endl;
        cleanup();
        auto finish2 = std::chrono::high_resolution_clock::now();
        std::cout << "Time with copy:" << std::chrono::duration_cast<std::chrono::nanoseconds>(finish2-start2).count() << "ns" << std::endl;
        std::cout << "Time without copy:" << std::chrono::duration_cast<std::chrono::nanoseconds>(finish1-start1).count() << "ns" << std::endl;
        std::cout << "Time copy:" << std::chrono::duration_cast<std::chrono::nanoseconds>(finish2-start2).count() - std::chrono::duration_cast<std::chrono::nanoseconds>(finish1-start1).count() << "ns" << std::endl;
        return FinalImage;
    }

    std::vector<std::vector<Pixel> > IMAGE(int width, int height)
    {
        Pixel tmp;
        tmp.a = 1.0f;
        std::vector<std::vector<Pixel> > res;
        res.resize(height);
        for (int i = 0; i < height; ++i) {
            res[i].resize(width);
            for (int j = 0; j < width; ++j) {
                float rr = (rand() % 100) / 1000.0f;
                if (sqr(i - height / 2) + sqr(j - width / 2) <= 625) {
                    tmp.r = 1.0f - rr;
                    tmp.g = 0.0f;
                    tmp.b = 0.0f;
                    res[i][j] = tmp;
                } else {
                    tmp.r = rr;
                    tmp.g = rr;
                    tmp.b = rr;
                    res[i][j] = tmp;
                }
            }
        }
        return res;
    }

    static float pnsr(std::vector<std::vector<Pixel> > &a, std::vector<std::vector<Pixel> > &b)
    {
        float res = 0.0;
        for (size_t i = 0; i < a.size(); ++i) {
            for (size_t j = 0; j < a[i].size(); ++j) {
                res += (sqr(a[i][j].r - b[i][j].r) + sqr(a[i][j].g - b[i][j].g) + sqr(a[i][j].b - b[i][j].b) + sqr(a[i][j].a - b[i][j].a)) / 4;
            }
        }
        return 10 * std::log10((a.size() * a[0].size()) * 4 / res);
    }

    static std::vector<std::vector<Pixel> > getImage(VkDevice a_device, VkDeviceMemory a_bufferMemory, int a_width, int a_height, std::vector<std::vector<Pixel> > &tmp)
    {
        std::vector<std::vector<Pixel> > FinalImage = COPY(a_device, a_bufferMemory, (2 + 2 * a_width * a_height) * sizeof(Pixel), a_width, a_height);
        std::vector<std::vector<Pixel> > FinalW = COPY(a_device, a_bufferMemory, (2 + a_width * a_height) * sizeof(Pixel), a_width, a_height);

        for (int i = 0; i < a_height; ++i) {
            for (int j = 0; j < R; ++j) {
                FinalImage[i][j] = tmp[i][j];
                FinalW[i][j].r = 1.0f;
                FinalImage[i][a_width - 1 - j] = tmp[i][a_width - 1 - j];
                FinalW[i][a_width - 1 - j].r = 1.0f;
            }
        }

        for (int i = 0; i < R; ++i) {
            for (int j = 0; j < a_width; ++j) {
                FinalImage[i][j] = tmp[i][j];
                FinalW[i][j].r = 1.0f;
                FinalImage[a_height - 1 - i][j] = tmp[a_height - 1 - i][j];
                FinalW[a_height - 1 - i][j].r = 1.0f;
            }
        }

        for (int i = 0; i < a_height; ++i) {
            for (int j = 0; j < a_width; ++j) {
                FinalImage[i][j].r = FinalImage[i][j].r / FinalW[i][j].r;
                FinalImage[i][j].g = FinalImage[i][j].g / FinalW[i][j].r;
                FinalImage[i][j].b = FinalImage[i][j].b / FinalW[i][j].r;
                FinalImage[i][j].a = FinalImage[i][j].a / FinalW[i][j].r;
            }
        }
        return FinalImage;
    }

    static std::vector<std::vector<Pixel> > COPY(VkDevice a_device, VkDeviceMemory a_bufferMemory, size_t a_offset, int a_width, int a_height)
    {
        std::vector<std::vector<Pixel> > ans;
        ans.resize(a_height);
        const int a_bufferSize = a_width * sizeof(Pixel);
        void* mappedMemory = nullptr;
        for (int i = 0; i < a_height; i += 1) 
        {
            size_t offset = a_offset + i * a_width * sizeof(Pixel);
            mappedMemory = nullptr;
            ans[i].resize(a_width);
            vkMapMemory(a_device, a_bufferMemory, offset, a_bufferSize, 0, &mappedMemory);
            Pixel* pmappedMemory = (Pixel *)mappedMemory;
            for (int j = 0; j < a_width; j += 1)
            {
                ans[i][j].r = pmappedMemory[j].r;
                ans[i][j].g = pmappedMemory[j].g;
                ans[i][j].b = pmappedMemory[j].b;
                ans[i][j].a = pmappedMemory[j].a;
            }
            vkUnmapMemory(a_device, a_bufferMemory);
        }
        return ans;
    }

    static void writeImageToDeviceMemory(VkDevice a_device, VkDeviceMemory a_bufferMemory, size_t a_offset, int a_width, int a_height, std::vector<std::vector<Pixel> > ex)
    {
        const int a_bufferSize = a_width * sizeof(Pixel);
        void* mappedMemory = nullptr;
        for (int i = 0; i < a_height; i += 1) {
            size_t offset = a_offset + i * a_width * sizeof(Pixel);
            mappedMemory = nullptr;
            vkMapMemory(a_device, a_bufferMemory, offset, a_bufferSize, 0, &mappedMemory);
            Pixel* pmappedMemory = (Pixel *)mappedMemory;
            for (int j = 0; j < a_width; j += 1) {
                pmappedMemory[j].r = ex[i][j].r;
                pmappedMemory[j].g = ex[i][j].g;
                pmappedMemory[j].b = ex[i][j].b;
                pmappedMemory[j].a = ex[i][j].a;
            }
            vkUnmapMemory(a_device, a_bufferMemory);
        }
    }

    static void writeImageToDeviceMemory(VkDevice a_device, VkDeviceMemory a_bufferMemory, size_t a_offset, Pixel ex)
    {
        void* mappedMemory = nullptr;
        vkMapMemory(a_device, a_bufferMemory, a_offset, sizeof(Pixel), 0, &mappedMemory);
        Pixel* pmappedMemory = (Pixel *)mappedMemory;
        pmappedMemory[0].r = ex.r;
        pmappedMemory[0].g = ex.g;
        pmappedMemory[0].b = ex.b;
        pmappedMemory[0].a = ex.a;
        vkUnmapMemory(a_device, a_bufferMemory);
    }

    static void clearMemory(VkDevice a_device, VkDeviceMemory a_bufferMemory, size_t a_offset, int a_width, int a_height)
    {
        const int a_bufferSize = a_width * sizeof(Pixel);
        void* mappedMemory = nullptr;
        for (int i = 0; i < a_height; i += 1) {
            size_t offset = a_offset + i * a_width * sizeof(Pixel);
            mappedMemory = nullptr;
            vkMapMemory(a_device, a_bufferMemory, offset, a_bufferSize, 0, &mappedMemory);
            Pixel* pmappedMemory = (Pixel *)mappedMemory;
            for (int j = 0; j < a_width; j += 1) {
                pmappedMemory[j].r = 0;
                pmappedMemory[j].g = 0;
                pmappedMemory[j].b = 0;
                pmappedMemory[j].a = 0;
            }
            vkUnmapMemory(a_device, a_bufferMemory);
        }
    }


    static void saveRenderedImage(std::vector<std::vector<Pixel> > im, int a_width, int a_height, const char *name)
    {
        std::vector<unsigned char> image;
        image.reserve(a_width * a_height * 3);

        for (int i = 0; i < a_height; ++i) {
            for (int j = 0; j < a_width; ++j) {
                image.push_back((unsigned char)(255.0f * (im[i][j].r)));
                image.push_back((unsigned char)(255.0f * (im[i][j].g)));
                image.push_back((unsigned char)(255.0f * (im[i][j].b)));
                // image.push_back((unsigned char)(255.0f * (im[i][j].a)));
            }
        }

        stbi_write_bmp(name, a_width, a_height, 3, (const uint32_t*)image.data());
    }

    static std::vector<std::vector<Pixel> > readBMP(const char *filename, int &a_width, int &a_height)
    {
        int comp;
        unsigned char* image = stbi_load(filename, &a_width, &a_height, &comp, STBI_rgb);
        std::vector<std::vector<Pixel> > res(a_height);
        for (int i = 0; i < a_height; ++i) {
            res[i].resize(a_width);
            for (int j = 0; j < a_width; ++j) {
            res[i][j].r = image[(i * a_width + j) * 3] / 256.0f;
            res[i][j].g = image[(i * a_width + j) * 3 + 1] / 256.0f;
            res[i][j].b = image[(i * a_width + j) * 3 + 2] / 256.0f;
            }
        }
        return res;
    }



    // assume simple pitch-linear data layout and 'a_bufferMemory' to be a mapped memory.
    //
    static void saveRenderedImageFromDeviceMemory(VkDevice a_device, VkDeviceMemory a_bufferMemory, size_t a_offset, int a_width, int a_height, const char *name)
    {
        const int a_bufferSize = a_width * sizeof(Pixel);
        void* mappedMemory = nullptr;
        std::vector<unsigned char> image;
        image.reserve(a_width * a_height * 4);
        for (int i = 0; i < a_height; i += 1) {
            size_t offset = a_offset + i * a_width * sizeof(Pixel);
            mappedMemory = nullptr;
            vkMapMemory(a_device, a_bufferMemory, offset, a_bufferSize, 0, &mappedMemory);
            Pixel* pmappedMemory = (Pixel *)mappedMemory;
            for (int j = 0; j < a_width; j += 1) {
                image.push_back((unsigned char)(255.0f * (pmappedMemory[j].r)));
                image.push_back((unsigned char)(255.0f * (pmappedMemory[j].g)));
                image.push_back((unsigned char)(255.0f * (pmappedMemory[j].b)));
                image.push_back((unsigned char)(255.0f * (pmappedMemory[j].a)));
            }
            vkUnmapMemory(a_device, a_bufferMemory);
        }
        SaveBMP(name, (const uint32_t*)image.data(), WIDTH, HEIGHT);
    }

    static VKAPI_ATTR VkBool32 VKAPI_CALL debugReportCallbackFn(
        VkDebugReportFlagsEXT                       flags,
        VkDebugReportObjectTypeEXT                  objectType,
        uint64_t                                    object,
        size_t                                      location,
        int32_t                                     messageCode,
        const char*                                 pLayerPrefix,
        const char*                                 pMessage,
        void*                                       pUserData)
    {
        printf("Debug Report: %s: %s\n", pLayerPrefix, pMessage);
        return VK_FALSE;
    }


    static void createBuffer(VkDevice a_device, VkPhysicalDevice a_physDevice, const size_t a_bufferSize, VkBuffer* a_pBuffer, VkDeviceMemory* a_pBufferMemory)
    {
        VkBufferCreateInfo bufferCreateInfo = {};
        bufferCreateInfo.sType       = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bufferCreateInfo.size        = a_bufferSize;
        bufferCreateInfo.usage       = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
        bufferCreateInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

        VK_CHECK_RESULT(vkCreateBuffer(a_device, &bufferCreateInfo, NULL, a_pBuffer));
        VkMemoryRequirements memoryRequirements;
        vkGetBufferMemoryRequirements(a_device, (*a_pBuffer), &memoryRequirements);
        VkMemoryAllocateInfo allocateInfo = {};
        allocateInfo.sType           = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        allocateInfo.allocationSize  = memoryRequirements.size;
        allocateInfo.memoryTypeIndex = vk_utils::FindMemoryType(memoryRequirements.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_COHERENT_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT, a_physDevice);

        VK_CHECK_RESULT(vkAllocateMemory(a_device, &allocateInfo, NULL, a_pBufferMemory));
        VK_CHECK_RESULT(vkBindBufferMemory(a_device, (*a_pBuffer), (*a_pBufferMemory), 0));
    }

    static void createDescriptorSetLayout(VkDevice a_device, VkDescriptorSetLayout* a_pDSLayout)
    {
        VkDescriptorSetLayoutBinding descriptorSetLayoutBinding = {};
        descriptorSetLayoutBinding.binding         = 0;
        descriptorSetLayoutBinding.descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        descriptorSetLayoutBinding.descriptorCount = 1;
        descriptorSetLayoutBinding.stageFlags      = VK_SHADER_STAGE_COMPUTE_BIT;

        VkDescriptorSetLayoutCreateInfo descriptorSetLayoutCreateInfo = {};
        descriptorSetLayoutCreateInfo.sType        = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        descriptorSetLayoutCreateInfo.bindingCount = 1;
        descriptorSetLayoutCreateInfo.pBindings    = &descriptorSetLayoutBinding;
        VK_CHECK_RESULT(vkCreateDescriptorSetLayout(a_device, &descriptorSetLayoutCreateInfo, NULL, a_pDSLayout));
    }

    static void createDescriptorSetForOurBuffer(VkDevice a_device, VkBuffer a_buffer, size_t a_bufferSize, const VkDescriptorSetLayout* a_pDSLayout, VkDescriptorPool* a_pDSPool, VkDescriptorSet* a_pDS)
    {
        VkDescriptorPoolSize descriptorPoolSize = {};
        descriptorPoolSize.type            = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        descriptorPoolSize.descriptorCount = 1;

        VkDescriptorPoolCreateInfo descriptorPoolCreateInfo = {};
        descriptorPoolCreateInfo.sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        descriptorPoolCreateInfo.maxSets       = 1;
        descriptorPoolCreateInfo.poolSizeCount = 1;
        descriptorPoolCreateInfo.pPoolSizes    = &descriptorPoolSize;

        VK_CHECK_RESULT(vkCreateDescriptorPool(a_device, &descriptorPoolCreateInfo, NULL, a_pDSPool));
        VkDescriptorSetAllocateInfo descriptorSetAllocateInfo = {};
        descriptorSetAllocateInfo.sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        descriptorSetAllocateInfo.descriptorPool     = (*a_pDSPool);
        descriptorSetAllocateInfo.descriptorSetCount = 1;
        descriptorSetAllocateInfo.pSetLayouts        = a_pDSLayout;
        VK_CHECK_RESULT(vkAllocateDescriptorSets(a_device, &descriptorSetAllocateInfo, a_pDS));
        VkDescriptorBufferInfo descriptorBufferInfo = {};
        descriptorBufferInfo.buffer = a_buffer;
        descriptorBufferInfo.offset = 0;
        descriptorBufferInfo.range  = a_bufferSize;

        VkWriteDescriptorSet writeDescriptorSet = {};
        writeDescriptorSet.sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writeDescriptorSet.dstSet          = (*a_pDS);
        writeDescriptorSet.dstBinding      = 0;
        writeDescriptorSet.descriptorCount = 1;
        writeDescriptorSet.descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writeDescriptorSet.pBufferInfo     = &descriptorBufferInfo;
        vkUpdateDescriptorSets(a_device, 1, &writeDescriptorSet, 0, NULL);
    }

    static void createComputePipeline(VkDevice a_device, const VkDescriptorSetLayout& a_dsLayout, VkShaderModule* a_pShaderModule, VkPipeline* a_pPipeline, VkPipelineLayout* a_pPipelineLayout, const char *s)
    {
        std::vector<uint32_t> code = vk_utils::ReadFile(s);
        VkShaderModuleCreateInfo createInfo = {};
        createInfo.sType    = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        createInfo.pCode    = code.data();
        createInfo.codeSize = code.size()*sizeof(uint32_t);
        VK_CHECK_RESULT(vkCreateShaderModule(a_device, &createInfo, NULL, a_pShaderModule));
        VkPipelineShaderStageCreateInfo shaderStageCreateInfo = {};
        shaderStageCreateInfo.sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        shaderStageCreateInfo.stage  = VK_SHADER_STAGE_COMPUTE_BIT;
        shaderStageCreateInfo.module = (*a_pShaderModule);
        shaderStageCreateInfo.pName  = "main";
        VkPipelineLayoutCreateInfo pipelineLayoutCreateInfo = {};
        pipelineLayoutCreateInfo.sType          = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        pipelineLayoutCreateInfo.setLayoutCount = 1;
        pipelineLayoutCreateInfo.pSetLayouts    = &a_dsLayout;
        VK_CHECK_RESULT(vkCreatePipelineLayout(a_device, &pipelineLayoutCreateInfo, NULL, a_pPipelineLayout));
        VkComputePipelineCreateInfo pipelineCreateInfo = {};
        pipelineCreateInfo.sType  = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
        pipelineCreateInfo.stage  = shaderStageCreateInfo;
        pipelineCreateInfo.layout = (*a_pPipelineLayout);
        VK_CHECK_RESULT(vkCreateComputePipelines(a_device, VK_NULL_HANDLE, 1, &pipelineCreateInfo, NULL, a_pPipeline));
    }

    static void createCommandBuffer(VkDevice a_device, uint32_t queueFamilyIndex, VkPipeline a_pipeline, VkPipelineLayout a_layout, VkCommandPool* a_pool, VkCommandBuffer* a_pCmdBuff)
    {
        VkCommandPoolCreateInfo commandPoolCreateInfo = {};
        commandPoolCreateInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        commandPoolCreateInfo.flags = 0;
        commandPoolCreateInfo.queueFamilyIndex = queueFamilyIndex;
        VK_CHECK_RESULT(vkCreateCommandPool(a_device, &commandPoolCreateInfo, NULL, a_pool));

        VkCommandBufferAllocateInfo commandBufferAllocateInfo = {};
        commandBufferAllocateInfo.sType       = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        commandBufferAllocateInfo.commandPool = (*a_pool);
        commandBufferAllocateInfo.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        commandBufferAllocateInfo.commandBufferCount = 1;
        VK_CHECK_RESULT(vkAllocateCommandBuffers(a_device, &commandBufferAllocateInfo, a_pCmdBuff));
    }

    static void recordCommandsTo(VkCommandBuffer a_cmdBuff, VkPipeline a_pipeline, VkPipelineLayout a_layout, const VkDescriptorSet& a_ds, uint32_t X, uint32_t Y)
    {
        VkCommandBufferBeginInfo beginInfo = {};
        beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        VK_CHECK_RESULT(vkBeginCommandBuffer(a_cmdBuff, &beginInfo));
        vkCmdBindPipeline(a_cmdBuff, VK_PIPELINE_BIND_POINT_COMPUTE, a_pipeline);
        vkCmdBindDescriptorSets(a_cmdBuff, VK_PIPELINE_BIND_POINT_COMPUTE, a_layout, 0, 1, &a_ds, 0, NULL);
        // vkCmdDispatch(a_cmdBuff, (uint32_t)1, (uint32_t)1, 1);
        vkCmdDispatch(a_cmdBuff, X, Y, 1);
        // vkCmdDispatch(a_cmdBuff, (uint32_t)(ceil(a_width / float(WORKGROUP_SIZE))), (uint32_t)(ceil(a_height / float(WORKGROUP_SIZE))), 1);
        // vkCmdDispatch(a_cmdBuff, (uint32_t)(ceil(a_width / float(WORKGROUP_SIZE * WIDTH))), (uint32_t)(ceil(a_height / float(WORKGROUP_SIZE * HEIGHT))), 1);
        // vkCmdDispatch(a_cmdBuff, (uint32_t)ceil(a_width / float(sqr(WORKGROUP_SIZE))), (uint32_t)ceil(a_height / (float(sqr(WORKGROUP_SIZE)))), 1);

        VK_CHECK_RESULT(vkEndCommandBuffer(a_cmdBuff));
    }


    static void runCommandBuffer(VkCommandBuffer a_cmdBuff, VkQueue a_queue, VkDevice a_device)
    {
        VkSubmitInfo submitInfo = {};
        submitInfo.sType              = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submitInfo.commandBufferCount = 1;
        submitInfo.pCommandBuffers    = &a_cmdBuff;

        VkFence fence;
        VkFenceCreateInfo fenceCreateInfo = {};
        fenceCreateInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        fenceCreateInfo.flags = 0;
        VK_CHECK_RESULT(vkCreateFence(a_device, &fenceCreateInfo, NULL, &fence));
        VK_CHECK_RESULT(vkQueueSubmit(a_queue, 1, &submitInfo, fence));
        VK_CHECK_RESULT(vkWaitForFences(a_device, 1, &fence, VK_TRUE, 100000000000));

        vkDestroyFence(a_device, fence, NULL);
    }

    void cleanup() {

        if (enableValidationLayers) {
            auto func = (PFN_vkDestroyDebugReportCallbackEXT)vkGetInstanceProcAddr(instance, "vkDestroyDebugReportCallbackEXT");
            if (func == nullptr) {
                throw std::runtime_error("Could not load vkDestroyDebugReportCallbackEXT");
            }
            func(instance, debugReportCallback, NULL);
        }

        vkFreeMemory(device, bufferMemory, NULL);
        vkDestroyBuffer(device, buffer, NULL);	
        //vkDestroyShaderModule(device, computeShaderModule, NULL);
        vkDestroyDescriptorPool(device, descriptorPool, NULL);
        vkDestroyDescriptorSetLayout(device, descriptorSetLayout, NULL);
        //vkDestroyPipelineLayout(device, pipelineLayout, NULL);
        //vkDestroyPipeline(device, pipeline, NULL);
        //vkDestroyCommandPool(device, commandPool, NULL);	
        vkDestroyDevice(device, NULL);
        vkDestroyInstance(instance, NULL);		
    }
};

int main()
{
    ComputeApplication app;

    try
    {
        app.run();
    }
    catch (const std::runtime_error& e)
    {
        printf("%s\n", e.what());
        return EXIT_FAILURE;
    }
        
    return EXIT_SUCCESS;
}
