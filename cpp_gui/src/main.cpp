#include <windows.h>
#include <commdlg.h>
#include <shlwapi.h>
#include <commctrl.h>
#include <shlobj.h>
#include <string>
#include <vector>
#include <thread>
#include <atomic>
#include <filesystem>
#include <chrono>
#include <mutex>
#include <optional>
#include <cwchar>

// TorchScript
#include <torch/script.h>
#include <torch/cuda.h>
#include <ATen/ATen.h>
#include <torch/nn/functional/upsampling.h>

// OpenCV for image IO and preprocessing
#include <opencv2/opencv.hpp>

// Link with Common Controls v6
#pragma comment(lib, "Comctl32.lib")
#pragma comment(lib, "Shlwapi.lib")

// Simple app to mirror Python's create_pbr.py inference flow with minimal GUI
// - Buttons: Input directory, Output directory, Run (disabled during processing)
// - Progress bar: based on processed/total pairs
// - Uses TorchScript models (.pt traced/scripted) provided by the user
// - Implements tiling for albedo and padding/downscaling for PBR maps
// - Generates poisson-coarse and mean-curvature from normal

// Controls IDs
#define IDC_BTN_INPUT 101
#define IDC_BTN_OUTPUT 102
#define IDC_BTN_RUN 103
#define IDC_PROGRESS 104
#define IDC_STATUS 105

static HINSTANCE g_hInst;
static HWND g_hWnd;
static HWND g_btnInput, g_btnOutput, g_btnRun, g_progress, g_status;
static std::wstring g_inputDir, g_outputDir;

static std::atomic<bool> g_isRunning{false};
static std::atomic<int> g_progressValue{0};
static std::atomic<int> g_progressMax{100};
static std::mutex g_logMutex;
static bool g_consoleReady = false;

// Console logging helpers
void InitConsole()
{
    if (g_consoleReady)
        return;
    if (AllocConsole())
    {
        // Optional: set a title
        SetConsoleTitleW(L"AI PBR GUI Log");
        g_consoleReady = true;
    }
}

void LogMsg(const std::wstring &msg, int indent = 0)
{
    if (!g_consoleReady)
        return;
    std::scoped_lock lk(g_logMutex);
    HANDLE h = GetStdHandle(STD_OUTPUT_HANDLE);
    if (h == NULL || h == INVALID_HANDLE_VALUE)
        return;
    std::wstring line;
    if (indent > 0)
        line.assign(indent * 2, L' ');
    line += msg;
    DWORD written = 0;
    WriteConsoleW(h, line.c_str(), (DWORD)line.size(), &written, NULL);
    WriteConsoleW(h, L"\r\n", 2, &written, NULL);
}

// Config similar to Python
static int MAX_TILE_SIZE = 2048; // adjustable

// Fatal error helper: show message and exit
void ShowFatalAndExit(const std::wstring &msg)
{
    MessageBoxW(g_hWnd ? g_hWnd : nullptr, msg.c_str(), L"Error", MB_OK | MB_ICONERROR);
    // Ask main window to close then terminate
    if (g_hWnd)
        PostMessageW(g_hWnd, WM_CLOSE, 0, 0);
    ExitProcess(1);
}

// Simple temp directory helper
std::filesystem::path GetTempDir()
{
    wchar_t buf[MAX_PATH];
    DWORD n = GetTempPathW(MAX_PATH, buf);
    if (n == 0 || n > MAX_PATH)
        return std::filesystem::temp_directory_path();
    return std::filesystem::path(buf);
}

void RunTexconv(const std::wstring &cmd)
{
    SECURITY_ATTRIBUTES sa;
    sa.nLength = sizeof(SECURITY_ATTRIBUTES);
    sa.bInheritHandle = TRUE;
    sa.lpSecurityDescriptor = NULL;

    HANDLE hStdoutRead, hStdoutWrite;
    HANDLE hStderrRead, hStderrWrite;

    if (!CreatePipe(&hStdoutRead, &hStdoutWrite, &sa, 0) ||
        !CreatePipe(&hStderrRead, &hStderrWrite, &sa, 0))
    {
        throw std::runtime_error("Failed to create pipes");
    }

    SetHandleInformation(hStdoutRead, HANDLE_FLAG_INHERIT, 0);
    SetHandleInformation(hStderrRead, HANDLE_FLAG_INHERIT, 0);

    STARTUPINFOW si = {};
    si.cb = sizeof(STARTUPINFOW);
    si.hStdOutput = hStdoutWrite;
    si.hStdError = hStderrWrite;
    si.dwFlags = STARTF_USESTDHANDLES;

    PROCESS_INFORMATION pi = {};

    std::wstring cmdCopy = cmd; // CreateProcess may modify the string
    LogMsg(L"[texconv] " + cmdCopy, 2);
    if (!CreateProcessW(NULL, &cmdCopy[0], NULL, NULL, TRUE, CREATE_NO_WINDOW, NULL, NULL, &si, &pi))
    {
        CloseHandle(hStdoutRead);
        CloseHandle(hStdoutWrite);
        CloseHandle(hStderrRead);
        CloseHandle(hStderrWrite);
        throw std::runtime_error("Failed to start texconv process");
    }

    CloseHandle(hStdoutWrite);
    CloseHandle(hStderrWrite);

    // Read stdout and stderr
    std::string stdout_output, stderr_output;
    char buffer[4096];
    DWORD bytesRead;

    while (ReadFile(hStdoutRead, buffer, sizeof(buffer), &bytesRead, NULL) && bytesRead > 0)
    {
        stdout_output.append(buffer, bytesRead);
    }

    while (ReadFile(hStderrRead, buffer, sizeof(buffer), &bytesRead, NULL) && bytesRead > 0)
    {
        stderr_output.append(buffer, bytesRead);
    }

    WaitForSingleObject(pi.hProcess, INFINITE);

    DWORD exitCode;
    GetExitCodeProcess(pi.hProcess, &exitCode);

    CloseHandle(pi.hProcess);
    CloseHandle(pi.hThread);
    CloseHandle(hStdoutRead);
    CloseHandle(hStderrRead);

    if (exitCode != 0)
    {
        std::string error = "texconv failed with exit code " + std::to_string(exitCode);
        if (!stderr_output.empty())
            error += "\nStderr: " + stderr_output;
        if (!stdout_output.empty())
            error += "\nStdout: " + stdout_output;
        throw std::runtime_error(error);
    }
    else
    {
        LogMsg(L"[texconv] done", 2);
    }
}

// Resolve texconv.exe or exit fatally
std::filesystem::path GetTexconvOrExit()
{
    std::filesystem::path texconv = std::filesystem::current_path() / L"texconv.exe";
    if (!std::filesystem::exists(texconv))
        texconv = std::filesystem::path(L"texconv.exe");
    if (!std::filesystem::exists(texconv))
        throw std::runtime_error("texconv.exe not found. Place it next to the executable or in the working directory.");
    return texconv;
}

// Convert DDS to PNG via texconv.exe using proper color space; returns path to PNG; throws on failure
// isSRGB=true for diffuse, false for normals (linear)
std::filesystem::path EnsureReadableImage(const std::filesystem::path &inPath, bool isSRGB)
{
    auto ext = inPath.extension().wstring();
    for (auto &ch : ext)
        ch = (wchar_t)towlower(ch);
    if (ext != L".dds")
    {
        LogMsg(L"Using image as-is: " + inPath.wstring(), 1);
        return inPath;
    }

    auto texconv = GetTexconvOrExit();
    auto tmpDir = GetTempDir() / L"ai_pbr_texconv";
    std::error_code ec;
    std::filesystem::create_directories(tmpDir, ec);

    // Build command similar to Python:
    // Diffuse (sRGB): -f R8G8B8A8_UNORM_SRGB --srgb-in --srgb-out
    // Normal  (linear): -f R8G8B8A8_UNORM
    std::wstring cmd = L"\"" + texconv.wstring() + L"\" -nologo -ft png -y -o \"" + tmpDir.wstring() + L"\" ";
    if (isSRGB)
        cmd += L"-f R8G8B8A8_UNORM_SRGB --srgb-in --srgb-out ";
    else
        cmd += L"-f R8G8B8A8_UNORM ";
    cmd += L"\"" + inPath.wstring() + L"\"";

    LogMsg(std::wstring(L"Converting DDS to PNG [") + (isSRGB ? L"sRGB" : L"linear") + L"]: " + inPath.wstring(), 1);
    RunTexconv(cmd);

    auto png = tmpDir / (inPath.stem().wstring() + L".png");
    if (!std::filesystem::exists(png))
        throw std::runtime_error("texconv did not produce expected PNG output");
    LogMsg(L"Converted: " + png.wstring(), 1);
    return png;
}

// Check if diffuse has meaningful (non-white) alpha
bool HasNonWhiteAlpha(const cv::Mat &img)
{
    if (img.empty() || img.channels() != 4)
        return false;
    std::vector<cv::Mat> ch;
    cv::split(img, ch);
    double minv = 0, maxv = 0;
    cv::minMaxLoc(ch[3], &minv, &maxv);
    // If both min and max are 255, alpha is all white
    return !(minv == 255.0 && maxv == 255.0);
}

// Helper: show folder picker
std::optional<std::wstring> PickFolder(HWND owner)
{
    BROWSEINFOW bi{};
    bi.hwndOwner = owner;
    bi.ulFlags = BIF_RETURNONLYFSDIRS | BIF_NEWDIALOGSTYLE;
    bi.lpszTitle = L"Select a folder";
    PIDLIST_ABSOLUTE pidl = SHBrowseForFolderW(&bi);
    if (!pidl)
        return std::nullopt;
    wchar_t path[MAX_PATH];
    if (!SHGetPathFromIDListW(pidl, path))
        return std::nullopt;
    return std::wstring(path);
}

void SetStatus(const std::wstring &msg)
{
    SendMessageW(g_status, WM_SETTEXT, 0, (LPARAM)msg.c_str());
}

// Torch helpers
at::Tensor toTensorFromCV(const cv::Mat &img)
{
    // Expect CV_8UC3 or CV_8UC4. Convert to RGB 3 channels.
    cv::Mat rgb;
    if (img.channels() == 4)
    {
        cv::cvtColor(img, rgb, cv::COLOR_BGRA2RGB);
    }
    else if (img.channels() == 3)
    {
        cv::cvtColor(img, rgb, cv::COLOR_BGR2RGB);
    }
    else if (img.channels() == 1)
    {
        cv::cvtColor(img, rgb, cv::COLOR_GRAY2RGB);
    }
    else
    {
        rgb = img.clone();
    }
    rgb.convertTo(rgb, CV_32F, 1.0 / 255.0);
    auto tensor = torch::from_blob(rgb.data, {rgb.rows, rgb.cols, 3}, torch::kFloat32).clone();
    tensor = tensor.permute({2, 0, 1}); // C,H,W
    return tensor;
}

cv::Mat toImageFromTensor(const at::Tensor &t)
{
    // t shape: (C,H,W), float [0,1]
    auto t_cpu = t.detach().to(torch::kCPU).contiguous();
    int C = (int)t_cpu.size(0), H = (int)t_cpu.size(1), W = (int)t_cpu.size(2);
    at::Tensor t_hwc = t_cpu.permute({1, 2, 0}).clamp(0.0, 1.0);
    cv::Mat img(H, W, CV_32FC(C));
    std::memcpy(img.data, t_hwc.data_ptr<float>(), sizeof(float) * C * H * W);
    cv::Mat u8;
    img.convertTo(u8, CV_8UC(C), 255.0);
    // Convert RGB->BGR for saving
    cv::Mat bgr;
    if (C == 3)
        cv::cvtColor(u8, bgr, cv::COLOR_RGB2BGR);
    else
        bgr = u8;
    return bgr;
}

// Normalization IMAGENET_STANDARD/DEFAULT (mean/std in [0,1])
static const float IMAGENET_STANDARD_MEAN[3] = {0.5f, 0.5f, 0.5f};
static const float IMAGENET_STANDARD_STD[3] = {0.5f, 0.5f, 0.5f};
static const float IMAGENET_DEFAULT_MEAN[3] = {0.485f, 0.456f, 0.406f};
static const float IMAGENET_DEFAULT_STD[3] = {0.229f, 0.224f, 0.225f};

at::Tensor normalizeCHW(const at::Tensor &chw, const float mean[3], const float stdv[3])
{
    auto t = chw.clone();
    for (int c = 0; c < 3; ++c)
    {
        t[c] = (t[c] - mean[c]) / stdv[c];
    }
    return t;
}

// Normal map renormalization: input BGR(A)/GRAY uint8 -> drop alpha, convert to RGB for math,
// normalize each vector to unit length, then return 0..255 8-bit 3-channel image.
// Note: We return BGR (OpenCV convention) to keep downstream conversions correct.
cv::Mat normalizeNormalImage(const cv::Mat &normalBGR)
{
    // 1) Convert to RGB 3-channel and drop alpha if present
    cv::Mat rgb;
    if (normalBGR.channels() == 4)
        cv::cvtColor(normalBGR, rgb, cv::COLOR_BGRA2RGB);
    else if (normalBGR.channels() == 3)
        cv::cvtColor(normalBGR, rgb, cv::COLOR_BGR2RGB);
    else
        cv::cvtColor(normalBGR, rgb, cv::COLOR_GRAY2RGB);

    // 2) To float [0,1]
    cv::Mat f;
    rgb.convertTo(f, CV_32FC3, 1.0 / 255.0);

    // 3) Map to [-1,1]
    cv::Mat vec = f * 2.0f - 1.0f;

    // 4) Normalize each pixel vector length to 1 (include Z in length)
    std::vector<cv::Mat> ch(3);
    cv::split(vec, ch);
    cv::Mat lenXY;
    cv::magnitude(ch[0], ch[1], lenXY);
    cv::Mat z2;
    cv::multiply(ch[2], ch[2], z2);
    cv::Mat xy2;
    cv::multiply(lenXY, lenXY, xy2);
    cv::Mat sum;
    cv::add(xy2, z2, sum);
    cv::Mat len;
    cv::sqrt(sum, len);
    len += 1e-6f; // avoid div-by-zero
    for (int i = 0; i < 3; ++i)
        ch[i] = ch[i].mul(1.0f / len);
    cv::merge(ch, vec);

    // 5) Back to [0,1] then to 0..255 u8
    cv::Mat vec01 = (vec + 1.0f) * 0.5f;
    cv::Mat vec01_clamped;
    cv::min(cv::max(vec01, 0.0f), 1.0f, vec01_clamped);
    cv::Mat rgb_u8;
    vec01_clamped.convertTo(rgb_u8, CV_8UC3, 255.0);

    // We return BGR (convert back) to match OpenCV's default expectations elsewhere in the pipeline
    cv::Mat bgr_u8;
    cv::cvtColor(rgb_u8, bgr_u8, cv::COLOR_RGB2BGR);
    return bgr_u8;
}

// Simple median blur using OpenCV
cv::Mat medianBlur3(const cv::Mat &src)
{
    cv::Mat out;
    cv::medianBlur(src, out, 3);
    return out;
}

// Spatial gradient approximations (Sobel)
void sobelXY(const cv::Mat &src, cv::Mat &gx, cv::Mat &gy)
{
    cv::Sobel(src, gx, CV_32F, 1, 0, 3);
    cv::Sobel(src, gy, CV_32F, 0, 1, 3);
}

// Mean curvature approximation similar to Python mean_curvature_map
// The Python code applies this to an ImageNet-standardized normal; we mirror that directly.
// input: normal tensor (3,H,W)
at::Tensor meanCurvature(at::Tensor normalCHW)
{
    auto H = (int)normalCHW.size(1);
    auto W = (int)normalCHW.size(2);
    // to HxWx3
    at::Tensor nhwc = normalCHW.permute({1, 2, 0}).contiguous();
    cv::Mat n(H, W, CV_32FC3, nhwc.data_ptr<float>());
    cv::Mat n_blur = medianBlur3(n);

    std::vector<cv::Mat> ch(3);
    cv::split(n_blur, ch);
    cv::Mat dnx_dx, tmp; // gx for x-channel
    sobelXY(ch[0], dnx_dx, tmp);
    cv::Mat dny_dy;
    sobelXY(ch[1], tmp, dny_dy);

    cv::Mat curv = 0.5f * (dnx_dx + dny_dy);
    curv = cv::abs(curv);

    // 99th percentile
    std::vector<float> vals;
    vals.reserve(H * W);
    vals.assign((float *)curv.datastart, (float *)curv.dataend);
    std::nth_element(vals.begin(), vals.begin() + (size_t)(0.99 * vals.size()), vals.end());
    float p99 = std::max(1e-6f, vals[(size_t)(0.99 * vals.size())]);

    curv = curv * (1.0f / p99);
    cv::threshold(curv, curv, 1.0, 1.0, cv::THRESH_TRUNC);
    curv = (curv - 0.5f) * 2.0f; // [-1,1]

    at::Tensor out = torch::from_blob(curv.data, {1, H, W}, torch::kFloat32).clone();
    out = out.unsqueeze(0); // (1,1,H,W)
    return out;
}

// Poisson coarse height from normals (OpenCV + FFT via dft). Mirrors Python using standardized normals.
at::Tensor poissonCoarseFromNormal(at::Tensor normalCHW /*(3,H,W)*/)
{
    int H = (int)normalCHW.size(1);
    int W = (int)normalCHW.size(2);
    // to HxWx3 float
    auto nhwc = normalCHW.permute({1, 2, 0}).contiguous();
    cv::Mat n(H, W, CV_32FC3, nhwc.data_ptr<float>());
    cv::Mat n_med = medianBlur3(n);

    std::vector<cv::Mat> ch(3);
    cv::split(n_med, ch);
    cv::Mat nz = ch[2].clone();
    cv::threshold(nz, nz, 1e-3, 1.0, cv::THRESH_TOZERO);
    cv::Mat gx = ch[0] / nz;
    cv::Mat gy = ch[1] / nz;

    float clip_val = 10.0f;
    cv::threshold(gx, gx, clip_val, clip_val, cv::THRESH_TRUNC);
    cv::threshold(gx, gx, -clip_val, -clip_val, cv::THRESH_TRUNC);
    cv::threshold(gy, gy, clip_val, clip_val, cv::THRESH_TRUNC);
    cv::threshold(gy, gy, -clip_val, -clip_val, cv::THRESH_TRUNC);

    // Compute divergence in Fourier domain using DFT grids.
    // Build frequency grids fx, fy using OpenCV (normalized frequency).
    cv::Mat Fx(1, W, CV_32F), Fy(H, 1, CV_32F);
    for (int x = 0; x < W; ++x)
        Fx.at<float>(0, x) = (x <= W / 2) ? (float)x / W : (float)(x - W) / W;
    for (int y = 0; y < H; ++y)
        Fy.at<float>(y, 0) = (y <= H / 2) ? (float)y / H : (float)(y - H) / H;

    // DFT of gx, gy
    cv::Mat Gx, Gy;
    cv::dft(gx, Gx, cv::DFT_COMPLEX_OUTPUT);
    cv::dft(gy, Gy, cv::DFT_COMPLEX_OUTPUT);

    // Prepare 2-channel complex Mats for frequency multiplications
    cv::Mat Hx(H, W, CV_32FC2), Hy(H, W, CV_32FC2);
    for (int y = 0; y < H; ++y)
    {
        float fy = Fy.at<float>(y, 0);
        for (int x = 0; x < W; ++x)
        {
            float fx = Fx.at<float>(0, x);
            float a = 2.0f * 3.1415926535f;
            // (i * a * fx) and (i * a * fy)
            Hx.at<cv::Vec2f>(y, x) = cv::Vec2f(0.0f, a * fx);
            Hy.at<cv::Vec2f>(y, x) = cv::Vec2f(0.0f, a * fy);
        }
    }

    cv::Mat iGx, iGy; // i*2*pi*fx * FFT(gx) etc.
    cv::mulSpectrums(Hx, Gx, iGx, 0);
    cv::mulSpectrums(Hy, Gy, iGy, 0);
    cv::Mat Div;
    cv::add(iGx, iGy, Div);

    // Denominator (2*pi)^2*(fx^2+fy^2)
    cv::Mat Den(H, W, CV_32FC2);
    for (int y = 0; y < H; ++y)
    {
        float fy = Fy.at<float>(y, 0);
        for (int x = 0; x < W; ++x)
        {
            float fx = Fx.at<float>(0, x);
            float denom = (2.0f * 3.1415926535f) * (2.0f * 3.1415926535f) * (fx * fx + fy * fy);
            if (x == 0 && y == 0)
                denom = std::numeric_limits<float>::infinity();
            Den.at<cv::Vec2f>(y, x) = cv::Vec2f(denom, 0.0f);
        }
    }

    cv::Mat Hspec;
    // Div / Den (complex division)
    Hspec.create(H, W, CV_32FC2);
    for (int y = 0; y < H; ++y)
    {
        for (int x = 0; x < W; ++x)
        {
            cv::Vec2f a = Div.at<cv::Vec2f>(y, x);
            cv::Vec2f b = Den.at<cv::Vec2f>(y, x);
            float denom = b[0] * b[0] + b[1] * b[1];
            if (denom < 1e-12f)
            {
                Hspec.at<cv::Vec2f>(y, x) = cv::Vec2f(0.0f, 0.0f);
            }
            else
            {
                // (a/b) for complex numbers (b is real)
                Hspec.at<cv::Vec2f>(y, x) = cv::Vec2f(a[0] / b[0], a[1] / b[0]);
            }
        }
    }

    cv::Mat h;
    cv::dft(Hspec, h, cv::DFT_INVERSE | cv::DFT_SCALE | cv::DFT_REAL_OUTPUT);

    double minv, maxv;
    cv::minMaxLoc(h, &minv, &maxv);
    cv::Mat h01 = (h - minv) / std::max(1e-6, (maxv - minv));

    // clamp 1..99 percentile
    cv::Mat sorted;
    h01.reshape(1, 1).copyTo(sorted);
    cv::sort(sorted, sorted, cv::SORT_EVERY_ROW + cv::SORT_ASCENDING);
    int total = sorted.cols;
    float p1 = sorted.at<float>(0, std::max(0, (int)(0.01 * total) - 1));
    float p99 = sorted.at<float>(0, std::max(0, (int)(0.99 * total) - 1));
    cv::Mat hclamped;
    cv::threshold(h01, hclamped, p1, p1, cv::THRESH_TOZERO);
    cv::threshold(hclamped, hclamped, p99, p99, cv::THRESH_TRUNC);
    h01 = (hclamped - p1) / std::max(1e-6f, (p99 - p1));

    // gaussian blur
    cv::Mat blurred;
    cv::GaussianBlur(h01, blurred, cv::Size(15, 15), 5.0, 5.0);

    cv::Mat out = (blurred - 0.5f) * 2.0f; // [-1,1]
    at::Tensor t = torch::from_blob(out.data, {1, 1, H, W}, torch::kFloat32).clone();
    return t;
}

// Tiling for albedo similar to Python predict_albedo
at::Tensor runAlbedoTiled(torch::jit::Module &unet_albedo,
                          torch::jit::Module &segformer,
                          const cv::Mat &diffuse,
                          const cv::Mat &normal,
                          torch::Device device)
{
    auto diffuseCHW = toTensorFromCV(diffuse); // [0,1] [C,H,W]
    auto normalCHW = toTensorFromCV(normal);   // [0,1] [C,H,W]
    diffuseCHW = normalizeCHW(diffuseCHW, IMAGENET_STANDARD_MEAN, IMAGENET_STANDARD_STD);
    normalCHW = normalizeCHW(normalCHW, IMAGENET_STANDARD_MEAN, IMAGENET_STANDARD_STD);

    int H = (int)diffuseCHW.size(1);
    int W = (int)diffuseCHW.size(2);
    int tile = std::min(H, W);
    tile = std::min(tile, MAX_TILE_SIZE);

    int overlap = 0;
    std::vector<int> xs{0}, ys{0};
    bool useTiling = (W != H) || (tile < std::min(H, W));
    LogMsg(L"Albedo stage: " + std::to_wstring(W) + L"x" + std::to_wstring(H) + L", tile=" + std::to_wstring(tile) + (useTiling ? L" (tiling)" : L" (single tile)"), 1);
    if (useTiling)
    {
        overlap = tile / 8;
        int stride = tile - overlap;
        xs.clear();
        ys.clear();
        for (int x = 0; x <= W - tile; x += stride)
            xs.push_back(x);
        if (xs.empty() || xs.back() + tile < W)
            xs.push_back(W - tile);
        for (int y = 0; y <= H - tile; y += stride)
            ys.push_back(y);
        if (ys.empty() || ys.back() + tile < H)
            ys.push_back(H - tile);
    }

    at::Tensor out = torch::zeros({1, 3, H, W});
    at::Tensor weight = torch::zeros_like(out);

    at::Tensor window1d = torch::ones({tile});
    if (overlap > 0)
    {
        at::Tensor ramp = torch::linspace(0, 1, overlap);
        window1d.index_put_({torch::indexing::Slice(0, overlap)}, ramp);
        window1d.index_put_({torch::indexing::Slice(tile - overlap, tile)}, ramp.flip(0));
    }
    at::Tensor w2d = window1d.unsqueeze(0).unsqueeze(0).unsqueeze(-1) * window1d.unsqueeze(0).unsqueeze(0).unsqueeze(0);

    for (int y : ys)
    {
        for (int x : xs)
        {
            auto tile_img = diffuseCHW.index({torch::indexing::Slice(), torch::indexing::Slice(y, y + tile), torch::indexing::Slice(x, x + tile)}).unsqueeze(0).to(device);
            auto tile_nrm = normalCHW.index({torch::indexing::Slice(), torch::indexing::Slice(y, y + tile), torch::indexing::Slice(x, x + tile)}).unsqueeze(0).to(device);
            auto albedo_in = torch::cat({tile_img, tile_nrm}, 1);

            // First pass (handle 1 or 2 args)
            at::Tensor pred1;
            try
            {
                pred1 = unet_albedo.forward({albedo_in, torch::IValue()}).toTensor();
            }
            catch (...)
            {
                pred1 = unet_albedo.forward({albedo_in}).toTensor();
            }
            // Normalize for segformer default stats
            for (int c = 0; c < 3; ++c)
            {
                auto ch = pred1.index({0, c, torch::indexing::Slice(), torch::indexing::Slice()});
                ch = (ch - IMAGENET_DEFAULT_MEAN[c]) / IMAGENET_DEFAULT_STD[c];
                pred1.index_put_({0, c, torch::indexing::Slice(), torch::indexing::Slice()}, ch);
            }
            auto seg_in = torch::cat({pred1, tile_nrm}, 1);
            // Handle segformer outputs flexibly: tuple (logits, last_hidden)
            auto tup = segformer.forward({seg_in}).toTuple();
            at::Tensor seg_feats = tup->elements()[1].toTensor();

            if (!seg_feats.defined())
            {
                ShowFatalAndExit(L"Segformer hidden states are required but not available (albedo stage).");
            }

            at::Tensor pred = unet_albedo.forward({albedo_in, seg_feats}).toTensor();
            pred = pred.to(torch::kCPU);

            if (overlap > 0)
            {
                auto w = w2d.to(torch::kCPU).to(pred.dtype());
                out.index_put_({0, torch::indexing::Slice(), torch::indexing::Slice(y, y + tile), torch::indexing::Slice(x, x + tile)},
                               out.index({0, torch::indexing::Slice(), torch::indexing::Slice(y, y + tile), torch::indexing::Slice(x, x + tile)}) + pred * w);
                weight.index_put_({0, torch::indexing::Slice(), torch::indexing::Slice(y, y + tile), torch::indexing::Slice(x, x + tile)},
                                  weight.index({0, torch::indexing::Slice(), torch::indexing::Slice(y, y + tile), torch::indexing::Slice(x, x + tile)}) + w);
            }
            else
            {
                out.index_put_({0, torch::indexing::Slice(), torch::indexing::Slice(y, y + tile), torch::indexing::Slice(x, x + tile)}, pred);
                weight.index_put_({0, torch::indexing::Slice(), torch::indexing::Slice(y, y + tile), torch::indexing::Slice(x, x + tile)}, 1.0);
            }
        }
    }

    if (overlap > 0)
        out = out / weight.clamp_min(1e-6);
    LogMsg(L"Albedo predicted", 1);
    return out.squeeze(0);
}

struct Models
{
    torch::jit::Module &segformer;
    torch::jit::Module &unetAlbedo;
    torch::jit::Module &unetParallax;
    torch::jit::Module &unetAO;
    torch::jit::Module &unetMetallic;
    torch::jit::Module &unetRoughness;
};

// Run PBR maps stage (padding, scaling, masks)
void runPBR(const Models &M,
            const cv::Mat &albedoOrig,
            const cv::Mat &normalOrig,
            torch::Device device,
            cv::Mat &outParallax, cv::Mat &outAO, cv::Mat &outMetallic, cv::Mat &outRoughness)
{
    auto albedo = albedoOrig.clone();
    auto normal = normalOrig.clone();

    int origW = albedo.cols;
    int origH = albedo.rows;

    // Downscale to <=1024 max-side, then enforce MAX_TILE_SIZE cap (no tiling here)
    int downsampleFactor = 1;

    int albedoH = albedo.rows, albedoW = albedo.cols;
    if (std::max(albedoW, albedoH) > 1024)
    {
        cv::resize(albedo, albedo, cv::Size(), 0.5, 0.5, cv::INTER_LANCZOS4);
        downsampleFactor *= 2;
        LogMsg(L"Downscale albedo x0.5 (<=1024) -> " + std::to_wstring(albedo.cols) + L"x" + std::to_wstring(albedo.rows), 2);
    }
    albedoH = albedo.rows, albedoW = albedo.cols;
    if (std::max(albedoW, albedoH) > MAX_TILE_SIZE)
    {
        int max = std::max(albedoW, albedoH);
        int factor = max / MAX_TILE_SIZE;
        if (factor < 1)
            factor = 1;

        int newH = albedoH / factor;
        int newW = albedoW / factor;

        cv::resize(albedo, albedo, cv::Size(newW, newH), 0, 0, cv::INTER_LANCZOS4);
        downsampleFactor *= factor;
        LogMsg(L"Downscale albedo /" + std::to_wstring(factor) + L" (<=MAX_TILE_SIZE) -> " + std::to_wstring(newW) + L"x" + std::to_wstring(newH), 2);
    }

    if (albedo.rows != normal.rows || albedo.cols != normal.cols)
    {
        cv::resize(normal, normal, albedo.size(), 0, 0, cv::INTER_LINEAR);
        normal = normalizeNormalImage(normal);
        LogMsg(L"Resize normal to match albedo -> " + std::to_wstring(normal.cols) + L"x" + std::to_wstring(normal.rows) + L" and renormalize", 2);
    }

    // Pad to square if needed (right/bottom)
    auto pad = [&](const at::Tensor &chw)
    {
        int H = (int)chw.size(1), W = (int)chw.size(2);
        int side = std::max(H, W);

        if (H == W)
            return chw;
        at::Tensor padded = torch::zeros({chw.size(0), side, side}, chw.options());
        padded.index_put_({torch::indexing::Slice(), torch::indexing::Slice(0, H), torch::indexing::Slice(0, W)}, chw);
        return padded;
    };

    auto albedoCHW = toTensorFromCV(albedo);
    auto normalCHW = toTensorFromCV(normal);

    albedoCHW = pad(albedoCHW);
    normalCHW = pad(normalCHW);

    auto albedoDefaultCHW = albedoCHW.clone();

    albedoCHW = normalizeCHW(albedoCHW, IMAGENET_STANDARD_MEAN, IMAGENET_STANDARD_STD);
    normalCHW = normalizeCHW(normalCHW, IMAGENET_STANDARD_MEAN, IMAGENET_STANDARD_STD);
    albedoDefaultCHW = normalizeCHW(albedoDefaultCHW, IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD);

    // Curvature and Poisson directly from standardized normals (to mirror Python)
    auto Poisson = poissonCoarseFromNormal(normalCHW); // (1,1,H,W)
    auto Curv = meanCurvature(normalCHW);              // (1,1,H,W)
    LogMsg(L"Computed Poisson coarse height and mean curvature", 2);

    // Segformer features and mask (robust to different TS export styles)
    auto seg_in = torch::cat({albedoDefaultCHW.unsqueeze(0).to(device), normalCHW.unsqueeze(0).to(device)}, 1);

    auto tup = M.segformer.forward({seg_in}).toTuple();
    auto elems = tup->elements();
    at::Tensor logits = elems[0].toTensor();
    at::Tensor last_hidden = elems[1].toTensor();

    logits = torch::nn::functional::interpolate(
        logits,
        torch::nn::functional::InterpolateFuncOptions().size(std::vector<int64_t>{albedoDefaultCHW.size(1), albedoDefaultCHW.size(2)}).mode(torch::kBilinear).align_corners(false));
    auto seg_probs = torch::softmax(logits, 1);
    auto max_vals_and_idx = torch::max(seg_probs, 1, true);
    auto max_probs = std::get<0>(max_vals_and_idx);
    auto max_indices = std::get<1>(max_vals_and_idx);
    auto one_hot = torch::zeros_like(seg_probs);
    one_hot = one_hot.scatter(1, max_indices, 1);
    float thresh = 0.5f;
    auto high_conf = (max_probs > thresh).to(seg_probs.dtype());
    at::Tensor final_mask = one_hot * high_conf;

    auto parallax_ao_in = torch::cat({normalCHW.unsqueeze(0).to(device), Curv.to(device), Poisson.to(device)}, 1);

    at::Tensor parallax = M.unetParallax.forward({parallax_ao_in, last_hidden}).toTensor();
    parallax = parallax.to(torch::kCPU);

    at::Tensor ao = M.unetAO.forward({parallax_ao_in, last_hidden}).toTensor();
    ao = ao.to(torch::kCPU);

    auto metal_in = torch::cat({albedoCHW.unsqueeze(0).to(device), normalCHW.unsqueeze(0).to(device), final_mask}, 1);

    at::Tensor metallic = M.unetMetallic.forward({metal_in, last_hidden}).toTensor();
    metallic = metallic.to(torch::kCPU);

    at::Tensor roughness = M.unetRoughness.forward({metal_in, last_hidden}).toTensor();
    roughness = roughness.to(torch::kCPU);
    LogMsg(L"Predicted PBR maps: parallax, AO, metallic, roughness", 2);

    auto to_u8_gray = [&](const at::Tensor &t1c)
    {
        auto s = torch::sigmoid(t1c).squeeze(0).clamp(0, 1);
        return toImageFromTensor(s);
    };

    outParallax = to_u8_gray(parallax);
    outAO = to_u8_gray(ao);
    outMetallic = to_u8_gray(metallic);
    outRoughness = to_u8_gray(roughness);

    // Crop to original size divided by downsampleFactor
    int cropW = std::max(1, origW / downsampleFactor);
    int cropH = std::max(1, origH / downsampleFactor);
    outParallax = outParallax(cv::Rect(0, 0, cropW, cropH));
    outAO = outAO(cv::Rect(0, 0, cropW, cropH));
    outMetallic = outMetallic(cv::Rect(0, 0, cropW, cropH));
    outRoughness = outRoughness(cv::Rect(0, 0, cropW, cropH));
    LogMsg(L"Cropped outputs to " + std::to_wstring(cropW) + L"x" + std::to_wstring(cropH), 2);
}

// Process a single pair (normal, diffuse)
void processPair(const std::filesystem::path &normalPath,
                 const std::filesystem::path &diffusePath,
                 const std::filesystem::path &outDir,
                 const Models &M,
                 torch::Device device)
{
    LogMsg(L"Processing pair:", 0);
    LogMsg(L"Diffuse: " + diffusePath.wstring(), 1);
    LogMsg(L"Normal:  " + normalPath.wstring(), 1);
    // Load images
    // Use texconv to convert DDS to PNG with correct color spaces (sRGB for diffuse, linear for normals)
    auto dPath = EnsureReadableImage(diffusePath, /*isSRGB=*/true);
    auto nPath = EnsureReadableImage(normalPath, /*isSRGB=*/false);
    // Helper to convert std::wstring to UTF-8 std::string for OpenCV
    auto ws2s = [](const std::wstring &ws) -> std::string
    {
        int size_needed = WideCharToMultiByte(CP_UTF8, 0, ws.c_str(), (int)ws.size(), NULL, 0, NULL, NULL);
        std::string strTo(size_needed, 0);
        WideCharToMultiByte(CP_UTF8, 0, ws.c_str(), (int)ws.size(), &strTo[0], size_needed, NULL, NULL);
        return strTo;
    };
    cv::Mat diffuse = cv::imread(ws2s(dPath.wstring()), cv::IMREAD_UNCHANGED);
    cv::Mat normal = cv::imread(ws2s(nPath.wstring()), cv::IMREAD_UNCHANGED);
    if (diffuse.empty() || normal.empty())
    {
        throw std::runtime_error("Failed to load input images (diffuse/normal)");
    }
    LogMsg(L"Loaded: diffuse=" + std::to_wstring(diffuse.cols) + L"x" + std::to_wstring(diffuse.rows) + L", normal=" + std::to_wstring(normal.cols) + L"x" + std::to_wstring(normal.rows), 1);

    // Resize normal to diffuse size if needed, and renormalize
    if (normal.cols != diffuse.cols || normal.rows != diffuse.rows)
    {
        cv::resize(normal, normal, diffuse.size(), 0, 0, cv::INTER_LINEAR);
        LogMsg(L"Resized normal to diffuse size", 1);
    }
    normal = normalizeNormalImage(normal);
    LogMsg(L"Renormalized normal map", 1);

    // Run albedo tiled with segformer refinement
    auto albedoCHW = runAlbedoTiled(M.unetAlbedo, M.segformer, diffuse, normal, device);
    cv::Mat albedoBGR = toImageFromTensor(albedoCHW.clamp(0, 1));

    // Prepare inputs for PBR stage
    // auto albedoStd = albedoCHW; // already 0..1
    // auto albedoDef = albedoCHW.clone();
    // for (int c = 0; c < 3; ++c)
    //     albedoDef[c] = (albedoDef[c] - IMAGENET_DEFAULT_MEAN[c]) / IMAGENET_DEFAULT_STD[c];

    cv::Mat parallax, ao, metallic, roughness;
    runPBR(M, albedoBGR, normal, device, parallax, ao, metallic, roughness);
    LogMsg(L"Generated PBR PNGs (parallax, AO, metallic, roughness)", 1);

    std::filesystem::create_directories(outDir);
    auto base = normalPath.stem().wstring();
    // Create RMAOS: R=roughness, G=metallic, B=AO, A=255
    cv::Mat r_gray, m_gray, ao_gray, alpha;
    if (roughness.channels() == 3)
        cv::cvtColor(roughness, r_gray, cv::COLOR_BGR2GRAY);
    else
        r_gray = roughness;
    if (metallic.channels() == 3)
        cv::cvtColor(metallic, m_gray, cv::COLOR_BGR2GRAY);
    else
        m_gray = metallic;
    if (ao.channels() == 3)
        cv::cvtColor(ao, ao_gray, cv::COLOR_BGR2GRAY);
    else
        ao_gray = ao;
    alpha = cv::Mat(r_gray.size(), CV_8UC1, cv::Scalar(255));
    std::vector<cv::Mat> rmaos_ch{r_gray, m_gray, ao_gray, alpha};
    cv::Mat rmaos;
    cv::merge(rmaos_ch, rmaos);

    // Preserve diffuse alpha if it's meaningful (non-white)
    bool keepAlpha = HasNonWhiteAlpha(diffuse);
    cv::Mat albedoOut;
    if (keepAlpha)
    {
        // Make BGRA albedo and copy alpha from diffuse
        cv::Mat albedoBGRA;
        cv::cvtColor(albedoBGR, albedoBGRA, cv::COLOR_BGR2BGRA);
        std::vector<cv::Mat> difCh;
        cv::split(diffuse, difCh);
        std::vector<cv::Mat> albCh;
        cv::split(albedoBGRA, albCh);
        albCh[3] = difCh.back();
        cv::merge(albCh, albedoOut);
    }
    else
    {
        albedoOut = albedoBGR;
    }

    // Write PNG intermediates
    auto albedo_png = outDir / (base + L".png");
    auto rmaos_png = outDir / (base + L"_rmaos.png");
    auto p_png = outDir / (base + L"_p.png");
    cv::imwrite(ws2s(albedo_png.wstring()), albedoOut);
    cv::imwrite(ws2s(rmaos_png.wstring()), rmaos);
    cv::imwrite(ws2s(p_png.wstring()), parallax);
    LogMsg(L"Wrote PNGs:", 1);
    LogMsg(L"- " + albedo_png.wstring(), 2);
    LogMsg(L"- " + rmaos_png.wstring(), 2);
    LogMsg(L"- " + p_png.wstring(), 2);

    // Convert PNGs to DDS similar to Python
    auto texconv = GetTexconvOrExit();
    std::wstring outDirW = outDir.wstring();
    // Albedo
    std::wstring albFmt = keepAlpha ? L"BC7_UNORM_SRGB" : L"BC1_UNORM_SRGB";
    std::wstring albCmd = L"\"" + texconv.wstring() + L"\" -nologo -f " + albFmt + L" -ft dds --srgb-in -y -m 0 -o \"" + outDirW + L"\" \"" + albedo_png.wstring() + L"\"";
    if (keepAlpha)
        albCmd += L" --separate-alpha";
    LogMsg(L"Converting Albedo PNG -> DDS (" + albFmt + L")", 1);
    RunTexconv(albCmd);

    // RMAOS: BC1_UNORM
    std::wstring rmaosCmd = L"\"" + texconv.wstring() + L"\" -nologo -f BC1_UNORM -ft dds -y -m 0 -o \"" + outDirW + L"\" \"" + rmaos_png.wstring() + L"\"";
    LogMsg(L"Converting RMAOS PNG -> DDS (BC1_UNORM)", 1);
    RunTexconv(rmaosCmd);

    // Parallax: BC4_UNORM
    std::wstring pCmd = L"\"" + texconv.wstring() + L"\" -nologo -f BC4_UNORM -ft dds -y -m 0 -o \"" + outDirW + L"\" \"" + p_png.wstring() + L"\"";
    LogMsg(L"Converting Parallax PNG -> DDS (BC4_UNORM)", 1);
    RunTexconv(pCmd);

    // Normal: drop alpha and convert to BC7_UNORM
    cv::Mat normalRGB;
    if (normal.channels() == 4)
        cv::cvtColor(normal, normalRGB, cv::COLOR_BGRA2BGR);
    else if (normal.channels() == 3)
        normalRGB = normal;
    else
        cv::cvtColor(normal, normalRGB, cv::COLOR_GRAY2BGR);
    auto normal_png = outDir / (base + L"_n.png");
    cv::imwrite(ws2s(normal_png.wstring()), normalRGB);
    std::wstring nCmd = L"\"" + texconv.wstring() + L"\" -nologo -f BC7_UNORM -ft dds -y -m 0 -o \"" + outDirW + L"\" \"" + normal_png.wstring() + L"\"";
    LogMsg(L"Converting Normal PNG -> DDS (BC7_UNORM)", 1);
    RunTexconv(nCmd);

    // Remove PNG intermediates
    std::error_code ec;
    std::filesystem::remove(albedo_png, ec);
    std::filesystem::remove(rmaos_png, ec);
    std::filesystem::remove(p_png, ec);
    std::filesystem::remove(normal_png, ec);
    LogMsg(L"DDS written to: " + outDirW, 1);

    // Group end
    LogMsg(L"Pair done.", 0);
}

// Thread worker: scan input for *_n.* and corresponding diffuse files and process
void workerRun(std::filesystem::path inDir, std::filesystem::path outDir)
{
    g_isRunning = true;
    EnableWindow(g_btnRun, FALSE);

    try
    {
        // Load TorchScript models from a conventional directory relative to exe or workspace
        // Expect files:
        //  segformer.ts, unet_albedo.ts, unet_parallax.ts, unet_ao.ts, unet_metallic.ts, unet_roughness.ts
        std::filesystem::path modelRoot = std::filesystem::current_path() / "weights_ts";
        torch::jit::script::Module segformer = torch::jit::load((modelRoot / "segformer.ts").string());
        torch::jit::script::Module unetAlbedo = torch::jit::load((modelRoot / "unet_albedo.ts").string());
        torch::jit::script::Module unetParallax = torch::jit::load((modelRoot / "unet_parallax.ts").string());
        torch::jit::script::Module unetAO = torch::jit::load((modelRoot / "unet_ao.ts").string());
        torch::jit::script::Module unetMetallic = torch::jit::load((modelRoot / "unet_metallic.ts").string());
        torch::jit::script::Module unetRoughness = torch::jit::load((modelRoot / "unet_roughness.ts").string());

        torch::Device device(torch::kCPU);
        if (torch::cuda::is_available())
            device = torch::Device(torch::kCUDA);

        segformer.to(device);
        unetAlbedo.to(device);
        unetParallax.to(device);
        unetAO.to(device);
        unetMetallic.to(device);
        unetRoughness.to(device);
        segformer.eval();
        unetAlbedo.eval();
        unetParallax.eval();
        unetAO.eval();
        unetMetallic.eval();
        unetRoughness.eval();

        Models M{segformer, unetAlbedo, unetParallax, unetAO, unetMetallic, unetRoughness};

        LogMsg(L"Models loaded on device: " + std::wstring(device.is_cuda() ? L"CUDA" : L"CPU"));
        LogMsg(L"Scanning input: " + inDir.wstring());
        // Gather normal files
        std::vector<std::filesystem::path> normals;
        for (auto &p : std::filesystem::recursive_directory_iterator(inDir))
        {
            if (!p.is_regular_file())
                continue;
            auto ext = p.path().extension().wstring();
            std::wstring name = p.path().filename().wstring();
            if (ext == L".png" || ext == L".dds" || ext == L".jpg" || ext == L".tif")
            {
                if (name.size() > 2 && (name.find(L"_n.") != std::wstring::npos))
                {
                    normals.push_back(p.path());
                }
            }
        }
        g_progressMax = (int)normals.size();
        SendMessage(g_progress, PBM_SETRANGE, 0, MAKELPARAM(0, g_progressMax.load()));
        g_progressValue = 0;
        LogMsg(L"Found normal maps: " + std::to_wstring(normals.size()));

        for (size_t i = 0; i < normals.size() && g_isRunning; ++i)
        {
            auto normalPath = normals[i];
            auto diffusePath = normalPath;
            // Try _d then without suffix
            auto name = normalPath.filename().wstring();
            std::filesystem::path alt = normalPath;
            std::wstring stem = normalPath.stem().wstring();
            // remove trailing _n from stem
            if (stem.size() > 2 && stem.substr(stem.size() - 2) == L"_n")
            {
                std::wstring base = stem.substr(0, stem.size() - 2);
                diffusePath = normalPath.parent_path() / (base + L"_d" + normalPath.extension().wstring());
                if (!std::filesystem::exists(diffusePath))
                    diffusePath = normalPath.parent_path() / (base + normalPath.extension().wstring());
            }

            if (!std::filesystem::exists(diffusePath))
            {
                LogMsg(L"Skipping: diffuse not found for normal " + normalPath.wstring());
                std::scoped_lock lk(g_logMutex);
                SetStatus(L"Skipping: diffuse not found for " + normalPath.wstring());
                g_progressValue++;
                SendMessage(g_progress, PBM_SETPOS, g_progressValue.load(), 0);
                continue;
            }

            auto rel = std::filesystem::relative(normalPath.parent_path(), inDir);
            auto outSub = outDir / rel;
            try
            {
                processPair(normalPath, diffusePath, outSub, M, device);
            }
            catch (const std::bad_alloc &)
            {
                ShowFatalAndExit(L"Out of memory during inference.");
            }
            catch (const c10::Error &e)
            {
                std::wstring w = L"Torch error: ";
                w += std::wstring(e.msg().c_str(), e.msg().c_str() + e.msg().size());
                ShowFatalAndExit(w);
            }
            catch (const std::exception &e)
            {
                std::wstring w = L"Error: ";
                w += std::wstring(e.what(), e.what() + strlen(e.what()));
                ShowFatalAndExit(w);
            }
            g_progressValue++;
            SendMessage(g_progress, PBM_SETPOS, g_progressValue.load(), 0);
        }

        SetStatus(L"Done.");
        LogMsg(L"All done.");
    }
    catch (const c10::Error &e)
    {
        std::wstring msg = L"Torch error: ";
        msg += std::wstring(e.msg().c_str(), e.msg().c_str() + e.msg().size());
        ShowFatalAndExit(msg);
    }
    catch (const std::exception &e)
    {
        std::wstring msg = L"Error: ";
        msg += std::wstring(e.what(), e.what() + strlen(e.what()));
        ShowFatalAndExit(msg);
    }

    g_isRunning = false;
    EnableWindow(g_btnRun, TRUE);
}

LRESULT CALLBACK WndProc(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam)
{
    switch (msg)
    {
    case WM_CREATE:
    {
        INITCOMMONCONTROLSEX icc{sizeof(INITCOMMONCONTROLSEX), ICC_PROGRESS_CLASS};
        InitCommonControlsEx(&icc);

        g_btnInput = CreateWindowW(L"BUTTON", L"Input directory", WS_VISIBLE | WS_CHILD | BS_PUSHBUTTON,
                                   10, 10, 150, 30, hWnd, (HMENU)IDC_BTN_INPUT, g_hInst, nullptr);
        g_btnOutput = CreateWindowW(L"BUTTON", L"Output directory", WS_VISIBLE | WS_CHILD | BS_PUSHBUTTON,
                                    170, 10, 150, 30, hWnd, (HMENU)IDC_BTN_OUTPUT, g_hInst, nullptr);
        g_btnRun = CreateWindowW(L"BUTTON", L"Run", WS_VISIBLE | WS_CHILD | BS_PUSHBUTTON,
                                 330, 10, 80, 30, hWnd, (HMENU)IDC_BTN_RUN, g_hInst, nullptr);
        g_progress = CreateWindowW(PROGRESS_CLASSW, nullptr, WS_VISIBLE | WS_CHILD,
                                   10, 50, 400, 20, hWnd, (HMENU)IDC_PROGRESS, g_hInst, nullptr);
        g_status = CreateWindowW(L"STATIC", L"Select input and output directories.", WS_VISIBLE | WS_CHILD,
                                 10, 80, 600, 20, hWnd, (HMENU)IDC_STATUS, g_hInst, nullptr);
        SendMessage(g_progress, PBM_SETRANGE, 0, MAKELPARAM(0, 100));
        break;
    }
    case WM_COMMAND:
    {
        switch (LOWORD(wParam))
        {
        case IDC_BTN_INPUT:
        {
            auto sel = PickFolder(hWnd);
            if (sel)
            {
                g_inputDir = *sel;
                SetStatus(L"Input: " + g_inputDir);
            }
            break;
        }
        case IDC_BTN_OUTPUT:
        {
            auto sel = PickFolder(hWnd);
            if (sel)
            {
                g_outputDir = *sel;
                SetStatus(L"Output: " + g_outputDir);
            }
            break;
        }
        case IDC_BTN_RUN:
        {
            if (g_isRunning)
                break;
            if (g_inputDir.empty() || g_outputDir.empty())
            {
                SetStatus(L"Please select both input and output directories.");
                break;
            }
            std::thread(workerRun, std::filesystem::path(g_inputDir), std::filesystem::path(g_outputDir)).detach();
            break;
        }
        }
        break;
    }
    case WM_DESTROY:
        PostQuitMessage(0);
        break;
    default:
        return DefWindowProc(hWnd, msg, wParam, lParam);
    }
    return 0;
}

int APIENTRY wWinMain(HINSTANCE hInstance, HINSTANCE, LPWSTR, int nCmdShow)
{
    // Open console for logging
    InitConsole();
    LogMsg(L"Starting AI PBR GUI...");
    g_hInst = hInstance;
    WNDCLASSEXW wc{sizeof(WNDCLASSEXW)};
    wc.lpfnWndProc = WndProc;
    wc.hInstance = hInstance;
    wc.lpszClassName = L"AIPBRGuiWnd";
    wc.hCursor = LoadCursor(nullptr, IDC_ARROW);
    wc.hbrBackground = (HBRUSH)(COLOR_WINDOW + 1);
    RegisterClassExW(&wc);

    g_hWnd = CreateWindowW(wc.lpszClassName, L"AI PBR GUI", WS_OVERLAPPEDWINDOW,
                           CW_USEDEFAULT, CW_USEDEFAULT, 640, 160,
                           nullptr, nullptr, hInstance, nullptr);
    ShowWindow(g_hWnd, nCmdShow);
    UpdateWindow(g_hWnd);

    MSG msg;
    while (GetMessage(&msg, nullptr, 0, 0))
    {
        TranslateMessage(&msg);
        DispatchMessage(&msg);
    }
    return 0;
}
