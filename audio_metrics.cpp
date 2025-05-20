#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include <sndfile.h>
#include <complex>
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

using namespace std;

// Calculate Mean Squared Error
double calculateMSE(const vector<double>& original, const vector<double>& compressed) {
    double sum = 0.0;
    for (size_t i = 0; i < original.size(); ++i) {
        double diff = original[i] - compressed[i];
        sum += diff * diff;
    }
    return sum / original.size();
}

// Calculate PSNR
double calculatePSNR(double mse) {
    if (mse == 0) return INFINITY;
    return 10.0 * log10((32767.0 * 32767.0) / mse);
}

// Calculate SNR
// SNR = 10 * log10(sum(x^2) / sum((x-x_hat)^2))
double calculateSNR(const std::vector<double>& original, const std::vector<double>& compressed) {
    double signalEnergy = 0.0;
    double noiseEnergy = 0.0;
    for (size_t i = 0; i < original.size(); ++i) {
        signalEnergy += original[i] * original[i];
        double noise = original[i] - compressed[i];
        noiseEnergy += noise * noise;
    }
    if (noiseEnergy == 0) return INFINITY;
    return 10.0 * log10(signalEnergy / noiseEnergy);
}

// Calculate compression ratios
struct CompressionRatios {
    double wavRatio;      // ratio between input.wav and output.wav
    double actualRatio;   // ratio between input.wav and compressed.bin
};

CompressionRatios calculateCompressionRatios(const string& inputFile, const string& outputFile) {
    ifstream input(inputFile, ios::binary | ios::ate);
    ifstream output(outputFile, ios::binary | ios::ate);
    ifstream compressed("compressed.bin", ios::binary | ios::ate);
    
    CompressionRatios ratios = {0.0, 0.0};
    
    if (!input) {
        cout << "Warning: Cannot open input file for size calculation" << endl;
        return ratios;
    }
    
    double inputSize = input.tellg();
    
    if (output) {
        double outputSize = output.tellg();
        ratios.wavRatio = inputSize / outputSize;
    }
    
    if (compressed) {
        double compressedSize = compressed.tellg();
        ratios.actualRatio = inputSize / compressedSize;
    } else {
        cout << "Warning: compressed.bin not found" << endl;
    }
    
    return ratios;
}

// Normalize audio to [-1, 1]
vector<double> normalizeAudio(const vector<int16_t>& audio) {
    vector<double> normalized(audio.size());
    for (size_t i = 0; i < audio.size(); ++i) {
        normalized[i] = audio[i] / 32768.0;
    }
    return normalized;
}

// FFT implementation for real signals
void fft(const std::vector<double>& in, std::vector<std::complex<double>>& out) {
    size_t N = in.size();
    out.resize(N);
    if (N == 0) return;
    if ((N & (N - 1)) != 0) throw std::runtime_error("FFT input size must be a power of 2");
    // Bit reversal
    size_t j = 0;
    for (size_t i = 0; i < N; ++i) {
        if (i < j) std::swap(out[i], out[j]);
        out[i] = in[i];
        size_t m = N >> 1;
        while (m >= 1 && j >= m) {
            j -= m;
            m >>= 1;
        }
        j += m;
    }
    // Danielson-Lanczos
    for (size_t s = 1; s <= static_cast<size_t>(log2(N)); ++s) {
        size_t m = 1 << s;
        std::complex<double> wm = std::exp(std::complex<double>(0, -2.0 * M_PI / m));
        for (size_t k = 0; k < N; k += m) {
            std::complex<double> w = 1;
            for (size_t j = 0; j < m / 2; ++j) {
                std::complex<double> t = w * out[k + j + m / 2];
                std::complex<double> u = out[k + j];
                out[k + j] = u + t;
                out[k + j + m / 2] = u - t;
                w *= wm;
            }
        }
    }
}

// Calculate THD (proper version)
double calculateTHD(const std::vector<double>& signal, double sampleRate) {
    // Zero-pad to next power of 2
    size_t N = 1;
    while (N < signal.size()) N <<= 1;
    std::vector<double> padded = signal;
    padded.resize(N, 0.0);
    std::vector<std::complex<double>> spectrum(N);
    fft(padded, spectrum);
    // Compute magnitude spectrum
    std::vector<double> mag(N / 2);
    for (size_t i = 0; i < N / 2; ++i) {
        mag[i] = std::abs(spectrum[i]) / (N / 2);
    }
    // Find fundamental frequency bin (ignore DC)
    size_t fundamentalBin = 1;
    double maxMag = mag[1];
    for (size_t i = 2; i < N / 2; ++i) {
        if (mag[i] > maxMag) {
            maxMag = mag[i];
            fundamentalBin = i;
        }
    }
    // RMS of fundamental
    double V1 = mag[fundamentalBin] / std::sqrt(2.0);
    // RMS of harmonics (2nd to 5th)
    double sumHarmonics2 = 0.0;
    int maxHarmonic = 5;
    for (int h = 2; h <= maxHarmonic; ++h) {
        size_t bin = fundamentalBin * h;
        if (bin < N / 2) {
            double Vh = mag[bin] / std::sqrt(2.0);
            sumHarmonics2 += Vh * Vh;
        }
    }
    double thd = (V1 > 0) ? std::sqrt(sumHarmonics2) / V1 : 0.0;
    return thd;
}

void analyzeAudio(const string& inputFile, const string& outputFile) {
    cout << "Opening files..." << endl;
    
    // Open input file
    SF_INFO inputInfo = {};
    SNDFILE* inFile = sf_open(inputFile.c_str(), SFM_READ, &inputInfo);
    if (!inFile) {
        throw runtime_error("Cannot open input file: " + string(sf_strerror(nullptr)));
    }
    
    // Open output file
    SF_INFO outputInfo = {};
    SNDFILE* outFile = sf_open(outputFile.c_str(), SFM_READ, &outputInfo);
    if (!outFile) {
        sf_close(inFile);
        throw runtime_error("Cannot open output file: " + string(sf_strerror(nullptr)));
    }
    
    cout << "Reading audio data..." << endl;
    
    // Read audio data
    vector<int16_t> inputAudio(inputInfo.frames * inputInfo.channels);
    vector<int16_t> outputAudio(outputInfo.frames * outputInfo.channels);
    
    sf_read_short(inFile, inputAudio.data(), inputAudio.size());
    sf_read_short(outFile, outputAudio.data(), outputAudio.size());
    
    sf_close(inFile);
    sf_close(outFile);
    
    cout << "Processing..." << endl;
    
    // Normalize audio
    vector<double> normalizedInput = normalizeAudio(inputAudio);
    vector<double> normalizedOutput = normalizeAudio(outputAudio);
    
    // Calculate metrics
    double mse = calculateMSE(normalizedInput, normalizedOutput);
    double psnr = calculatePSNR(mse);
    double snr = calculateSNR(normalizedInput, normalizedOutput);
    CompressionRatios ratios = calculateCompressionRatios(inputFile, outputFile);
    double thd = calculateTHD(normalizedOutput, outputInfo.samplerate);
    
    // Print results
    cout << "\n=== Audio Quality Analysis Results ===\n";
    cout << "Sample Rate: " << inputInfo.samplerate << " Hz\n";
    cout << "Channels: " << inputInfo.channels << "\n";
    cout << "Duration: " << static_cast<double>(inputInfo.frames) / inputInfo.samplerate << " seconds\n";
    cout << "\nQuality Metrics:\n";
    cout << "MSE: " << mse << "\n";
    cout << "PSNR: " << psnr << " dB\n";
    cout << "SNR: " << snr << " dB\n";
    cout << "WAV Compression Ratio (input.wav:output.wav): " << ratios.wavRatio << ":1\n";
    cout << "Actual Compression Ratio (input.wav:compressed.bin): " << ratios.actualRatio << ":1\n";
    cout << "THD: " << (thd * 100) << "%\n";
}

int main(int argc, char* argv[]) {
    try {
        if (argc != 3) {
            cout << "Usage: " << argv[0] << " <input.wav> <output.wav>\n";
            return 1;
        }
        
        cout << "Starting audio analysis...\n";
        cout << "Input file: " << argv[1] << "\n";
        cout << "Output file: " << argv[2] << "\n\n";
        
        analyzeAudio(argv[1], argv[2]);
        return 0;
    }
    catch (const exception& e) {
        cerr << "Error: " << e.what() << endl;
        return 1;
    }
} 