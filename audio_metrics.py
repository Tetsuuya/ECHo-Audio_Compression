import numpy as np
import soundfile as sf
import os
from typing import Tuple, List
import math
from scipy.fft import fft

def calculate_mse(original: np.ndarray, compressed: np.ndarray) -> float:
    """Calculate Mean Squared Error between original and compressed signals."""
    return np.mean((original - compressed) ** 2)

def calculate_psnr(mse: float) -> float:
    """Calculate Peak Signal-to-Noise Ratio."""
    if mse == 0:
        return float('inf')
    # MAX^2 for normalized audio in [-1, 1] is 1.0 * 1.0 = 1.0
    return 10.0 * math.log10(1.0 / mse)

def calculate_snr(original: np.ndarray, compressed: np.ndarray) -> float:
    """Calculate Signal-to-Noise Ratio."""
    signal_energy = np.sum(original ** 2)
    noise_energy = np.sum((original - compressed) ** 2)
    if noise_energy == 0:
        return float('inf')
    return 10.0 * math.log10(signal_energy / noise_energy)

def calculate_compression_ratios(input_file: str, output_file: str) -> Tuple[float, float]:
    """Calculate compression ratios between input and output files."""
    input_size = os.path.getsize(input_file)
    output_size = os.path.getsize(output_file)
    compressed_size = os.path.getsize('compressed.bin') if os.path.exists('compressed.bin') else 0
    
    wav_ratio = input_size / output_size if output_size > 0 else 0
    actual_ratio = input_size / compressed_size if compressed_size > 0 else 0
    
    return wav_ratio, actual_ratio

def calculate_thd(audio_data, sample_rate):
    """
    Calculate Total Harmonic Distortion (THD) from audio data.
    
    Args:
        audio_data (numpy.ndarray): Audio signal data
        sample_rate (int): Sample rate of the audio
        
    Returns:
        float: THD value as a percentage
    """
    # Convert to float32 for better precision
    audio_data = audio_data.astype(np.float32)
    
    # Normalize the signal
    if np.max(np.abs(audio_data)) > 0:
        audio_data = audio_data / np.max(np.abs(audio_data))
    
    # Perform FFT
    n = len(audio_data)
    if n == 0:
        return 0.0
    fft_result = fft(audio_data)
    fft_magnitude = np.abs(fft_result[:n//2])
    
    # Find the fundamental frequency (highest peak, ignore DC)
    if len(fft_magnitude[1:]) == 0:
        return 0.0
    fundamental_idx = np.argmax(fft_magnitude[1:]) + 1
    
    # Calculate RMS values
    # Fundamental frequency component
    v1 = fft_magnitude[fundamental_idx]
    
    # Harmonic components (2nd to 10th)
    harmonic_sum_sq = 0
    for i in range(2, 11):
        harmonic_idx = fundamental_idx * i
        if harmonic_idx < len(fft_magnitude):
            # Calculate RMS of harmonic and sum squares
            vh = fft_magnitude[harmonic_idx]
            harmonic_sum_sq += vh ** 2
    
    # Calculate THD
    # The formula in the image uses amplitude for V1 and Vh, so we don't divide by sqrt(2) here
    thd = np.sqrt(harmonic_sum_sq) / v1 if v1 > 0 else 0.0
    thd_percentage = thd * 100
    
    return thd_percentage

def analyze_audio(input_file: str, output_file: str) -> None:
    """Analyze audio files and print quality metrics."""
    print("Opening files...")
    
    # Read input file
    input_data, input_samplerate = sf.read(input_file)
    output_data, output_samplerate = sf.read(output_file)
    
    # Convert to mono if stereo and ensure float type for calculations
    if len(input_data.shape) > 1:
        input_data = np.mean(input_data, axis=1)
    if len(output_data.shape) > 1:
        output_data = np.mean(output_data, axis=1)
    
    # Ensure data is float type for calculations if not already (sf.read normally returns float for WAV)
    if not np.issubdtype(input_data.dtype, np.floating):
         input_data = input_data.astype(np.float32)
         if np.max(np.abs(input_data)) > 0:
             input_data = input_data / np.max(np.abs(input_data))

    if not np.issubdtype(output_data.dtype, np.floating):
         output_data = output_data.astype(np.float32)
         if np.max(np.abs(output_data)) > 0:
             output_data = output_data / np.max(np.abs(output_data))


    print("Processing...")
    
    # Calculate metrics
    # Ensure inputs for MSE and SNR are the same length, truncate to shorter if necessary
    min_len = min(len(input_data), len(output_data))
    input_data_trunc = input_data[:min_len]
    output_data_trunc = output_data[:min_len]

    mse = calculate_mse(input_data_trunc, output_data_trunc)
    psnr = calculate_psnr(mse)
    snr = calculate_snr(input_data_trunc, output_data_trunc)
    compression_ratios = calculate_compression_ratios(input_file, output_file)
    # THD calculated using the new formula
    thd_input = calculate_thd(input_data, input_samplerate)
    thd_output = calculate_thd(output_data, output_samplerate)
    
    # Print results
    print("\n=== Audio Quality Analysis Results ===")
    print(f"Sample Rate: {input_samplerate} Hz")
    print(f"Input Channels: {1 if len(input_data.shape) == 1 else input_data.shape[1]}")
    print(f"Output Channels: {1 if len(output_data.shape) == 1 else output_data.shape[1]}")
    print(f"Input Duration: {len(input_data) / input_samplerate:.2f} seconds")
    print(f"Output Duration: {len(output_data) / output_samplerate:.2f} seconds")
    print("\nQuality Metrics:")
    print(f"MSE: {mse:.6f}")
    print(f"PSNR: {psnr:.2f} dB")
    print(f"SNR: {snr:.2f} dB")
    print(f"WAV Compression Ratio (input.wav:output.wav): {compression_ratios[0]:.2f}:1")
    print(f"Actual Compression Ratio (input.wav:compressed.bin): {compression_ratios[1]:.2f}:1")
    print(f"THD (Input): {thd_input:.2f}%")
    print(f"THD (Output): {thd_output:.2f}%")

def main():
    import sys
    
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <input.wav> <output.wav>")
        sys.exit(1)
    
    print("Starting audio analysis...")
    print(f"Input file: {sys.argv[1]}")
    print(f"Output file: {sys.argv[2]}\n")
    
    try:
        analyze_audio(sys.argv[1], sys.argv[2])
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
