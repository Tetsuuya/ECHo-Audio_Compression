# Audio Compression Utility

This project implements an audio compression utility that uses several algorithms:
- Discrete Cosine Transform (DCT)
- Run-Length Encoding (RLE)
- Adaptive Huffman coding

## Project Structure

The code has been organized into two files:

- `compression_algorithms.cpp` - Contains the implementations of the three core compression algorithms
- `audio_compressor.cpp` - Main application that uses the algorithms to compress/decompress audio

## Algorithm Details

### DCT (Discrete Cosine Transform)
Transforms audio samples from the time domain to the frequency domain, allowing for 
better compression. The implementation includes both forward and inverse DCT.

### RLE (Run-Length Encoding)
Compresses sequences of repeated values by storing them as pairs of (value, count).
This is particularly effective after DCT and quantization, where many zeros appear in a row.

### Adaptive Huffman Coding
A variable-length prefix code that adapts to the data being encoded. Frequently occurring
symbols are assigned shorter codes, while less frequent ones get longer codes.

## Usage

```
# Compile the program
$ g++ -std=c++11 -O3 -o audio_compressor audio_compressor.cpp -lsndfile -lpthread

# Compress an audio file
./audio_compressor input.wav

# Decompress (creates output.wav)
./audio_compressor -d
```

## Requirements

- libsndfile (for audio I/O)
- C++11 compatible compiler
- Input audio should be in WAV format

## Performance

The compression ratio depends on the audio content. The implementation uses multi-threading
to improve performance on systems with multiple CPU cores. 