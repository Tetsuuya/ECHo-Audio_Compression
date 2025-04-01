# Testing STEPS

1. Open MSYS2 MINGW64
2.  cd "File directry"
3. g++ -o audio_compressor audio_compressor.cpp -lsndfile -std=c++14 -O2
4.  ./audio_compressor input.wav
5. ./audio_compressor -d
