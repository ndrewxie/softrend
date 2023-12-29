# softrend
A SIMD accelerated software renderer, written in Rust. Largely an experiment in the limits of software rendering on modern CPUs, using Rust's (currently unstable) std::simd library to handle instruction selection in a portable manner.
Features a matrix stack for 3D transformations and a hierarchical tile-based rasterizer, with optimizations to early-out tiles when they can be trivially accepted / rejected.
