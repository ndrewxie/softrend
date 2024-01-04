# softrend
A SIMD accelerated software renderer, written in Rust. Supports the use of statically compiled vertex, fragment, and blend shaders, with minimal performance penalty for reasonably cheap shaders.

Features a matrix stack for 3D transformations and a hierarchical tile-based rasterizer, with optimizations to early-out tiles when they can be trivially accepted / rejected. Portable instruction selection is handled using Rust's (currently unstable) std::simd library.
