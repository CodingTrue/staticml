<p align="center">
    <img src="assets/StaticML-Logo.png" width="256" height="64"/><br>
    <strong>A Python ML framework based on PyOpenCL</strong><br><br>
    <img alt="Static Badge" src="https://img.shields.io/badge/Python-v3.14-blue?style=flat-square&logo=python&logoColor=yellow&link=https%3A%2F%2Fwww.python.org%2Fdownloads%2Frelease%2Fpython-3140%2F">
    <img alt="Static Badge" src="https://img.shields.io/badge/PyOpenCL-v2026-blue?style=flat-square&link=https%3A%2F%2Fpypi.org%2Fproject%2Fpyopencl%2F">
</p>

# StaticML

> **Build once. Run everywhere.**

**StaticML** is an early-stage machine learning framework for Python, built on top of **PyOpenCL** and designed with **AMD hardware** in mind.

The goal is to provide a flexible foundation for accelerated machine learning while keeping both high-level prototyping and low-level control accessible.

> ⚠️ **Early Development**
>
> StaticML is currently under active development and **is not yet ready for out-of-the-box use**.

## Expected Features

* 🧩 **Custom OpenCL Kernels** - Write and integrate your own kernels for specialized workloads.
* 🟥 **AMD Acceleration** - First-class focus on AMD hardware, including newer **RDNA 4** GPUs.
* 🛠️ **Easy Prototyping** - Keep experimentation and development simple and approachable.
* 🌍 **Broad Device Compatibility** - Aim for support across a wide range of OpenCL-capable devices.
* 📦 **ONNX Support** - Load and work with models distributed in the ONNX format.
* ⚙️ **Low-Level Access** - Drop down to the underlying compute layer when custom logic or fine-grained control is needed.

## 🎯 Vision

StaticML aims to be a **lightweight, flexible, and widely compatible alternative to heavyweight machine learning frameworks**. It is designed to address some of the frustrations of PyTorch - particularly its large footprint, complexity, and limited AMD-focused support - while giving developers greater control over the underlying compute stack.

The goal is to make accelerated machine learning **simpler, leaner, and more accessible across hardware**, without sacrificing the low-level control needed for advanced workloads.
