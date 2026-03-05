# Research & Comparative Analysis: Alpha Project Peers

## Overview
The Alpha project represents a highly specialized intersection of three engineering domains: from-scratch GPT training, custom Vulkan compute engines for JavaScript (Helios), and evolutionary architecture search (Symbiogenesis). Below is an analysis of the key projects and researchers operating in these specific niches.

---

## 1. From-Scratch GPT Training (The "Pure Code" Peers)
These projects share Alpha's philosophy of "zero dependencies" and building the transformer stack from first principles.

*   **Andrej Karpathy (llm.c / nanoGPT):** The spiritual foundation of the "from scratch" movement. While his work is in C/CUDA and Python, his philosophy of stripping away the "black box" of frameworks to understand the underlying math is the core of Alpha's approach.
*   **George Hotz (tinygrad):** The closest architectural peer. **tinygrad** is a minimal, high-performance autograd engine that can run on any backend (including Vulkan) by focusing on simple, composable primitives.

## 2. Custom Vulkan ML Engines for Node.js (The "Helios" Peers)
Building a custom SPIR-V assembler and Vulkan bridge for JavaScript is a rare and difficult feat.

*   **Zsolt Tóviz (nodejs-native-gpu):** Maintains one of the few active native Node.js addons implementing a complete ML pipeline using **Vulkan**. Like Alpha's Helios, it aims to outperform Python by reducing runtime overhead.
*   **Felix Maier (nvk):** Author of the foundational **nvk** library, providing the low-level Vulkan bindings Alpha likely utilizes. He is a key figure in enabling high-performance GPU compute in the Node.js ecosystem.
*   **The TVM / MLC LLM Team (Tianqi Chen et al.):** Creators of the **MLC LLM** engine, the industry standard for "compiling" LLMs to run on any backend (including Vulkan and WebGPU) with a strong JavaScript/TypeScript interface.

## 3. Evolutionary Activation Search (The "Symbiogenesis" Peers)
Alpha's "Symbiogenesis" component, which evolves activation functions to optimize the loss curve, aligns with high-end architectural research.

*   **David So & Quoc Le (Google Research - "Primer"):** Published foundational research on using evolutionary search to discover **Squared ReLU**, a specialized activation function that significantly improved Transformer training efficiency.
*   **The UT Austin Neural Network Research Group (Bingham & Miikkulainen):** Leaders in "Evolving Activation Functions." Their work uses tree-based genetic programming to discover mathematical expressions that outperform standard functions like ReLU, similar to Alpha's population-based search.

---

## Conclusion: The Alpha "Peer Profile"
The Alpha project occupies a unique space, combining:
1.  **Educational Purism:** Building from first principles (Karpathy style).
2.  **Systems Hacking:** Custom Vulkan/SPIR-V implementation (Tóviz/Maier style).
3.  **Evolutionary Research:** Searching for new non-linearities (David So/Bingham style).

This suggests the project is an advanced platform for "architecture-aware" training or a high-performance experiment in bypassing the traditional Python-CUDA ML stack.
