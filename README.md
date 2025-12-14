# Cartographers of the Invisible 🗺️

> Mapping the hidden geometries of meaning in Large Language Models

## Project Overview

**Cartographers of the Invisible** is a research and visualization project aimed at **mapping how Large Language Models (LLMs) represent meaning across linguistic and multimodal spaces**. By combining embedding analysis, visualization, and interpretability techniques, this project makes abstract representations visible, revealing how models cluster, separate, and relate concepts.

The goal is to design an interactive pipeline that transforms raw embeddings into navigable semantic maps, helping humans intuitively explore what the model "knows" and how that knowledge is structured.

## Project Goals

### 🎯 Core Objectives

1. **Extract and Analyze Embeddings**: Generate embeddings from LLMs across diverse linguistic and multimodal inputs to understand how models encode semantic information.

2. **Visualize Semantic Spaces**: Create interactive, intuitive visualizations that transform high-dimensional embedding spaces into comprehensible 2D/3D maps, revealing:

   - How concepts cluster together
   - How meanings separate and relate
   - The geometric structure of model knowledge
   - Cross-modal relationships (text, images, etc.)

3. **Enable Explorability**: Build tools that allow researchers and students to:

   - Navigate semantic landscapes interactively
   - Query relationships between concepts
   - Discover emergent patterns in model representations
   - Understand how context affects meaning

4. **Foster Interpretability**: Make the "invisible" visible by providing insights into:
   - What features models use to distinguish concepts
   - How similar or different concepts are in the model's internal space
   - What biases or structures emerge from training data
   - How representations evolve across model layers

## Research Questions

This project explores fundamental questions about LLM representations:

- **Topology of Meaning**: What geometric patterns emerge in semantic spaces? Are certain concepts universally close or distant across models?

- **Multimodal Alignment**: How do models align representations across modalities (e.g., text and images)? What does this reveal about cross-modal understanding?

- **Semantic Neighborhoods**: What makes concepts "neighbors" in embedding space? Can we identify meaningful clusters?

- **Context Sensitivity**: How do embeddings shift with context? What does this tell us about how models handle ambiguity and polysemy?

- **Model Comparison**: How do different models or architectures structure semantic space differently?

## Methodology

The project combines several technical approaches:

1. **Embedding Extraction**: Leverage pre-trained LLMs (e.g., BERT, GPT, CLIP) to generate embeddings for diverse inputs

2. **Dimensionality Reduction**: Apply techniques like t-SNE, UMAP, PCA to project high-dimensional embeddings into visualizable spaces

3. **Clustering & Analysis**: Use clustering algorithms to identify semantic groupings and analyze their characteristics

4. **Interactive Visualization**: Build web-based interfaces for exploring semantic maps with zooming, filtering, and querying capabilities

5. **Interpretability Tools**: Implement techniques to understand what features drive similarity/difference in the embedding space

## Technical Stack

This project provides a foundational toolkit built with:

- **Python** for data processing and analysis
- **Pandas & NumPy** for data manipulation
- **Matplotlib & Plotly** for static and interactive visualizations
- **Streamlit** for building interactive web applications
- **Jupyter Notebooks** for exploratory analysis and documentation

> **Note**: The current repository contains the scaffolding and foundational tools for this project. See [USAGE.md](USAGE.md) for technical documentation on the available modules and how to use them.

## Getting Started

### Installation

```bash
git clone https://github.com/Zografska/cartographers-of-the-invisible.git
cd cartographers-of-the-invisible
pip install -e .

# For BERT
pip install transformers torch numpy

# For Ollama
pip install ollama numpy
```

### Quick Start

The project provides Python modules for data analysis and an interactive Streamlit app for visualization. For detailed usage instructions, see [USAGE.md](USAGE.md).

## Project Structure

```
cartographers-of-the-invisible/
├── src/cartographers/       # Python package with reusable utilities
├── notebooks/               # Jupyter notebooks for analysis
├── app.py                   # Streamlit visualization application
├── README.md               # Project overview (this file)
└── USAGE.md                # Technical documentation
```

## Future Directions

This project is designed to evolve as research progresses. Future enhancements may include:

- Integration with specific LLM APIs for embedding extraction
- Advanced clustering and semantic analysis algorithms
- Multi-layer embedding analysis to track how representations evolve
- Comparative analysis across different model architectures
- Real-time embedding generation and visualization
- Tools for bias detection and fairness analysis in semantic spaces

## Contributing

This is an educational and research project. Contributions, ideas, and discussions are welcome! Please feel free to:

- Open issues to discuss ideas or report problems
- Submit pull requests with improvements
- Share interesting findings from your explorations

## Educational Context

This project is designed for students and researchers exploring:

- Natural Language Processing and LLMs
- Machine Learning interpretability
- Data visualization and visual analytics
- Computational semantics
- Human-AI interaction

## License

This project is open source and available under the MIT License.

## Acknowledgments

Inspired by the need to make AI systems more interpretable and the fascinating geometry of meaning encoded in neural networks.

---

**For technical documentation and usage instructions, see [USAGE.md](USAGE.md).**
