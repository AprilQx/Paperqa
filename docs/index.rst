SciRAG: Scientific Retrieval-Augmented Generation
==================================================

.. image:: ../notebooks/results/comparison.png
   :width: 800
   :alt: SciRAG Performance Comparison
   :align: center

**A comprehensive research framework for evaluating Retrieval-Augmented Generation (RAG) systems on scientific literature**

Overview
--------

SciRAG addresses a fundamental question in AI research: Can AI achievements in scientific literature synthesis generalize across different scientific domains, or do different fields require fundamentally different approaches?

Our research demonstrates:

* **Successfully reproduced PaperQA2** achieving 71% accuracy on LitQA2 biology benchmark
* **Critical train/test performance gap**: GPT-4o-mini baseline shows 40% accuracy drop from train (30%) to test (18%)
* **Domain-specific optimization** is crucial: Commercial systems significantly outperformed PaperQA2 on astronomical tasks
* **Cost-performance insights**: VertexAI optimal for cost-conscious applications (13.3× cost advantage)

.. figure:: ../notebooks/results/paperqa2_performance_on_test.png
   :width: 800px
   :align: center
   :alt: PaperQA2 Performance on Test Dataset

   PaperQA2 Performance on Test Dataset - Achieving 71% accuracy matching the original paper

Performance Tiers Identified
-----------------------------

1. **Tier 1 - Premium Commercial Systems**
   
   - OpenAI Assistant: 91.4% accuracy
   - VertexAI: 86.7% accuracy

2. **Tier 2 - Enhanced RAG Systems**
   
   - PaperQA2: 81.9% accuracy
   - Gemini with grounding: 78.3% accuracy

3. **Tier 3 - Standard RAG Systems**
   
   - OpenAI RAG: 72.1% accuracy
   - Hybrid OCR+PDF: 69.8% accuracy

4. **Tier 4 - Baseline Systems**
   
   - No-RAG Gemini: 45.2% accuracy
   - GPT-4o-mini baseline: 18% accuracy

Key Features
------------

* **Multiple AI Backends**: OpenAI GPT-4, Google Gemini, Vertex AI, Perplexity, and more
* **Advanced Document Processing**: Support for PDFs, scientific papers, and various document formats
* **Hybrid RAG Systems**: Combine multiple embeddings and retrieval strategies
* **Comprehensive Evaluation**: Built-in human and AI evaluation frameworks
* **Cost Analysis**: Track and optimize API costs across different providers
* **Scientific Focus**: Optimized for academic papers, research documents, and scientific queries

Processing Time Analysis
------------------------

.. list-table:: System Performance Comparison
   :widths: 25 25 25 25
   :header-rows: 1

   * - System
     - Avg Response Time
     - Questions/Hour
     - Cost per Hour
   * - VertexAI
     - 12.3s
     - 292
     - $0.35
   * - Gemini
     - 8.7s
     - 414
     - $0.52
   * - OpenAI RAG
     - 15.2s
     - 237
     - $3.79
   * - PaperQA2
     - 45.6s
     - 79
     - $2.14

Quick Start
-----------

.. code-block:: bash

   # Clone the repository
   git clone git@gitlab.developers.cam.ac.uk:phy/data-intensive-science-mphil/assessments/projects/xx823.git
   cd xx823
   
   # Install dependencies
   pip install -r requirements.txt
   
   # Set up environment variables
   cp .env.example .env
   # Edit .env with your API keys

Table of Contents
-----------------

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   user_guide/paperqa2_reproduction
   user_guide/scirag_framework

.. toctree::
   :maxdepth: 1
   :caption: Development

   results

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`