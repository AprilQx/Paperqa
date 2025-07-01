SciRAG Framework Guide
======================

This guide covers the comprehensive SciRAG framework for multi-platform scientific literature analysis.

Overview
--------

SciRAG (Scientific Retrieval-Augmented Generation) is a unified framework that enables comparative evaluation of RAG systems across multiple AI platforms and scientific domains.

Key Features
~~~~~~~~~~~~

* **Multi-Platform Support**: OpenAI GPT-4, Google Gemini, Vertex AI, PaperQA2
* **Advanced Document Processing**: PDF extraction, OCR capabilities, scientific paper handling
* **Hybrid RAG Systems**: Combine multiple embeddings and retrieval strategies
* **Domain-Specific Optimization**: Tailored for biology, astronomy, and other scientific fields
* **Cost-Performance Analysis**: Comprehensive tracking and optimization tools

Installation
------------

Install dependencies::

    pip install -e .

Configure API keys by creating a ``.env`` file in the root directory::

    OPENAI_API_KEY=your_openai_api_key_here
    GOOGLE_API_KEY=your_google_api_key_here
    MISTRAL_API_KEY=your_mistral_api_key_here
    PERPLEXITY_API_KEY=your_perplexity_api_key_here

Platform Implementations
------------------------

OpenAI GPT-4 Integration
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    import os
    from dotenv import load_dotenv
    from scirag import SciRagDataSet

    load_dotenv()
    api_key = os.environ.get("OPENAI_API_KEY")

    # Initialize OpenAI RAG
    from scirag import SciRagOpenAI
    scirag = SciRagOpenAI(openai_api_key=api_key)

    # Load dataset and ask questions
    dataset = SciRagDataSet()
    qa = dataset.load_dataset()

    # Ask a question
    qid = 4
    question = qa['question'].iloc[qid]
    response = scirag.get_response(question)
    print(response)

Google Gemini Integration
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    import os
    from dotenv import load_dotenv
    from scirag import SciRagDataSet
    
    load_dotenv()
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "gemini.json"

    from scirag import SciRagHybrid
    scirag = SciRagHybrid(
        google_api_key=os.environ.get("GOOGLE_API_KEY")
    )

    # Load dataset and ask questions
    dataset = SciRagDataSet()
    qa = dataset.load_dataset()

    # Create vector database and query
    scirag.create_vector_db()
    
    # Ask a question
    qid = 4
    question = qa['question'].iloc[qid]
    response = scirag.get_response(question)
    print(response)

Vertex AI Integration
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from scirag import SciRagVertexAI
    scirag = SciRagVertexAI()
    from scirag import SciRagDataSet

    # Enhanced query processing
    question = "How large is the impact of beam window functions on the 2018 spectra?"
    enhanced_question = scirag.enhanced_query(question)
    response = scirag.get_response(question)

PaperQA2 Integration
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from scirag import SciRagPaperQA2
    paperqa = SciRagPaperQA2()

    # Load dataset and ask questions
    dataset = SciRagDataSet()
    qa = dataset.load_dataset()

    # Ask a question
    qid = 4
    question = qa['question'].iloc[qid]

    # Process with timing
    import time
    start_time = time.time()
    response = paperqa.get_response(question)
    processing_time = time.time() - start_time

Perplexity Agent
~~~~~~~~~~~~~~~~

.. code-block:: python

    import os
    from dotenv import load_dotenv
    from scirag import PerplexityAgent

    load_dotenv()

    # Initialize Perplexity
    perplexity = PerplexityAgent(
        api_key=os.environ.get("PERPLEXITY_API_KEY")
    )
    
    # Load dataset and ask questions
    dataset = SciRagDataSet()
    qa = dataset.load_dataset()

    # Ask a question
    qid = 4
    question = qa['question'].iloc[qid]

    # Get web-enhanced response
    response = perplexity.get_response(question)
    print(response)

Gemini Grounded Agent
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    import os
    from dotenv import load_dotenv
    
    load_dotenv()

    # Note: You might not need OPENAI_API_KEY for Gemini
    google_api_key = os.environ.get("GOOGLE_API_KEY")
    if google_api_key is None:
        raise RuntimeError("Please set the GOOGLE_API_KEY environment variable")

    # Initialize Agent
    from scirag import GeminiGroundedAgent, SciRagDataSet

    scirag = GeminiGroundedAgent(api_key=google_api_key)
    dataset = SciRagDataSet()

    # Load Dataset
    qa = dataset.load_dataset()

    # Query with Timing
    qid = 4
    question = qa['question'].iloc[qid]
    response = scirag.get_response(question)

    # Display Question
    print(f"**Question:**\n{question}")

    # Display Response
    print(f"**Grounded Response:**\n{response}")

    # Compare with Ideal
    ideal_answer = qa['ideal'].iloc[qid]
    print(f"**Ideal Answer:**\n{ideal_answer}")

    # Cost Summary
    summary = scirag.get_cost_summary()
    print(f"Total cost: ${summary['total_cost']:.6f}")
    print(f"Total tokens: {summary['total_tokens']}")
    print(f"Grounding enabled: {summary['grounding_enabled']}")

Performance Optimization
-----------------------

Cost-Performance Trade-offs
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Different platforms offer varying cost-performance characteristics:

* **VertexAI**: Most cost-effective at $0.12 per 100 questions
* **Gemini**: Balanced performance and cost
* **OpenAI GPT-4**: Premium accuracy at higher cost
* **PaperQA2**: Research-focused with good citation support

Best Practices
~~~~~~~~~~~~~~

1. **Choose appropriate models** for different complexity levels
2. **Implement caching** for repeated queries
3. **Use batch processing** when possible
4. **Monitor costs** with built-in tracking
5. **Validate results** with human evaluation

Advanced Features
-----------------

Custom Evaluation Metrics
~~~~~~~~~~~~~~~~~~~~~~~~~

The framework includes comprehensive evaluation capabilities:

* Accuracy assessment across domains
* Citation quality analysis
* Response time optimization
* Cost-per-question tracking

Batch Processing Example
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from scirag import SciRagEvaluator
    import pandas as pd

    # Initialize evaluator for multiple platforms
    evaluator = SciRagEvaluator()
    
    # Load your question dataset
    questions_df = pd.read_json("your_questions.json")
    
    # Evaluate across multiple platforms
    platforms = ["openai", "gemini", "vertexai", "paperqa2"]
    results = {}
    
    for platform in platforms:
        print(f"Evaluating {platform}...")
        platform_results = evaluator.evaluate_platform(
            platform=platform,
            questions=questions_df,
            max_questions=100  # Limit for cost control
        )
        results[platform] = platform_results
        
    # Compare results
    comparison_df = evaluator.compare_platforms(results)
    print(comparison_df)

Integration Examples
~~~~~~~~~~~~~~~~~~~

See the ``notebooks/`` directory for complete examples:

* Basic usage patterns
* Advanced RAG configurations
* Cost optimization strategies
* Domain-specific adaptations

Troubleshooting
---------------

Common Issues
~~~~~~~~~~~~~

1. **API Key Issues**:
   
   * Ensure all required API keys are set in ``.env``
   * Check API key permissions and quotas
   * Verify environment variable loading

2. **Rate Limiting**:
   
   * Implement exponential backoff
   * Use appropriate delays between requests
   * Consider using multiple API keys

3. **Memory Issues**:
   
   * Process documents in smaller batches
   * Clear vector databases when not needed
   * Monitor memory usage during large evaluations

Error Resolution
~~~~~~~~~~~~~~~

.. code-block:: python

    # Example error handling
    import time
    import logging
    from scirag import SciRagOpenAI

    def robust_query(scirag_instance, question, max_retries=3):
        """Query with automatic retry on failures"""
        for attempt in range(max_retries):
            try:
                response = scirag_instance.get_response(question)
                return response
            except Exception as e:
                logging.warning(f"Attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                else:
                    raise e

Configuration Templates
-----------------------

Development Configuration
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    # Fast, low-cost configuration for development
    dev_config = {
        "platform": "gemini",
        "model": "gemini-pro",
        "max_questions": 10,
        "evidence_k": 5,
        "batch_size": 1
    }

Production Configuration
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    # High-accuracy configuration for production
    prod_config = {
        "platform": "openai",
        "model": "gpt-4o",
        "max_questions": 1000,
        "evidence_k": 30,
        "batch_size": 5,
        "validation_enabled": True
    }

Research Configuration
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    # Comprehensive evaluation for research
    research_config = {
        "platforms": ["openai", "gemini", "vertexai", "paperqa2"],
        "models": ["gpt-4o", "gpt-4o-mini", "gemini-pro"],
        "evidence_depths": [5, 10, 15, 30],
        "human_evaluation": True,
        "statistical_analysis": True
    }


