PaperQA2 Reproduction Guide
===========================

This guide covers the complete reproduction of the PaperQA2 framework on the LitQA2 biology benchmark.

Overview
--------

Our reproduction study validates the PaperQA2 framework's performance claims and identifies critical factors affecting generalization to new scientific domains.

Key Findings
~~~~~~~~~~~~

* **71% accuracy achieved** on LitQA2 biology benchmark (matching original paper)
* **40% performance gap** between training (30%) and test (18%) data for GPT-4o-mini baseline
* **RAG systems crucial** for handling recent literature effectively

Running the Reproduction
-------------------------

Basic Reproduction
~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Navigate to reproduction directory
   cd Reproduction/Inspect_AI

   # Run basic evaluation
   inspect eval eval_pipeline.py --max-connections=1

Parameter Customization
~~~~~~~~~~~~~~~~~~~~~~~

Key parameters in ``eval_pipeline.py``:

1. **Evidence Settings**:

.. code-block:: python

   answer=AnswerSettings(
       evidence_k=30,                    # Number of evidence pieces (1, 5, 10, 15, 30)
       answer_max_sources=15,            # Max sources per question (5, 15)
       evidence_skip_summary=False       # Skip evidence summarization
   )

2. **Model Configuration**:

.. code-block:: python

   settings = Settings(
       llm="gpt-4o-mini",               # "gpt-4o-mini", "gpt-4o", "gpt-4.1"
       summary_llm="gpt-4o-mini",       # Summary agent model
       agent=AgentSettings(
           agent_llm="gpt-4o-mini",     # Search agent model
       )
   )

Ablation Studies
~~~~~~~~~~~~~~~~

Study 1: Evidence Depth Analysis
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Test different evidence retrieval depths:

.. code-block:: python

   # Configuration for evidence_k=1
   answer=AnswerSettings(evidence_k=1, answer_max_sources=15, evidence_skip_summary=False)

   # Configuration for evidence_k=5  
   answer=AnswerSettings(evidence_k=5, answer_max_sources=15, evidence_skip_summary=False)

   # Configuration for evidence_k=10
   answer=AnswerSettings(evidence_k=10, answer_max_sources=15, evidence_skip_summary=False)

Study 2: Model Component Analysis
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Compare different model configurations:

.. code-block:: python

   # All GPT-4o configuration
   settings = Settings(
       llm="gpt-4o",
       summary_llm="gpt-4o", 
       agent=AgentSettings(agent_llm="gpt-4o")
   )

   # Search agent only GPT-4o
   settings = Settings(
       llm="gpt-4o-mini",
       summary_llm="gpt-4o-mini",
       agent=AgentSettings(agent_llm="gpt-4o")  # Only search agent upgraded
   )

Results Analysis
----------------

Use the provided notebooks for comprehensive analysis:

.. code-block:: bash

   # PaperQA2 reproduction analysis
   jupyter notebook notebooks/reproduction_eval.ipynb

   # Multiple choice evaluation demo
   jupyter notebook notebooks/Multiple_Choice_Evaluation_2a.ipynb

Troubleshooting
---------------

Common Issues
~~~~~~~~~~~~~

1. **Index Building Failures**:
   
   - Ensure sufficient OpenAI API quota
   - Reduce ``--max-connections`` if hitting rate limits
   - Clear corrupted index: ``rm -rf /Users/apple/.pqa/indexes/*``

2. **API Quota Issues**:
   
   - Monitor usage at https://platform.openai.com/usage
   - Recommended minimum quota: $50 for full reproduction

3. **Rate Limiting**:
   
   - Use ``--max-connections=1`` for conservative rate limiting
   - Add delays between API calls if needed

Error Solutions
~~~~~~~~~~~~~~~

.. list-table:: Common Errors and Solutions
   :widths: 30 70
   :header-rows: 1

   * - Error
     - Solution
   * - ``RateLimitError``
     - Reduce connections, check API quota
   * - ``IndexBuildError``
     - Clear cache, ensure disk space, retry
   * - ``QuotaExceededError``
     - Add funds to OpenAI account

Performance Metrics
-------------------

The reproduction achieves:

* **71% accuracy** on test set (matching original paper)
* **Consistent performance** across different model configurations
* **Effective evidence integration** with optimal evidence_k=15-30

Advanced Configuration
----------------------

Custom Dataset Processing
~~~~~~~~~~~~~~~~~~~~~~~~~

For processing custom datasets:

.. code-block:: python

   from paperqa import Settings, ask
   from paperqa.contrib.inspect_ai import PaperQAEvaluator

   # Custom settings for your domain
   settings = Settings(
       llm="gpt-4o-mini",
       summary_llm="gpt-4o-mini",
       embedding="text-embedding-3-small",
       answer=AnswerSettings(
           evidence_k=20,
           answer_max_sources=10,
           evidence_skip_summary=False
       )
   )

   # Initialize evaluator
   evaluator = PaperQAEvaluator(
       config=settings,
       dataset_path="path/to/your/dataset.json"
   )

Index Management
~~~~~~~~~~~~~~~~

Best practices for managing PaperQA2 indexes:

.. code-block:: bash

   # Check index status
   ls -la ~/.pqa/indexes/

   # Backup important indexes
   cp -r ~/.pqa/indexes/ ~/paperqa_backup/

   # Clean corrupted indexes
   rm -rf ~/.pqa/indexes/corrupted_index_name

   # Rebuild index from scratch
   python -c "from paperqa import build_index; build_index('path/to/papers')"

Cost Optimization
~~~~~~~~~~~~~~~~~

Strategies to minimize API costs:

1. **Start with smaller evidence_k values** (5-10) for initial testing
2. **Use gpt-4o-mini** for all components during development
3. **Implement caching** for repeated queries
4. **Batch process** multiple questions when possible

.. code-block:: python

   # Cost-optimized configuration
   cost_optimized_settings = Settings(
       llm="gpt-4o-mini",           # Cheapest model
       summary_llm="gpt-4o-mini",   # Use same model
       embedding="text-embedding-3-small",  # Cheaper embedding
       answer=AnswerSettings(
           evidence_k=10,           # Reduced evidence
           answer_max_sources=5,    # Fewer sources
           evidence_skip_summary=True  # Skip expensive summarization
       )
   )

Reproducibility Checklist
-------------------------

To ensure reproducible results:

.. checklist::

   * Set random seeds in configuration
   * Use specific model versions (not latest)
   * Document exact API versions used
   * Save all configuration files
   * Log all API calls and responses
   * Use deterministic processing where possible

.. code-block:: python

   # Reproducible configuration
   reproducible_settings = Settings(
       llm="gpt-4o-mini-2024-07-18",  # Specific version
       seed=42,                       # Fixed random seed
       temperature=0,                 # Deterministic output
   )

Integration with Other Tools
----------------------------

Using PaperQA2 with External Evaluation Frameworks
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Integration with inspect_ai
   from inspect_ai import eval, Task
   from paperqa.contrib.inspect_ai import paperqa_task

   # Create evaluation task
   task = paperqa_task(
       dataset="biology_questions.json",
       settings=your_settings
   )

   # Run evaluation
   results = eval(task, model="paperqa2")

