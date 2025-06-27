import os
import re
import requests
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import json

# Import the base SciRag class
from .scirag import SciRag
from .config import PERPLEXITY_MODEL,AnswerFormat

@dataclass
class Paper:
    """Represents a cosmology paper in our knowledge base"""
    id: int
    title: str
    authors: str
    journal: str
    year: int
    citation: str
    arxiv_id: Optional[str] = None
    doi: Optional[str] = None

@dataclass
class CostInfo:
    """Tracks cost information for API calls"""
    input_tokens: int = 0
    output_tokens: int = 0
    search_queries: int = 0
    total_cost: float = 0.0

class PerplexityAgent(SciRag):
    """A Perplexity-style agent for cosmology questions, inheriting from SciRag"""
    # Perplexity API pricing (as of 2024 - update as needed)
    PRICING = {
        "sonar-pro": {
            "input": 0.001,    # $0.001 per 1K input tokens
            "output": 0.001,   # $0.001 per 1K output tokens
            "search": 0.005    # $0.005 per search query
        },
        "sonar": {
            "input": 0.0002,   # $0.0002 per 1K input tokens  
            "output": 0.0002,  # $0.0002 per 1K output tokens
            "search": 0.005    # $0.005 per search query
        }
    }
    def __init__(self, 
                 client=None,
                 credentials=None, 
                 markdown_files_path=None,
                 corpus_name=None,
                 gen_model=None,
                 **kwargs):
        # Initialize parent SciRag class
        super().__init__(
            client=client,
            credentials=credentials,
            markdown_files_path=markdown_files_path,
            corpus_name=corpus_name,
            gen_model=gen_model,
            **kwargs
        )
        # Initialize cost tracking
        self.session_cost = CostInfo()
        self.total_cost = CostInfo()
        
        # Core paper database - exactly as specified
        self.papers = [
            Paper(
                id=1,
                title="Planck 2018 results. VI. Cosmological parameters",
                authors="Planck Collaboration",
                journal="Astron.Astrophys.",
                year=2020,
                citation="Planck Collaboration, Planck 2018 results. VI. Cosmological parameters, Astron.Astrophys. 641 (2020) A6",
                doi="10.1051/0004-6361/201833910"
            ),
            Paper(
                id=2,
                title="The CAMELS project: Cosmology and Astrophysics with MachinE Learning Simulations",
                authors="Villaescusa-Navarro et al.",
                journal="Astrophys.J.",
                year=2021,
                citation="Villaescusa-Navarro et al., The CAMELS project: Cosmology and Astrophysics with MachinE Learning Simulations, Astrophys.J. 915 (2021) 71",
                arxiv_id="2010.00619"
            ),
            Paper(
                id=3,
                title="Cosmology with one galaxy?",
                authors="Villaescusa-Navarro et al.",
                journal="Astrophys.J.",
                year=2022,
                citation="Villaescusa-Navarro et al., Cosmology with one galaxy? Astrophys.J. 929 (2022) 2, 132",
                arxiv_id="2109.04484"
            ),
            Paper(
                id=4,
                title="A 2.4% Determination of the Local Value of the Hubble Constant",
                authors="Riess et al.",
                journal="Astrophys.J.",
                year=2016,
                citation="Riess et al., A 2.4% Determination of the Local Value of the Hubble Constant, Astrophys.J. 826 (2016) 1, 56",
                arxiv_id="1604.01424"
            ),
            Paper(
                id=5,
                title="The Atacama Cosmology Telescope: DR6 Constraints on Extended Cosmological Models",
                authors="Calabrese et al.",
                journal="arXiv",
                year=2025,
                citation="Calabrese et al., The Atacama Cosmology Telescope: DR6 Constraints on Extended Cosmological Models, arXiv:2503.14454v1 (2025)",
                arxiv_id="2503.14454"
            )
        ]
        
        # Override the rag_prompt with Perplexity-specific prompt
        self.rag_prompt = self._create_perplexity_prompt()
        # Create a mapping of citation numbers to paper info for sources
        self.citation_to_paper = {
            str(i): paper for i, paper in enumerate(self.papers, 1)
        }
    
    def _calculate_cost(self, model: str, usage: Dict) -> float:
        """Calculate cost based on token usage and model pricing"""
        if model not in self.PRICING:
            # Default to sonar pricing if model not found
            model = "sonar"
        
        pricing = self.PRICING[model]
        
        input_tokens = usage.get("prompt_tokens", 0)
        output_tokens = usage.get("completion_tokens", 0)
        
        input_cost = (input_tokens / 1000) * pricing["input"]
        output_cost = (output_tokens / 1000) * pricing["output"]
        search_cost = pricing["search"]  # One search query per API call
        
        total_cost = input_cost + output_cost + search_cost
        
        return total_cost, input_tokens, output_tokens
    def _update_cost_tracking(self, cost: float, input_tokens: int, output_tokens: int):
        """Update both session and total cost tracking"""
        # Update session costs
        self.session_cost.input_tokens += input_tokens
        self.session_cost.output_tokens += output_tokens
        self.session_cost.search_queries += 1
        self.session_cost.total_cost += cost
        
        # Update total costs
        self.total_cost.input_tokens += input_tokens
        self.total_cost.output_tokens += output_tokens
        self.total_cost.search_queries += 1
        self.total_cost.total_cost += cost
    def get_cost_summary(self, session_only: bool = False) -> str:
        """Get a formatted cost summary"""
        cost_info = self.session_cost if session_only else self.total_cost
        period = "Session" if session_only else "Total"
        
        return f"""
{period} Cost Summary:
- Input tokens: {cost_info.input_tokens:,}
- Output tokens: {cost_info.output_tokens:,}
- Search queries: {cost_info.search_queries}
- Total cost: ${cost_info.total_cost:.4f}
"""
    def reset_session_costs(self):
        """Reset session cost tracking"""
        self.session_cost = CostInfo()
    def _create_perplexity_prompt(self) -> str:
        """Create the specialized prompt for Perplexity-style responses"""
        paper_list = "\n".join([f"{i}. {paper.citation}" for i, paper in enumerate(self.papers, 1)])
        
        return f"""You are a scientific literature search agent specializing in cosmology.

We perform retrieval on the following set of papers:
{paper_list}

Your task is to answer questions using ONLY information from these specific papers. 
Do not use any other sources or general knowledge beyond what these papers contain.

Instructions:
1. Search for information relevant to the question within the specified papers
2. Provide a CONCISE answer in EXACTLY 1-3 sentences. Do not exceed 3 sentences under any circumstances.
3. Add numerical references [1], [2], [3], etc. corresponding to the paper numbers listed above
4. If the papers don't contain sufficient information, state this clearly in 1-2 sentences maximum
5. Focus ONLY on the most important quantitative results or key findings
6. Be precise, direct, and avoid any unnecessary elaboration or context

CRITICAL: Your answer section must contain no more than 3 sentences total. Count your sentences carefully.

You must search your knowledge base calling your tool. The sources must be from the retrieval only.

Your response must be in JSON format with exactly these fields:
- "answer": Your 1-3 sentence response with citations
- "sources": Array of citation numbers used (e.g., ["1", "2"])
"""
    
    def _execute_perplexity_query(self, payload: Dict) -> Dict:
        """Execute a query using Perplexity API"""
        api_key = os.getenv("PERPLEXITY_API_KEY")
        if not api_key:
            raise ValueError("PERPLEXITY_API_KEY environment variable not set")
        
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        try:
            response = requests.post(
                "https://api.perplexity.ai/chat/completions",
                headers=headers,
                json=payload
            )
            response.raise_for_status()
            response_data = response.json()
            # Extract usage information and calculate cost
            usage = response_data.get("usage", {})
            model = payload.get("model", "sonar")
            
            cost, input_tokens, output_tokens = self._calculate_cost(model, usage)
            self._update_cost_tracking(cost, input_tokens, output_tokens)
            
            # Log cost information (optional)
            print(f"Query cost: ${cost:.4f} | Input: {input_tokens} tokens | Output: {output_tokens} tokens")
            
            return response.json()
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Error calling Perplexity API: {e}")
    
    def _create_constrained_query(self, question: str) -> str:
        """Create a query that constrains the search to our specific papers"""
        paper_list = "\n".join([f"{i}. {paper.citation}" for i, paper in enumerate(self.papers, 1)])
        
        query = f"""We perform retrieval on the following set of papers:
{paper_list}

Question: {question}

Search for information relevant to this question within the specified papers and provide a comprehensive answer with proper citations [1]-[5]."""
        
        return query
    
    def format_agent_output(self, response: str) -> str:
        """Format agent output with robust JSON parsing and fallback handling."""
        # Debug: Print the response we're trying to parse
        try:
            # Parse the JSON response
            parsed = json.loads(response)
            answer = parsed.get("answer", "")
            source_numbers = parsed.get("sources", [])
            
            # Format sources with full paper information
            formatted_sources = self._format_sources_from_numbers(source_numbers)
            
            return f"""**Answer**:

{answer}

**Sources**:

{formatted_sources}
"""
        
        except json.JSONDecodeError as e:
            print(f"Failed to parse structured JSON response: {e}")
            print(f"Raw response: {response[:300]}...")
            
            # Fallback to original parsing method
            return self.format_agent_output_fallback(response)
        
    def format_agent_output_fallback(self, response: str) -> str:
        """Fallback formatting when structured JSON parsing fails"""
        print(f"Using fallback formatting for response: {response[:200]}...")
        
        # Try to extract answer and sources from markdown-like format
        if "**Answer**:" in response and "**Sources**:" in response:
            parts = response.split("**Sources**:")
            answer = parts[0].replace("**Answer**:", "").strip()
            
            # Format sources with paper info
            sources_formatted = self._format_sources_with_papers(answer)
            
            return f"""**Answer**:

{answer}

**Sources**:

{sources_formatted}
"""
        else:
            # Response doesn't have expected format, treat entire response as answer
            sources_formatted = self._format_sources_with_papers(response)
            
            return f"""**Answer**:

{response}

**Sources**:

{sources_formatted}
""" 
    def _extract_citation_numbers(self, text: str) -> List[str]:
        """Extract citation numbers from text like [1], [2], etc."""
        citations = re.findall(r'\[(\d+)\]', text)
        return list(set(citations))  # Remove duplicates
    def _format_sources_from_numbers(self, source_numbers: List) -> str:
        """Format sources from citation numbers to full paper info - handles various formats"""
        if not source_numbers:
            return "(No sources provided)"
        
        
        # Extract all unique numbers from all source items
        all_numbers = set()
        for source_item in source_numbers:
            # Convert to string and strip brackets: "[1]" -> "1", 1 -> "1", etc.
            num_str = str(source_item).strip('[]')
            if num_str.isdigit():  # Make sure it's a valid number
                all_numbers.add(num_str)
        
        
        # Format each unique number
        formatted_sources = []
        for num_str in sorted(all_numbers, key=int):  # Sort numerically
            if num_str in self.citation_to_paper:
                paper = self.citation_to_paper[num_str]
                formatted_sources.append(f"[{num_str}] {paper.title} - {paper.citation}")
            else:
                formatted_sources.append(f"[{num_str}] Unknown source")
        
        result = "\n".join(formatted_sources) if formatted_sources else "(No valid sources found)"
        return result
        
       
    def _clean_response_for_reasoning_models(self, content: str) -> str:
        """Clean thinking tags from reasoning model responses"""
        if not content:
            return content
        
        # For reasoning models, the JSON comes after the </think> tag
        if "<think>" in content and "</think>" in content:
            # Extract everything after the last </think> tag
            parts = content.split("</think>")
            if len(parts) > 1:
                json_part = parts[-1].strip()
                return json_part
        
        # If no thinking tags, return as is
        return content.strip()
    



    def get_response(self, query: str) -> str:
        """
        Override SciRag's get_response method to use Perplexity API with structured output
        
        Args:
            query: The cosmology question to answer
            model: Perplexity model to use
            
        Returns:
            str: Formatted response with citations
        """
        
        constrained_query = self._create_constrained_query(query)
        
        payload = {
            "model": PERPLEXITY_MODEL,
            "messages": [
                {
                    "role": "system",
                    "content": "You are a precise scientific literature assistant. Follow instructions exactly and provide accurate, well-cited responses in the specified JSON format."
                },
                {
                    "role": "user",
                    "content": constrained_query
                }
            ],
            "search_domain_filter": ["arxiv.org", "adsabs.harvard.edu"],
            "search_recency_filter": "month",
            "temperature": 0.01,
            "max_tokens": 2000,
            # Add structured output formatting
            "response_format": {
                "type": "json_schema",
                "json_schema": {"schema": AnswerFormat.model_json_schema()}
            }
        }
        
        perplexity_response = self._execute_perplexity_query(payload)
        
        content = perplexity_response["choices"][0]["message"]["content"]
        
        # Clean any thinking tags if using reasoning models
        cleaned_response = self._clean_response_for_reasoning_models(content)
        
        # Parse the structured JSON response
        final_response = self.parse_structured_response(cleaned_response)
        
        return final_response
    
    def parse_structured_response(self, json_response: str) -> str:
        """Parse the structured JSON response and format it properly"""
        try:
            # Parse the JSON response
            parsed = json.loads(json_response)
            answer = parsed.get("answer", "")
            source_info = parsed.get("sources", [])  # Default to empty list
            
            # Debug: Print what we received
            print(f"Parsed answer: {answer[:100]}...")
            print(f"Parsed sources: {source_info}")
            print(f"Source info type: {type(source_info)}")
            
            # Ensure source_info is a list
            if isinstance(source_info, str):
                # If it's a string like "[1], [2]" or "1,2", try to parse it
                if source_info.strip():
                    # Extract numbers from string format
                    numbers = re.findall(r'\d+', source_info)
                    source_info = numbers  # Keep as plain numbers, not "[1]" format
                else:
                    source_info = []
            elif not isinstance(source_info, list):
                # Convert other types to list
                source_info = [source_info] if source_info else []
            
            print(f"Processed sources: {source_info}")
            
            # Format sources with full paper information
            formatted_sources = self._format_sources_from_numbers(source_info)
            
            return f"""**Answer**:

{answer}

**Sources**:

{formatted_sources}
"""
        
        except json.JSONDecodeError as e:
            print(f"Failed to parse structured JSON response: {e}")
            print(f"Raw response: {json_response[:300]}...")
            
            # Fallback to original parsing method
            return self.format_agent_output_fallback(json_response)
    
    def _format_response_with_links(self, response: str, citations: List[str]) -> str:
        """Format the response with clickable citation links"""
        def citation_repl(match):
            number_str = match.group(1)
            try:
                index = int(number_str) - 1
                if 0 <= index < len(citations):
                    return f'[[{number_str}]({citations[index]})]'
                else:
                    # Map to our internal papers if citation number matches
                    paper_index = int(number_str) - 1
                    if 0 <= paper_index < len(self.papers):
                        paper = self.papers[paper_index]
                        if paper.arxiv_id:
                            return f'[[{number_str}](https://arxiv.org/abs/{paper.arxiv_id})]'
                        elif paper.doi:
                            return f'[[{number_str}](https://doi.org/{paper.doi})]'
                return match.group(0)
            except (ValueError, IndexError):
                return match.group(0)
        
        return re.sub(r'\[(\d+)\]', citation_repl, response)
    
    def _format_sources_with_papers(self, response: str) -> str:
        """Format sources section with paper names and citations"""
        citation_numbers = self._extract_citation_numbers(response)
        
        if not citation_numbers:
            return "(Sources not found in standard format)"
        
        formatted_sources = []
        for num in sorted(citation_numbers, key=int):
            if num in self.citation_to_paper:
                paper = self.citation_to_paper[num]
                formatted_sources.append(f"[{num}] {paper.title} - {paper.citation}")
        
        return "\n".join(formatted_sources) if formatted_sources else "(Sources not found in standard format)"

    
    

