import re
from .scirag import SciRag
from .config import paperqa2_settings, AnswerFormat, index_settings
from paperqa import ask
from paperqa.agents.search import get_directory_index
import json

import nest_asyncio
nest_asyncio.apply()
import asyncio
import os


class SciRagPaperQA2(SciRag):
    def __init__(self, 
                 settings=paperqa2_settings, 
                 paper_directory=None, 
                 index_settings=index_settings,
                 index_built=False):
        super().__init__(
            client=None,  # paperqa2 doesn't use a client in the same way
            credentials=None,
            markdown_files_path=None, #paperqa2 cannot handle markdown files for now
            corpus_name=None,
            gen_model=None
        )
        if settings is None:
            self.settings = paperqa2_settings
        else:
            self.settings = settings
        self.paper_directory = paper_directory or settings.paper_directory
        self.index_settings = index_settings
        self._index_built = index_built
        print("[SciRagPaperQA2] Building index on initialization...")
        self.build_index()

    def build_index(self):
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(self.build_index_if_needed())

    async def build_index_if_needed(self):
        if not self._index_built:
            print(f"[SciRagPaperQA2] Checking for paper directory: {self.paper_directory}")
            if not os.path.exists(self.paper_directory):
                raise FileNotFoundError(f"Paper directory not found: {self.paper_directory}")
            print(f"[SciRagPaperQA2] Building PaperQA2 document index (only happens once)...")
            built_index = await get_directory_index(settings=self.index_settings)
            print(f"Using index: {index_settings.get_index_name()}")
            index_files = await built_index.index_files
            print(f"Index files: {index_files}")
            self._index_built = True
            print(f"[SciRagPaperQA2] Index built successfully.")
            return built_index

    def get_response(self, query: str) -> AnswerFormat:
        loop = asyncio.get_event_loop()
        answer_text = loop.run_until_complete(self.query_paperqa(query))
        # Extract sources from the answer text
        answer, sources = self.extract_answer_and_sources(answer_text)
        formatted = self.format_agent_output(answer, sources)
        return AnswerFormat(answer=formatted, sources=sources)

    def extract_answer_and_sources(self, answer_text: str):
        # Find all (filename chunk N) patterns in the answer text
        sources = re.findall(r'\(([^)]+? chunk \d+)\)', answer_text)
        # Remove all (filename chunk N) patterns from the answer text
        answer_cleaned = re.sub(r'\(([^)]+? chunk \d+)\)', '', answer_text)
        answer_cleaned = re.sub(r'\s+', ' ', answer_cleaned).strip()
        return answer_cleaned, sources

    def format_agent_output(self, answer: str, sources: list) -> str:
        sources_str = ", ".join(sources) if sources else "N/A"
        return f"""**Answer**:\n\n{answer}\n\n**Sources**:\n\n{sources_str}\n"""

    async def query_paperqa(self,query: str) -> str:
        """Query PaperQA2 for scientific evidence using OCR-processed documents"""
        response = await ask(query, settings=self.settings)
        return response.model_dump()['session']['answer']
    
    def create_vector_db(self, *args, **kwargs):
        pass

    def delete_vector_db(self, *args, **kwargs):
        pass

    def get_chunks(self, query: str):
        pass

    def cost_dict(self, *args, **kwargs):
        pass
