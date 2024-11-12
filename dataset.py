from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import pandas as pd
import logging
from pathlib import Path
from torch.utils.data import Dataset
from transformers import AutoTokenizer

@dataclass
class MIMICSExample:
    query: str
    question: str
    facets: List[str]
    evidence_documents: List[str]  # We'll use the options as evidence
    total_users: Optional[int] = None

class MIMICSDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        retriever_type: str = "bing",
        max_documents: int = 10,
        tokenizer_name: str = "facebook/bart-base",
        skip_template_questions: bool = True
    ):
        self.split = split
        self.max_documents = max_documents
        self.retriever_type = retriever_type
        self.skip_template_questions = skip_template_questions
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        
        if split == "train":
            self.data_path = Path(data_dir) / "mimics-click.tsv"
        elif split == "test":
            self.data_path = Path(data_dir) / "mimics-manual.tsv"
        else:
            raise ValueError(f"Unknown split: {split}")
            
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data file not found: {self.data_path}")
            
        self.examples = self._load_data()
        logging.info(f"Loaded {len(self.examples)} examples from {self.data_path}")
        
        self.setup_retriever()
        
    def _load_data(self) -> List[MIMICSExample]:
        examples = []
        skipped_template = 0
        skipped_no_options = 0
        total_processed = 0
        
        try:
            # Read dataset
            df = pd.read_csv(self.data_path, sep='\t', encoding='utf-8')
            logging.info(f"Loaded {len(df)} rows from {self.data_path}")
            
            # Process each row
            for idx, row in df.iterrows():
                total_processed += 1
                
                try:
                    # Check template questions
                    if self.skip_template_questions and row.get('question') == "Select one to refine your search":
                        skipped_template += 1
                        continue
                    
                    # Extract options (facets)
                    facets = []
                    evidence = []
                    for i in range(1, 6):
                        option = row.get(f'option_{i}')
                        if pd.notna(option) and str(option).strip():
                            facet = str(option).strip()
                            facets.append(facet)
                            # Create evidence document from query and option
                            evidence.append(f"Query: {row['query']} Option: {facet}")
                    
                    # Skip if no valid options
                    if not facets:
                        skipped_no_options += 1
                        continue
                    
                    # Create example
                    example = MIMICSExample(
                        query=str(row['query']).strip(),
                        question=str(row.get('question', '')).strip(),
                        facets=facets,
                        evidence_documents=evidence
                    )
                    examples.append(example)
                    
                    # Print progress
                    if len(examples) % 10000 == 0:
                        logging.info(f"Processed {total_processed} rows, created {len(examples)} examples")
                
                except Exception as e:
                    logging.error(f"Error processing row {idx}: {e}")
                    continue
        
        except Exception as e:
            logging.error(f"Error reading TSV file: {e}")
            raise
            
        logging.info(f"""
        Data loading summary:
        - Total rows processed: {total_processed}
        - Template questions skipped: {skipped_template}
        - Rows skipped due to no options: {skipped_no_options}
        - Valid examples created: {len(examples)}
        """)
        
        if not examples:
            logging.warning("No valid examples loaded from dataset!")
        else:
            # Print first example for debugging
            first_ex = examples[0]
            logging.info(f"""
            Sample example:
            Query: {first_ex.query}
            Question: {first_ex.question}
            Facets: {first_ex.facets}
            Evidence: {first_ex.evidence_documents}
            """)
            
        return examples

    def _retrieve_bing_snippets(self, query: str, facets: List[str] = None) -> List[str]:
        """Return the evidence documents for the query"""
        idx = next(i for i, ex in enumerate(self.examples) if ex.query == query)
        return self.examples[idx].evidence_documents[:self.max_documents]
    
    def __len__(self) -> int:
        return len(self.examples)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        example = self.examples[idx]
        
        # For this dataset, we'll use the options directly as evidence
        evidence = self.retrieve(
            query=example.query,
            facets=example.facets if self.retriever_type.endswith('|Q,F') else None
        )
        
        return {
            'query': example.query,
            'question': example.question,
            'facets': example.facets,
            'evidence_documents': evidence,
        }
        
    def setup_retriever(self):
        self.retrieve = self._retrieve_bing_snippets  # For now, just use the options as evidence