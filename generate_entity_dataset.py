#!/usr/bin/env python3
"""
ShareGPT Dataset Generator for Entity and Relationship Extraction
This script generates training data in ShareGPT format for finetuning models 
on knowledge graph entity extraction tasks.
"""

import json
import random
from typing import List, Dict, Any, Tuple

class EntityExtractionDatasetGenerator:
    def __init__(self):
        # Define entity type combinations
        self.entity_type_combinations = [
            ["PERSON", "ORGANIZATION"],
            ["ORGANIZATION", "TECHNOLOGY"],
            ["PERSON", "SKILL", "PROJECT"],
            ["ORGANIZATION", "GEO", "PERSON"],
            ["PERSON", "ORGANIZATION", "EVENT"],
            ["TECHNOLOGY", "ORGANIZATION", "PERSON"],
            ["LOCATION", "ORGANIZATION", "PRODUCT"],
            ["PERSON", "ACADEMIC_FIELD", "INSTITUTION"],
            ["COMPANY", "PERSON", "FINANCIAL_INSTRUMENT"],
            ["GOVERNMENT", "POLICY", "CITIZEN_GROUP"]
        ]
        
        # Sample texts with different domains and complexity levels
        self.sample_texts = [
            # Technology domain
            {
                "text": "Microsoft's CEO Satya Nadella announced the integration of OpenAI's GPT-4 technology into Microsoft Office applications. The partnership between Microsoft and OpenAI, valued at $10 billion, will bring advanced AI capabilities to Word, Excel, and PowerPoint. Nadella emphasized that the neural networks will transform productivity software.",
                "entity_types": ["PERSON", "ORGANIZATION", "TECHNOLOGY"],
                "expected_entities": [
                    ("SATYA NADELLA", "PERSON", "CEO of Microsoft who announced GPT-4 integration"),
                    ("MICROSOFT", "ORGANIZATION", "Technology company integrating AI into Office applications"),
                    ("OPENAI", "ORGANIZATION", "AI research company that developed GPT-4"),
                    ("GPT-4", "TECHNOLOGY", "Advanced AI language model being integrated into Office"),
                    ("MICROSOFT OFFICE", "TECHNOLOGY", "Productivity software suite getting AI integration"),
                    ("WORD", "TECHNOLOGY", "Word processing application receiving AI features"),
                    ("EXCEL", "TECHNOLOGY", "Spreadsheet application receiving AI features"),
                    ("POWERPOINT", "TECHNOLOGY", "Presentation software receiving AI features"),
                    ("NEURAL NETWORKS", "TECHNOLOGY", "AI technology that powers the productivity features")
                ],
                "expected_relationships": [
                    ("SATYA NADELLA", "MICROSOFT", "Satya Nadella is the CEO of Microsoft", 10),
                    ("MICROSOFT", "OPENAI", "Microsoft has a $10 billion partnership with OpenAI", 9),
                    ("OPENAI", "GPT-4", "OpenAI developed the GPT-4 technology", 10),
                    ("MICROSOFT", "MICROSOFT OFFICE", "Microsoft owns and develops Microsoft Office", 10),
                    ("GPT-4", "MICROSOFT OFFICE", "GPT-4 is being integrated into Microsoft Office", 9),
                    ("MICROSOFT OFFICE", "WORD", "Word is part of Microsoft Office suite", 10),
                    ("MICROSOFT OFFICE", "EXCEL", "Excel is part of Microsoft Office suite", 10),
                    ("MICROSOFT OFFICE", "POWERPOINT", "PowerPoint is part of Microsoft Office suite", 10)
                ]
            },
            # Healthcare domain
            {
                "text": "Dr. Jennifer Walsh from Johns Hopkins Hospital led a groundbreaking study on CRISPR gene editing published in Nature Medicine. The research involved collaboration with Stanford University and received funding from the National Institutes of Health. Walsh's team successfully treated genetic disorders using advanced gene therapy techniques.",
                "entity_types": ["PERSON", "ORGANIZATION", "TECHNOLOGY"],
                "expected_entities": [
                    ("JENNIFER WALSH", "PERSON", "Doctor who led CRISPR gene editing study"),
                    ("JOHNS HOPKINS HOSPITAL", "ORGANIZATION", "Medical institution where Dr. Walsh works"),
                    ("CRISPR", "TECHNOLOGY", "Gene editing technology used in the study"),
                    ("NATURE MEDICINE", "ORGANIZATION", "Scientific journal that published the research"),
                    ("STANFORD UNIVERSITY", "ORGANIZATION", "University that collaborated on the research"),
                    ("NATIONAL INSTITUTES OF HEALTH", "ORGANIZATION", "Government agency that funded the research"),
                    ("GENE THERAPY", "TECHNOLOGY", "Medical technique for treating genetic disorders")
                ],
                "expected_relationships": [
                    ("JENNIFER WALSH", "JOHNS HOPKINS HOSPITAL", "Dr. Jennifer Walsh works at Johns Hopkins Hospital", 10),
                    ("JENNIFER WALSH", "CRISPR", "Dr. Walsh led a study on CRISPR gene editing", 9),
                    ("JOHNS HOPKINS HOSPITAL", "STANFORD UNIVERSITY", "Johns Hopkins collaborated with Stanford on the research", 8),
                    ("NATIONAL INSTITUTES OF HEALTH", "JENNIFER WALSH", "NIH provided funding for Dr. Walsh's research", 8),
                    ("CRISPR", "GENE THERAPY", "CRISPR is used as a gene therapy technique", 9),
                    ("NATURE MEDICINE", "JENNIFER WALSH", "Nature Medicine published Dr. Walsh's study", 8)
                ]
            },
            # Finance domain
            {
                "text": "Goldman Sachs analyst Maria Rodriguez issued a buy recommendation for Tesla stock following the company's Q3 earnings report. CEO Elon Musk announced record deliveries and plans for expansion into emerging markets. The investment bank raised its price target from $200 to $250 per share.",
                "entity_types": ["PERSON", "ORGANIZATION", "FINANCIAL_INSTRUMENT"],
                "expected_entities": [
                    ("GOLDMAN SACHS", "ORGANIZATION", "Investment bank that issued stock recommendation"),
                    ("MARIA RODRIGUEZ", "PERSON", "Analyst at Goldman Sachs who recommended Tesla stock"),
                    ("TESLA", "ORGANIZATION", "Electric vehicle company with strong Q3 earnings"),
                    ("TESLA STOCK", "FINANCIAL_INSTRUMENT", "Publicly traded shares of Tesla company"),
                    ("ELON MUSK", "PERSON", "CEO of Tesla who announced record deliveries"),
                    ("Q3 EARNINGS REPORT", "FINANCIAL_INSTRUMENT", "Quarterly financial disclosure showing company performance")
                ],
                "expected_relationships": [
                    ("MARIA RODRIGUEZ", "GOLDMAN SACHS", "Maria Rodriguez is an analyst at Goldman Sachs", 10),
                    ("GOLDMAN SACHS", "TESLA STOCK", "Goldman Sachs issued a buy recommendation for Tesla stock", 9),
                    ("ELON MUSK", "TESLA", "Elon Musk is the CEO of Tesla", 10),
                    ("TESLA", "TESLA STOCK", "Tesla stock represents ownership in Tesla company", 10),
                    ("TESLA", "Q3 EARNINGS REPORT", "Tesla published Q3 earnings report showing record performance", 9),
                    ("GOLDMAN SACHS", "TESLA", "Goldman Sachs raised price target for Tesla", 8)
                ]
            }
        ]
    
    def create_conversation(self, text_data: Dict) -> Dict[str, Any]:
        """Create a single ShareGPT conversation from text data."""
        entity_types_str = ",".join(text_data["entity_types"])
        
        # Create the human prompt
        human_prompt = f"""Given a text document that is potentially relevant to this activity and a list of entity types, identify all entities of those types from the text and all relationships among the identified entities.

Steps:
1. Identify all entities. For each identified entity, extract the following information:
- entity_name: Name of the entity, capitalized
- entity_type: One of the following types: [{entity_types_str}]
- entity_description: Comprehensive description of the entity's attributes and activities
Format each entity as ("entity"|<entity_name>|<entity_type>|<entity_description>)

2. From the entities identified in step 1, identify all pairs of (source_entity, target_entity) that are *clearly related* to each other.
For each pair of related entities, extract the following information:
- source_entity: name of the source entity, as identified in step 1
- target_entity: name of the target entity, as identified in step 1
- relationship_description: explanation as to why you think the source entity and the target entity are related to each other
- relationship_strength: a numeric score indicating strength of the relationship between the source entity and target entity
Format each relationship as ("relationship"|<source_entity>|<target_entity>|<relationship_description>|<relationship_strength>)

3. Return output in English as a single list of all the entities and relationships identified in steps 1 and 2. Use **<|>** as the list delimiter.

4. When finished, output <|COMPLETE|>

Entity_types: {entity_types_str}
Text:
{text_data["text"]}"""
        
        # Create the GPT response
        response_parts = []
        
        # Add entities
        for entity_name, entity_type, entity_description in text_data["expected_entities"]:
            response_parts.append(f'("entity"|{entity_name}|{entity_type}|{entity_description})')
        
        # Add relationships
        for source, target, description, strength in text_data["expected_relationships"]:
            response_parts.append(f'("relationship"|{source}|{target}|{description}|{strength})')
        
        gpt_response = "\n<|>\n".join(response_parts) + "\n<|COMPLETE|>"
        
        return {
            "conversations": [
                {
                    "from": "human",
                    "value": human_prompt
                },
                {
                    "from": "gpt", 
                    "value": gpt_response
                }
            ]
        }
    
    def generate_dataset(self, num_examples: int | None = None) -> List[Dict[str, Any]]:
        """Generate a complete ShareGPT dataset."""
        if num_examples is None:
            # Use all sample texts
            examples_to_use = self.sample_texts
        else:
            # Randomly sample the requested number of examples
            examples_to_use = random.sample(self.sample_texts, min(num_examples, len(self.sample_texts)))
        
        dataset = []
        for text_data in examples_to_use:
            conversation = self.create_conversation(text_data)
            dataset.append(conversation)
        
        return dataset
    
    def save_dataset(self, dataset: List[Dict[str, Any]], filename: str) -> None:
        """Save the dataset to a JSON file."""
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(dataset, f, indent=2, ensure_ascii=False)
        print(f"Dataset saved to {filename}")
    
    def add_custom_example(self, text: str, entity_types: List[str], 
                          entities: List[Tuple[str, str, str]], 
                          relationships: List[Tuple[str, str, str, int]]) -> Dict[str, Any]:
        """Add a custom training example."""
        text_data = {
            "text": text,
            "entity_types": entity_types,
            "expected_entities": entities,
            "expected_relationships": relationships
        }
        return self.create_conversation(text_data)

def main():
    """Generate and save the ShareGPT dataset."""
    generator = EntityExtractionDatasetGenerator()
    
    # Generate dataset with all examples
    dataset = generator.generate_dataset()
    
    # Save to file
    generator.save_dataset(dataset, "extended_sharegpt_entity_extraction_dataset.json")
    
    print(f"Generated {len(dataset)} training examples")
    print("Entity types covered:")
    all_entity_types = set()
    for text_data in generator.sample_texts:
        all_entity_types.update(text_data["entity_types"])
    for entity_type in sorted(all_entity_types):
        print(f"  - {entity_type}")

if __name__ == "__main__":
    main()