# ShareGPT Entity Extraction Dataset

This repository contains ShareGPT-formatted datasets for fine-tuning language models on knowledge graph entity and relationship extraction tasks.

## Files

- `sharegpt_entity_extraction_dataset.json` - Manual dataset with 6 diverse training examples
- `extended_sharegpt_entity_extraction_dataset.json` - Generated dataset with additional domain-specific examples  
- `generate_entity_dataset.py` - Python script to generate more training data programmatically

## Dataset Format

The datasets use ShareGPT conversation format with the following structure:

```json
{
  "conversations": [
    {
      "from": "human",
      "value": "Given a text document... [full prompt with instructions and text]"
    },
    {
      "from": "gpt", 
      "value": "(\"entity\"|ENTITY_NAME|ENTITY_TYPE|description)\n<|>\n(\"relationship\"|SOURCE|TARGET|description|strength)\n<|COMPLETE|>"
    }
  ]
}
```

## Entity Types Covered

The datasets include examples with various entity type combinations:
- **PERSON, ORGANIZATION** - Business/corporate contexts
- **ORGANIZATION, TECHNOLOGY** - Tech industry scenarios  
- **PERSON, SKILL, PROJECT** - Professional development contexts
- **ORGANIZATION, GEO, PERSON** - International relations/geopolitics
- **PERSON, ORGANIZATION, EVENT** - Healthcare/conference scenarios
- **TECHNOLOGY, ORGANIZATION, PERSON** - Technology partnerships
- **FINANCIAL_INSTRUMENT, ORGANIZATION, PERSON** - Financial analysis

## Text Domains

Training examples span multiple domains:
- **Technology** - AI/software announcements, product launches
- **Healthcare** - Medical research, clinical studies
- **Finance** - Stock analysis, investment recommendations
- **Business** - Corporate partnerships, organizational changes
- **Geopolitics** - International relations, diplomatic events

## Output Format

The model learns to extract:

### Entities
Format: `("entity"|<NAME>|<TYPE>|<DESCRIPTION>)`
- **entity_name**: Capitalized entity name
- **entity_type**: One of the specified types
- **entity_description**: Comprehensive description of attributes/activities

### Relationships  
Format: `("relationship"|<SOURCE>|<TARGET>|<DESCRIPTION>|<STRENGTH>)`
- **source_entity**: Name of source entity
- **target_entity**: Name of target entity  
- **relationship_description**: Explanation of the relationship
- **relationship_strength**: Numeric score (1-10) indicating relationship strength

## Usage

### Using Pre-built Datasets
```python
import json

# Load the dataset
with open('sharegpt_entity_extraction_dataset.json', 'r') as f:
    dataset = json.load(f)

# Each item contains a conversation with human prompt and GPT response
for example in dataset:
    human_input = example['conversations'][0]['value']
    expected_output = example['conversations'][1]['value']
```

### Generating Custom Examples
```python
from generate_entity_dataset import EntityExtractionDatasetGenerator

generator = EntityExtractionDatasetGenerator()

# Add custom example
custom_example = generator.add_custom_example(
    text="Your custom text here...",
    entity_types=["PERSON", "ORGANIZATION"],
    entities=[("JOHN DOE", "PERSON", "Software engineer at TechCorp")],
    relationships=[("JOHN DOE", "TECHCORP", "John Doe works at TechCorp", 10)]
)

# Generate dataset
dataset = generator.generate_dataset()
generator.save_dataset(dataset, "my_custom_dataset.json")
```

## Fine-tuning Considerations

### Model Selection
- Works well with instruction-tuned models (GPT-3.5/4, Claude, Llama-2-Chat, etc.)
- Recommended minimum size: 7B parameters for good performance

### Training Parameters
- **Learning Rate**: Start with 1e-5 to 5e-5
- **Batch Size**: 4-8 depending on model size and GPU memory
- **Epochs**: 2-5 epochs typically sufficient
- **Sequence Length**: 4096+ tokens recommended (prompts can be long)

### Data Augmentation
Use the generator script to create more examples:
- Vary entity types across domains
- Include different text complexities
- Balance entity-heavy vs. relationship-heavy examples
- Add negative examples (texts with few/no entities)

## Evaluation Metrics

Recommend tracking:
- **Entity Extraction**: Precision, Recall, F1 for each entity type
- **Relationship Extraction**: Precision, Recall, F1 for relationships
- **Format Compliance**: Percentage of outputs following exact format
- **Completion Rate**: Percentage ending with `<|COMPLETE|>` token

## Example Output

Given the input text about Apple's AI announcement, the model should output:
```
("entity"|TIM COOK|PERSON|CEO of Apple who announced iPhone 15 launch)
<|>
("entity"|APPLE|ORGANIZATION|Technology company that developed iPhone 15)
<|>
("entity"|IPHONE 15|TECHNOLOGY|New smartphone with advanced AI capabilities)
<|>
("relationship"|TIM COOK|APPLE|Tim Cook is the CEO of Apple|10)
<|>
("relationship"|APPLE|IPHONE 15|Apple developed the iPhone 15|10)
<|COMPLETE|>
```

## License

This dataset is provided for research and educational purposes. Please ensure compliance with your fine-tuning platform's terms of service.