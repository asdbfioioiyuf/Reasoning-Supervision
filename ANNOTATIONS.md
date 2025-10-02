# Annotation File Format

This document describes the format for spatial attention annotations used for training.

## File Format

Annotations are stored in JSON format with the following structure:

```json
{
  "video_id_1": {
    "0": {
      "text": "{ \"key_regions\": [[row1, col1], [row2, col2], ...] }"
    },
    "1": {
      "text": "{ \"key_regions\": [[row1, col1], ...] }"
    },
    ...
  },
  "video_id_2": { ... }
}
```

## Structure Details

### Top Level
- **Keys**: Video identifiers (e.g., filename without extension)
- **Values**: Dictionary of frame-level annotations

### Frame Level
- **Keys**: Frame indices as strings (e.g., "0", "1", "2", ...)
- **Values**: Dictionary containing annotation data

### Annotation Data
- **"text"**: JSON string containing spatial coordinates
  - **"key_regions"**: List of [row, col] coordinate pairs
  - Coordinates are in a 4×4 grid:
    - Rows: 0-3 (top to bottom)
    - Columns: 0-3 (left to right)

## Example

```json
{
  "basketball_shot_001": {
    "0": {
      "text": "{ \"key_regions\": [[1, 2], [2, 2]] }"
    },
    "1": {
      "text": "{ \"key_regions\": [[1, 1], [1, 2], [2, 2]] }"
    },
    "2": {
      "text": "{ \"key_regions\": [[2, 2]] }"
    }
  },
  "basketball_shot_002": {
    "0": {
      "text": "{ \"key_regions\": [[1, 3], [2, 3]] }"
    }
  }
}
```

## Usage in Training

The annotations are loaded during training to:
1. Build ground truth spatial attention masks
2. Compute KL divergence loss between predicted and ground truth attention
3. Guide the model to focus on task-relevant regions

See `atten.py` for implementation details of how these annotations are processed.

## Generating Annotations

You can generate annotations in two ways:

### 1. Manual Annotation
Create a custom annotation tool that:
- Displays video frames with a 4×4 grid overlay
- Allows users to click grid cells to mark relevant regions
- Exports to the JSON format above

### 2. LLM-based Annotation
Use the `prompt.py` template to:
- Send frames to a vision-language model
- Ask it to identify relevant grid cells
- Parse the response into the JSON format

Example with OpenAI GPT-4V:
```python
from prompt import generate_annotation_prompt, generate_system_message

# For each frame
prompt = generate_annotation_prompt(action_label="Basketball")
system_msg = generate_system_message()

# Send to LLM with frame image
response = llm.chat(
    messages=[
        {"role": "system", "content": system_msg},
        {"role": "user", "content": [
            {"type": "text", "text": prompt},
            {"type": "image", "image": frame_image}
        ]}
    ]
)

# Parse response and save to JSON
```

## Notes

- Not all frames need annotations - the model handles missing annotations gracefully
- You can annotate a subset of videos for semi-supervised training
- More annotations generally lead to better spatial attention alignment
- Annotations should focus on regions directly relevant to the action
