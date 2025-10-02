# Two-Stream Video Transformer for Action Recognition
 PyTorch implementation of reasoning supervison framework for ViTs in HAR tasks

## Overview

This repository implements a two-stream video transformer architecture that combines:
- **Spatial attention mechanism** for identifying task-relevant regions in video frames
- **Multi-scale temporal reasoning** across three temporal granularities (immediate, contextual, narrative)
- **Attention-guided supervision** using KL divergence for spatial alignment
- **Temporal consistency loss** using Jensen-Shannon divergence across temporal views

## Architecture

The model consists of two main stages:

### Stage 1: Spatial Transformer
- Processes pre-extracted visual features (e.g., from ResNet-50)
- Uses multi-head self-attention to capture spatial relationships
- Outputs frame-level representations

### Stage 2: Multi-Scale Temporal Reasoning
- **Immediate view**: Captures fine-grained, short-term motion cues
- **Contextual view**: Encodes mid-range temporal dependencies
- **Narrative view**: Represents long-term action semantics
- Cross-temporal fusion for final prediction

## Installation

### Requirements
```bash
Python >= 3.8
PyTorch >= 1.10
torchvision
numpy
scikit-learn
matplotlib
seaborn
tqdm
opencv-python
Pillow
```

### Setup
```bash
# Clone the repository
git clone <repository-url>
cd <repository-name>

# Install dependencies
pip install -r requirements.txt
```

## Data Preparation

### Dataset Structure
Organize your dataset in the following structure:
```
dataset/
├── class_1/
│   ├── video_1.avi
│   ├── video_2.avi
│   └── ...
├── class_2/
│   ├── video_1.avi
│   └── ...
└── ...
```

### Pre-extract Features (Optional but Recommended)
For faster training, pre-extract visual features using a pretrained CNN (e.g., ResNet-50):

```python
import torch
import torchvision.models as models
from torchvision import transforms

# Load pretrained ResNet-50
resnet = models.resnet50(pretrained=True).cuda()
resnet.eval()

# Remove final classification layer
feature_extractor = torch.nn.Sequential(*list(resnet.children())[:-2])

# Extract features from video frames
# Input: [B, 3, 224, 224] RGB frames
# Output: [B, 2048, 7, 7] features
```

Expected feature format: `[B, T, 2048, 4, 4]` where:
- `B`: Batch size
- `T`: Number of temporal frames
- `2048`: Feature dimension (ResNet-50)
- `4x4`: Spatial resolution (avg pooled from 7x7)

### Generate Annotations (Optional)
For spatial attention supervision, prepare annotations in JSON format:

```json
{
  "video_id_1": {
    "0": {
      "localization": [[row1, col1], [row2, col2], ...]
    },
    "1": {
      "localization": [[row1, col1], ...]
    },
    ...
  },
  "video_id_2": { ... }
}
```

Where `row` and `col` are grid coordinates in the 4x4 spatial grid.

Use the provided data preprocessing script:
```bash
python get_data.py \
  --dataset_dir /path/to/videos \
  --output_dir processed_data \
  --anno_file annotations.json
```

## Training

### Basic Training
```bash
python train.py \
  --data_dir /path/to/dataset \
  --batch_size 4 \
  --epochs 15 \
  --lr 1e-4 \
  --num_classes 101 \
  --T 64 \
  --out_dir checkpoints
```

### Training with Spatial Supervision
```bash
python train.py \
  --data_dir /path/to/dataset \
  --anno_path /path/to/annotations.json \
  --batch_size 4 \
  --epochs 15 \
  --lambda_spatial 0.15 \
  --lambda_js 0.25 \
  --out_dir checkpoints
```

### Key Hyperparameters
- `--lambda_spatial`: Weight for spatial attention loss (default: 0.15)
- `--lambda_js`: Weight for JS divergence loss (default: 0.25)
- `--T`: Number of temporal frames per clip (default: 64)
- `--batch_size`: Batch size (reduce if OOM)
- `--epochs`: Number of training epochs

### Memory Optimization
If you encounter OOM errors:
1. Reduce batch size: `--batch_size 2`
2. Reduce temporal length: `--T 32`
3. Use gradient accumulation (modify `accumulation_steps` in `train.py`)
4. Use smaller model: reduce `embed_dim`, `spatial_layers`, or `temporal_layers`

## Evaluation

```bash
python eva.py \
  --model_path checkpoints/model.pth \
  --data_dir /path/to/test/dataset \
  --batch_size 16 \
  --num_classes 101 \
  --T 64 \
  --out_dir evaluation_results
```

The evaluation script outputs:
- **Multi-view accuracies**: Immediate, Contextual, Narrative, and Ensemble
- **Confusion matrix**: Normalized confusion matrix visualization
- **Per-class metrics**: F1-scores, precision, recall
- **Performance plots**: Accuracy comparison across temporal views

## Model Architecture Details

### Spatial Transformer
- **Input**: `[B, T, C, H, W]` pre-extracted features
- **Patch embedding**: Linear projection to `embed_dim`
- **Positional encoding**: 2D sinusoidal embeddings
- **Transformer layers**: `spatial_layers` blocks with `spatial_heads` attention heads
- **Output**: Frame-level features `[B, T, embed_dim]`

### Multi-Scale Temporal Stage
- **Three temporal transformers**:
  - Immediate: Last 20% of frames
  - Contextual: Middle 60% of frames
  - Narrative: Full clip (100%)
- **Cross-temporal reasoning**: Fusion across temporal views
- **Classification heads**: Separate classifiers for each temporal view

### Loss Functions

#### 1. Classification Loss
Standard cross-entropy for each temporal view:
```
L_cls = CE(y_immediate, y_true) + CE(y_contextual, y_true) + CE(y_narrative, y_true)
```

#### 2. Spatial Alignment Loss (KL Divergence)
```
L_spatial = KL(q || p)
```
Where:
- `q`: Ground truth spatial distribution from annotations
- `p`: Predicted attention distribution from spatial transformer

#### 3. Temporal Consistency Loss (JS Divergence)
```
L_temporal = (1/3) * [KL(p_im || p_bar) + KL(p_ctx || p_bar) + KL(p_nar || p_bar)]
p_bar = (p_im + p_ctx + p_nar) / 3
```

#### Total Loss
```
L_total = L_cls + λ_spatial * L_spatial + λ_js * L_temporal
```

## File Structure

```
.
├── README.md                   # This file
├── ANNOTATIONS.md              # Annotation format documentation
├── model.py                    # Model architecture
├── atten.py                    # Attention loss functions
├── train.py                    # Training script
├── eva.py                      # Evaluation script
├── get_data.py                 # Data preprocessing utilities
├── prompt.py                   # Prompt templates for annotation generation
├── sample_annotations.json     # Example annotation file
├── requirements.txt            # Python dependencies
└── .gitignore                  # Git ignore rules
```

## Implementation Notes

### DataLoader Requirements
You need to implement your own `DataLoader` that returns:
- `clips`: `[B, T, C, H, W]` tensor of pre-extracted features
- `labels`: `[B]` tensor of class labels
- `metas`: List of dictionaries with metadata:
  ```python
  {
    'video_id': str,
    'annotation_path': str,  # Path to annotation JSON (optional)
    'start': int,            # Starting frame index (optional)
    ...
  }
  ```

### Annotation Format
For spatial attention supervision, annotations should specify:
- Frame-level spatial coordinates in a 4x4 grid
- Coordinates as `(row, col)` tuples, where `0 <= row, col < 4`

## Citation

If you use this code, please cite:

```bibtex
@inproceedings{anonymous2025video,
  title={Multi-Scale Temporal Reasoning for Video Action Recognition},
  author={Anonymous},
  booktitle={Conference},
  year={2025}
}
```

## License

This project is released under the MIT License.

## Acknowledgements

This implementation builds upon:
- [Timesformer](https://github.com/facebookresearch/TimeSformer) for temporal modeling
- [ViT](https://github.com/google-research/vision_transformer) for spatial attention
- PyTorch and Hugging Face Transformers

## Contact

For questions or issues, please open a GitHub issue.
