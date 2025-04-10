# SOVL System (Self-Organizing Virtual Lifeform)

## Overview
An AI agent with autonomous learning capabilities, combining a base LLM with a scaffolded second dynamic LLM for continuous learning via sleep mechanism

## Key Components of Prototype Implementation

### Core Models

- **Base Model:** Large frozen LLM (`deepseek-llm-67b-chat`)

- **Scaffold Model:** Smaller adaptable model (`deepseek-r1-distill-qwen1.5-1.5b`) with dynamic LoRA layers


## Innovative Modules

**AdaptiveLoRALinear:** Dynamically adjusts LoRA rank based on learned importance

**SparseCrossAttention:** Efficient top-k attention between models

**CrossAttentionFuser:** Intelligently combines base and scaffold outputs

**SalienceScorer:** Evaluates interaction importance using BERT

## Key Features

### Dynamic Adaptation:

- LoRA layers automatically adjust their rank

- Cross-attention gates scaffold contributions based on confidence

### Continuous Learning:

- Background training scheduler

- Cluster-based data sampling

- Automatic rollback on failure

### Resource Awareness:

- System load monitoring

- Training time limits

- Gradient checkpointing

## Configuration

The system has some configurable parameters including:

- Training intervals (default: 5 minutes)

- Minimum training examples (default: 50)

- System load limits (default: 70%)

- Salience thresholds (default: 0.75)

- Training epochs (default: 3)

- More planned (see TODO.md)

```
system = ASCSystem()
response = system.generate_response(user_input)
system.log_interaction(user_input, response)
```

## Requirements

- PyTorch

- Transformers

- PEFT (Parameter-Efficient Fine-Tuning)

- Scikit-learn (for clustering)

- PSutil (for system monitoring)

## Project Refactoring and Modularization

### Implementation Plan

The project has been refactored into a modular structure with the following components:

#### Core System Modules
- `system_config`: Configuration management
- `system_logging`: Thread-safe logging

#### Data Handling Modules
- `data_loader`: Data loading and validation
- `data_processing`: Tokenization and sequence mapping

#### Model Management Modules
- `model_loader`: Base and scaffold model loading
- `quantization`: Model quantization support
- `cross_attention`: Cross-attention mechanisms

#### Training and Optimization Modules
- `training`: Training steps and cycles
- `validation`: Model validation
- `gestation`: Gestation and sleep training

#### Memory Management Modules
- `dream`: Dreaming mechanism
- `memory_manager`: Memory decay and pruning

#### Curiosity and Feedback Modules
- `curiosity`: Novelty detection and exploration
- `feedback`: Feedback and temperament management

### Changes Made

1. **Directory Structure**
   - Created modular directory structure under `sovl/`
   - Each module has its own directory with `__init__.py`
   - Clear separation of concerns between modules

2. **Memory Management**
   - Implemented `MemoryManager` class with:
     - Token map management
     - Scaffold memory handling
     - Memory decay and pruning
     - Statistics tracking

3. **Curiosity Module**
   - Implemented `CuriosityModule` with:
     - History management using deque
     - Novelty detection via cosine similarity
     - Exploration control with configurable thresholds
     - Feedback integration
     - Statistics tracking

4. **File Cleanup**
   - Removed redundant files:
     - `sovl_main.py` (replaced by modular structure)
     - `old_prototype.py` (outdated version)
   - Preserved important files:
     - Training data and logs
     - Documentation files
     - Configuration files
     - Web interface

### Benefits of Modularization

1. **Improved Maintainability**
   - Each module has a single responsibility
   - Clear interfaces between components
   - Easier to test and debug

2. **Better Scalability**
   - Modules can be developed independently
   - Easier to add new features
   - Better resource management

3. **Enhanced Flexibility**
   - Components can be swapped or modified
   - Configuration is more granular
   - Easier to adapt to different use cases

4. **Improved Code Quality**
   - Better organization
   - Clearer dependencies
   - More consistent coding style
