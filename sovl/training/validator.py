from typing import Dict, Any, List, Optional
import torch
from torch.utils.data import DataLoader
from transformers import PreTrainedModel, PreTrainedTokenizer
import numpy as np

def validate_epoch(
    model: PreTrainedModel,
    valid_dataloader: DataLoader,
    scaffold_context: Optional[torch.Tensor] = None
) -> float:
    """
    Validate the model on a validation dataset.
    
    Args:
        model: Model to validate
        valid_dataloader: Validation data loader
        scaffold_context: Context from scaffold model (optional)
        
    Returns:
        Average validation loss
    """
    model.eval()
    total_loss = 0.0
    
    with torch.no_grad():
        for batch in valid_dataloader:
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
                scaffold_context=scaffold_context
            )
            total_loss += outputs.loss.item()
    
    return total_loss / len(valid_dataloader)

def calculate_confidence_score(
    logits: torch.Tensor,
    generated_ids: torch.Tensor
) -> float:
    """
    Calculate confidence score for generated text.
    
    Args:
        logits: Model logits
        generated_ids: Generated token IDs
        
    Returns:
        Confidence score between 0 and 1
    """
    # Get probabilities for generated tokens
    probs = torch.softmax(logits, dim=-1)
    token_probs = torch.gather(probs, -1, generated_ids.unsqueeze(-1)).squeeze(-1)
    
    # Calculate average log probability
    avg_log_prob = torch.mean(torch.log(token_probs + 1e-10))
    
    # Convert to confidence score (0 to 1)
    confidence = torch.sigmoid(avg_log_prob).item()
    
    return confidence

def evaluate_generation(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    prompts: List[str],
    max_length: int = 50,
    num_return_sequences: int = 1,
    temperature: float = 1.0,
    top_k: int = 50,
    top_p: float = 0.95,
    scaffold_context: Optional[torch.Tensor] = None
) -> Dict[str, Any]:
    """
    Evaluate model generation quality.
    
    Args:
        model: Model to evaluate
        tokenizer: Tokenizer for the model
        prompts: List of prompts to generate from
        max_length: Maximum generation length
        num_return_sequences: Number of sequences to generate per prompt
        temperature: Sampling temperature
        top_k: Top-k sampling parameter
        top_p: Top-p sampling parameter
        scaffold_context: Context from scaffold model (optional)
        
    Returns:
        Dictionary containing evaluation metrics
    """
    model.eval()
    results = {
        "generations": [],
        "confidence_scores": [],
        "diversity_scores": []
    }
    
    with torch.no_grad():
        for prompt in prompts:
            # Tokenize input
            inputs = tokenizer(prompt, return_tensors="pt")
            input_ids = inputs["input_ids"].to(model.device)
            
            # Generate
            outputs = model.generate(
                input_ids,
                max_length=max_length,
                num_return_sequences=num_return_sequences,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                do_sample=True,
                scaffold_context=scaffold_context
            )
            
            # Decode and calculate metrics
            generations = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            results["generations"].extend(generations)
            
            # Calculate confidence scores
            for gen_ids in outputs:
                logits = model(
                    input_ids=input_ids,
                    attention_mask=inputs["attention_mask"].to(model.device),
                    scaffold_context=scaffold_context
                ).logits
                confidence = calculate_confidence_score(logits, gen_ids)
                results["confidence_scores"].append(confidence)
            
            # Calculate diversity score (if multiple sequences)
            if num_return_sequences > 1:
                diversity = calculate_diversity_score(generations)
                results["diversity_scores"].append(diversity)
    
    # Calculate average metrics
    results["avg_confidence"] = np.mean(results["confidence_scores"])
    if results["diversity_scores"]:
        results["avg_diversity"] = np.mean(results["diversity_scores"])
    
    return results

def calculate_diversity_score(generations: List[str]) -> float:
    """
    Calculate diversity score between generated sequences.
    
    Args:
        generations: List of generated sequences
        
    Returns:
        Diversity score between 0 and 1
    """
    if len(generations) < 2:
        return 0.0
    
    # Calculate pairwise edit distances
    distances = []
    for i in range(len(generations)):
        for j in range(i + 1, len(generations)):
            # Simple character-level edit distance
            dist = levenshtein_distance(generations[i], generations[j])
            max_len = max(len(generations[i]), len(generations[j]))
            normalized_dist = dist / max_len
            distances.append(normalized_dist)
    
    return np.mean(distances) if distances else 0.0

def levenshtein_distance(s1: str, s2: str) -> int:
    """
    Calculate Levenshtein distance between two strings.
    
    Args:
        s1: First string
        s2: Second string
        
    Returns:
        Levenshtein distance
    """
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    
    if len(s2) == 0:
        return len(s1)
    
    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    
    return previous_row[-1] 