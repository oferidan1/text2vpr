#!/usr/bin/env python3
"""
Quick test script to demonstrate the LLM filtering functionality.
This simulates the filtering without requiring a full CSV run.

This test demonstrates the smart safeguards against LLM "chattiness":
- Dynamic max_new_tokens (calculated based on input length)
- END_OF_LIST marker (### END_OF_LIST ###) for clean truncation
- Temperature=0.0 for deterministic output
- Improved prompt that explicitly requests no explanations
- Minimal post-processing (marker does most of the work)
"""

from llm_client import LLMClient, LLMConfig
from main import filter_objects_with_llm, DEFAULT_FILTER_PROMPT_TEMPLATE

def test_filtering():
    """Test the LLM filter with sample object lists."""
    
    # Initialize the LLM client with Qwen2.5-3B using optimized settings for filtering
    print("Initializing Qwen2.5-3B-Instruct for filtering...")
    print("Using smart safeguards:")
    print("  • Dynamic max_new_tokens (calculated per input list length)")
    print("  • END_OF_LIST marker (### END_OF_LIST ###)")
    print("  • temperature=0.0 (deterministic)")
    print("  • Improved prompt (no explanations)")
    print("  • Minimal post-processing (marker does the work)")
    
    config = LLMConfig(
        model_name="Qwen/Qwen2.5-3B-Instruct",
        max_new_tokens=128,  # Default, will be overridden dynamically per call
        temperature=0.0,     # Deterministic
    )
    llm_client = LLMClient(config)
    print("✓ Model loaded\n")
    
    # Test cases with problematic noun phrase parser output
    # These test both objects (things) and stuff (regions) suitable for SAM segmentation
    test_cases = [
        {
            "name": "Abstract concepts mixed with objects and stuff",
            "input": "street. building. beauty. happiness. sky. emotion. pavement",
            "expected_removed": ["beauty", "happiness", "emotion"]
        },
        {
            "name": "Generic terms mixed with segmentable items",
            "input": "car. thing. stuff. person. something. road. grass",
            "expected_removed": ["thing", "stuff", "something"]
        },
        {
            "name": "Quality adjectives as nouns (not segmentable)",
            "input": "tree. greenness. brightness. flower. sky. darkness",
            "expected_removed": ["greenness", "brightness", "darkness"]
        },
        {
            "name": "All valid items - objects and regions (nothing to filter)",
            "input": "car. building. tree. person. sky. road. grass. water",
            "expected_removed": []
        },
        {
            "name": "Complex compound nouns and abstract concepts",
            "input": "traffic-light. sign-board. window. importance. value. wall",
            "expected_removed": ["importance", "value"]
        },
    ]
    
    print("=" * 70)
    print("Testing LLM Filtering")
    print("=" * 70)
    
    for i, test in enumerate(test_cases, 1):
        print(f"\nTest {i}: {test['name']}")
        print(f"Input:  {test['input']}")
        
        # Show dynamic token calculation
        input_length = len(test['input'])
        estimated_tokens = max(32, min(256, input_length + 20))
        print(f"Tokens: {estimated_tokens} (calculated from input length: {input_length})")
        
        # Apply filtering
        filtered = filter_objects_with_llm(
            test['input'],
            llm_client,
            DEFAULT_FILTER_PROMPT_TEMPLATE
        )
        
        print(f"Output: {filtered}")
        
        # Check if expected items were removed
        input_items = set(item.strip() for item in test['input'].split('.') if item.strip())
        output_items = set(item.strip() for item in filtered.split('.') if item.strip())
        removed_items = input_items - output_items
        
        print(f"Removed: {', '.join(removed_items) if removed_items else 'none'}")
        
        # Verify expected removals
        expected_removed_set = set(test['expected_removed'])
        if expected_removed_set.issubset(removed_items):
            print("✓ Filter working as expected")
        else:
            print(f"⚠ Expected to remove: {', '.join(expected_removed_set)}")
        
        print("-" * 70)
    
    print("\n" + "=" * 70)
    print("Test complete!")
    print("=" * 70)
    
    print("\n📋 Prompt Template Used:")
    print("-" * 70)
    print(DEFAULT_FILTER_PROMPT_TEMPLATE)
    print("-" * 70)
    
    print("\n✅ Smart Safeguards in Place:")
    print("  1. Prompt explicitly requests: 'NO explanations, notes, or reasoning'")
    print("  2. Prompt ends with 'Filtered list:' to direct output immediately")
    print("  3. END_OF_LIST marker: Model outputs '### END_OF_LIST ###' at the end")
    print("  4. LLMClient truncates at any prefix of the marker automatically")
    print("  5. Dynamic max_new_tokens: calculated based on input length (32-256 tokens)")
    print("  6. temperature=0.0 (deterministic, no creative additions)")
    print("  7. Minimal post-processing (marker handles most cleanup)")
    
    print("\n🧠 Dynamic Token Calculation:")
    print("  • Short input (20 chars)  → ~40 tokens")
    print("  • Medium input (100 chars) → ~120 tokens")
    print("  • Long input (200 chars)  → ~220 tokens")
    print("  • Maximum: 256 tokens (prevents excessive generation)")
    print("  • Minimum: 32 tokens (enough for small lists)")
    
    print("\n🚀 To use in your pipeline, run:")
    print("  python3 main.py --use_noun_phrase_parser --filter_with_llm [other args...]")
    
    print("\n💡 The combination of dynamic tokens + END_OF_LIST marker + prompt engineering")
    print("   ensures Qwen2.5 outputs ONLY the filtered list with maximum efficiency!")

if __name__ == "__main__":
    test_filtering()

