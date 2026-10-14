# TODO: LLM Component Optimizations

This document tracks planned improvements to the LLM-based XAI summarization component ([src/lcn/llm_summarizer.py](src/lcn/llm_summarizer.py)).

### 1. Improve Prompt Templates for Consistency

**Problem**: Current prompt template produces inconsistent outputs. LLM sometimes deviates from the required 4-section format or word count constraints.

### 2. Add Judge LLM for Quality Control
**Problem**: Single-pass generation lacks validation and quality assurance. No mechanism to catch hallucinations or format violations.

### 3. Support Multiple API Backends (Dynamic Model Routing)
**Problem**: Current implementation is hardcoded to OpenAI-compatible API (LivAPI). Not transferable to other deployment environments or model providers.

