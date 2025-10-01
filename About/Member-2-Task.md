# My Tasks: LLM & Detection Engineer (Member 2)

This document outlines my specific responsibilities, tasks, and integration points for the **AI Hallucination Detection System project**. My primary goal is to serve as the bridge between the core language model (Gemini Pro) and the system's fact-checking logic. I will build the module that generates the initial answer and then decides if that answer is factually sound based on retrieved evidence.

**Version 2 Update:** This plan has been updated to streamline collaboration with Member 3 (Correction Module) by passing the calculated confidence score downstream.

---

## Phase 1: Setup and Initial Connection (Weeks 1–2)
*(No changes in this phase)*

### Task 1.1: Environment and Library Setup
- **Action:** Set up a Python virtual environment and install `google-generativeai`, `langchain`, `sentence-transformers`, and `transformers`.

### Task 1.2: Authenticate and Test Gemini Pro
- **Action:** Write a test script to authenticate with the Gemini API using an environment variable for the API key and confirm a successful response.

### Task 1.3: Integrate Gemini with LangChain
- **Action:** Refactor the test script to use the `ChatGoogleGenerativeAI` class and a simple `LLMChain` to create a modular and integrable component.

---

## Phase 2: Core Detection Logic Implementation (Weeks 3–4)
*This phase focuses on building the multi-step detection logic. The output is now more comprehensive to support the downstream correction and logging modules.*

### Task 2.1: Implement Semantic Similarity Check
- **Action:**
    - Define a function `check_similarity(raw_answer: str, evidence_docs: list[str]) -> float`.
    - Use the `all-MiniLM-L6-v2` model to generate embeddings.
    - Calculate and return the highest cosine similarity score found between the answer and any evidence document.

### Task 2.2: Implement Natural Language Inference (NLI) for High-Confidence Detection
- **Action:**
    - Define a function `check_contradiction(raw_answer: str, evidence_docs: list[str]) -> bool`.
    - Load the `roberta-large-mnli` model using the `transformers` pipeline.
    - Return `True` if the model classifies the relationship as `contradiction` for any evidence document.

### Task 2.3: Combine Detection Methods into Final Logic
- **Description:** I will create a single, unified detection function that orchestrates both checks and produces a final, detailed verdict for the main pipeline.
- **Action:**
    - Define a main function `detect_hallucination(raw_answer, evidence_docs) -> tuple[bool, float]`.
    - Inside, first call `check_contradiction`.
    - Then, call `check_similarity` to get the similarity score.
    - If a contradiction was found OR if the similarity score is below a tunable threshold (e.g., 0.7), the function will return `(True, similarity_score)`.
    - Otherwise, it will return `(False, similarity_score)`.
- **Rationale:** This two-step process is robust. The new return type `(is_hallucination, confidence_score)` is more efficient, as the score calculated here can be used directly as the "confidence score" that Member 3 needs to log.

---

## Phase 3: System Integration (Weeks 4-5)
*This phase focuses on connecting my module with the rest of the pipeline, now with a clearer data contract.*

### Task 3.1: Integrate with Retrieval Module (Member 1)
- **Action:** Collaborate with Member 1 to get the `evidence_docs` from their `retrieve_evidence` function and pass it, along with the `raw_answer`, to my detection module.

### Task 3.2: Provide Detailed Output to the Main Pipeline and Correction Module (Member 3)
- **Description:** The output of my module is the trigger for the correction pipeline and provides data for the logging step. I need to provide a clear, unambiguous signal and the associated data.
- **Action:**
    - My `detect_hallucination` function will now return a tuple containing the boolean verdict and the calculated similarity score.
    - Work with Member 3 and the main pipeline owner to ensure the logic is handled correctly:
    ```python
    # Main pipeline logic
    is_hallucination, score = my_module.detect_hallucination(raw_answer, evidence)
    final_confidence_score = score

    if is_hallucination:
        # Pass score to Member 3's module if they need it
        corrected_answer_data = member3_module.correct_answer(raw_answer, evidence)
        # The final log will use the initial score from my module
    else:
        # Use raw_answer

    # Log the final answer and its confidence score (final_confidence_score)
    ```
- **Rationale:** This updated integration plan prevents Member 3 from having to recalculate a score that I already have. It makes the system more efficient and clarifies the data being passed between our modules, reducing the chances of integration errors.
