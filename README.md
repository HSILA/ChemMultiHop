# ChemMultiHopBench
ChemMultiHopBench is a repository that accompanies our study on multi-hop reasoning in the chemistry domain. It includes the multi-hop question and answer datasets, evaluation code, and the model responses that were used to populate the figures and tables in the paper.

## Repository Contents

- **Graph**: The constructed graph from extracted entities and generated relations which is used for multi-hop question generation.
- **Q&A Data:** A collection of multi-hop chemistry questions and answers with the context required to answer them.
- **HotpotQA Data:** A chemistry-specific subset of the HotpotQA dataset consisting of questions, answers and context.
- **Evaluation Code:** Script used to assess model performance and reproduce evaluation results presented in the paper.
- **Model Responses:** The outputs from various models evaluated under different conditions (contextual and non-contextual) as reported in the study.
- **HotpotQA Responses:** The outputs from various models evaluated on the HotpotQA dataset under different conditions (contextual and non-contextual) as reported in the study.
- **Kept/Removed Data:**  
  - `kept_data.json`: All multi-hop items with at least one correct model response.  
  - `removed_data.json`: All multi-hop items where `number_of_corrects` is zero (i.e., no model got them right).  
  Both files include:
    - `q`: question text  
    - `a`: expected answer  
    - `path`: list of `{entity1, relation, entity2, text, meta1, q, a}` objects showing exactly which graph edges and evidence snippets were used to generate this question  
    - `number_of_corrects`: total times any model answered correctly (contextual + non-contextual)
- **Experts Feedback:** The feedback from experts on the quality of the questions, dividing the generated questions in 3 groups, Poor, Ok, and Good.
- **Appendix:** A comprehensive explanation of each part of the pipeline (Both Graph Generation and QA Generation), including algorithms, modules, and prompts, added in the appendix of the paper.

## How to Run The Evaluation

1. **Clone the Repository:**
   ```bash
   git clone <repo-url>
   cd ChemMultiHop
    ```

2. **Install Dependencies:**
    ```bash
    python -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
    ```

3. **Evaluate Models:**
    ```
    python evaluate.py
    ```

The code takes `ChemMultiHopBench.json`, which contains a collection of questions, answers, context, and lengths (hops), evaluates all the models on it with and without context, saves the model-specific results in the `responses` directory, and creates a `results.csv` file with the overall performance and usage details of the models.

Additionally, the repository consists of `HotpotQA-Chemistry.json`, which is a subset of the HotpotQA dataset that was used to evaluate the models on the chemistry-specific multi-hop reasoning task. The format of this file is compatible with the `evaluate.py` script. We sampled chemistry questions by starting from Wikipedia’s Chemistry category, recursively exploring its subcategories (up to three levels), and then filtering HotpotQA based on exact title matches. To maintain consistency with our evaluation scheme, we excluded distractors and included only supporting documents as context.

The question and answers are generated from the `Graph.json`, which contains the constructed graph with keys `nodes` and `edges` that represent the entities and relations, respectively.
