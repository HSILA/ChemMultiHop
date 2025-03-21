# ChemMultiHop
ChemMultiHop is a repository that accompanies our study on multi-hop reasoning in the chemistry domain. It includes the multi-hop question and answer datasets, evaluation code, and the model responses that were used to populate the figures and tables in the paper.

## Repository Contents

- **Q&A Data:** A collection of multi-hop chemistry questions and answers with the context required to answer them.
- **Evaluation Code:** Script used to assess model performance and reproduce evaluation results presented in the paper.
- **Model Responses:** The outputs from various models evaluated under different conditions (contextual and non-contextual) as reported in the study.

## How to Use

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

The code takes `records.json`, which contains a collection of questions, answers, context, and lengths (hops), evaluates all the models on it with and without context, saves the model-specific results in the `responses` directory, and creates a `results.csv` file with the overall performance and usage details of the models.