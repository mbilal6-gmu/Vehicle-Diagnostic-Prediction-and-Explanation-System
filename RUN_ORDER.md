# Setup & Run Order

## 1. Install dependencies
```
pip install -r requirements.txt
```

## 2. Configure environment
```
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY
```

## 3. Preprocess the Excel data
```
python src/preprocess.py
```
Outputs: `Data/processed/` (X_train, X_test, y_* CSVs + label encoders)

## 4. (Optional) Fetch NHTSA complaints
```
python src/fetch_nhtsa.py
```
Outputs: `Data/nhtsa/nhtsa_complaints.csv`

## 4b. (Optional) Extract LEMON repair manuals — run in WSL
Adds real repair procedures and wiring diagrams for 7,063 Toyota vehicles.
Requires the LEMON database at `X:\lemon-manuals\lemon`.

```bash
# In WSL terminal:
sudo apt-get install -y libmtbl-dev
pip install pymtbl
python src/extract_lemon.py --lemon-path "/mnt/x/lemon-manuals/lemon"
```
Outputs: `Data/lemon/toyota_pages.jsonl` + `Data/lemon/images/*.png`
Then run step 5 as normal — build_vectorstore.py picks up the JSONL automatically.

## 5. Build the vector store
```
python src/build_vectorstore.py
```
Outputs: `vectorstore/chroma_db/` (ChromaDB + embeddings)

## 6. Train the ML models
```
python src/train_model.py
```
Outputs: `models/xgb_failure_risk.pkl`, `models/xgb_check_engine.pkl`

## 7. Run the web app
```
streamlit run app/streamlit_app.py
```

## 8. Run the test harness
```
python tests/test_harness.py
# For faster run (skip LLM calls):
python tests/test_harness.py --skip-llm
```

## Offline / No OpenAI key?
Install Ollama and pull DeepSeek:
```
ollama pull deepseek-r1:7b
```
The app will automatically fall back to DeepSeek when no valid OpenAI key is set.
