# Build notes (delete before submitting if you like)

This project was generated in a sandbox with **no network and only numpy installed**.

Verified there:  `dataset.py` (all thresholds met), the 12 KB docs, both chunkers, PII/injection
guardrails, escalation score, budget + cache logic, least-autonomy allow-list, `MockLLM` ReAct
logic against a stubbed `BaseLLM` (incl. both brief pitfalls), 8 unit tests.

NOT executed there (needs `pip install -r requirements.txt` on your machine):
crewai Agent/Task/Crew wiring, ChromaDB + SentenceTransformers, LangChain memory, Autogen team,
FastAPI/TestClient. These were written against the documented APIs but have never run.
Expect to fix small API-version mismatches. Likeliest spots:
  1. `mock_llm.py`   - CrewAI's tool-description format / `BaseLLM.__init__` signature.
  2. `crew.py`       - `output_pydantic` conversion with the mock; `Task(context=...)`.
  3. `review.py`     - `ChatCompletionClient` abstract methods / `model_info` keys.
  4. `memory.py`     - RunnableWithMessageHistory config keys.
Run `python scripts/05_crew_demo.py` first - it exercises the riskiest code.

All `TODO` numbers in README.md must come from YOUR run of `./run_all.sh`.
