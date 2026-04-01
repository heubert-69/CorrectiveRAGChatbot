🤖 CRAG Chatbot – Transformer-Based Conversational AI
--
A Streamlit-powered chatbot built using Hugging Face Transformers with a Corrective Retrieval-Augmented Generation (CRAG) pipeline to reduce hallucinations and improve response reliability.
--
Overview:

This project implements a conversational AI system using a pre-trained transformer model enhanced with a self-correcting retrieval mechanism.

Instead of relying purely on generation, the chatbot:

- Generates an initial response
- Evaluates its quality using heuristic checks
- Retrieves external knowledge if needed
- Regenerates a more grounded response
- Key Features
- Context-aware conversation (chat memory)
- Continuous multi-turn interaction
- Controlled text generation (temperature, top-k, top-p)
- Corrective RAG (CRAG) pipeline
- External knowledge retrieval (Wikipedia)
- Input validation & response filtering
- Chat reset functionality
- Reduced hallucination vs standard chatbots
- Interactive UI using Streamlit
--
Architecture:
User Input
   ↓
Transformer Model (DialoGPT)
   ↓
Response Evaluation (Heuristics)
   ↓
[Low Confidence?]
   → YES → Retrieve Knowledge (Wikipedia)
           → Augmented Prompt
           → Regenerate Response
   → NO  → Return Response
--
Tech Stack:
- Python
- Hugging Face Transformers
- PyTorch
- Streamlit
- Wikipedia API

--
Installation:
```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name

pip install -r requirements.txt
```

Or install manually:
```bash
pip install streamlit transformers torch wikipedia
```

--
Running the App:
```bash
streamlit run app.py
```
Then open the local URL provided by Streamlit (usually http://localhost:8501).
--

Example Interaction:
```bash
User: What is Artificial Intelligence?
Bot: Artificial Intelligence refers to systems that simulate human intelligence such as learning and reasoning.

User: Who created Python?
Bot: Python was created by Guido van Rossum and released in 1991.
```

If the model is uncertain:
```bash
Bot: Let me verify that...
Bot: [Improved grounded response]
```
--
CRAG Mechanism (Core Idea):

This project uses a simplified Corrective RAG approach:

- Initial response is generated
- A confidence heuristic evaluates quality
If low confidence:
  - External knowledge is retrieved
  - Prompt is augmented with factual context
  - Response is regenerated

This helps reduce:

- hallucinations
- vague answers
- repetitive outputs

--

Limitations:
- Heuristic-based confidence (not probabilistic)
- Wikipedia retrieval may fail on ambiguous queries
- No vector database or semantic search
- Limited long-term memory
- Still susceptible to hallucinations in edge cases
--
Future Improvements:
- FAISS / vector database for semantic retrieval
- Embedding-based similarity search
- Log-probability confidence scoring
- Multi-hop retrieval
- Streaming responses (real-time token output)
- Deployment (Docker / cloud)
--
Project Structure:
```bash
├── app.py
├── README.md
├── requirements.txt
└── RobustChatbot.ipynb #Notebook Submission
```
--
License

This project is for educational purposes as part of an NLP assignment.
--
