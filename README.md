# Google Drive RAG Assistant

LangGraph-powered RAG system that loads your entire Google Drive, creates a vector database, and lets you chat with your documents using Claude 3.5 Sonnet. Automatically retrieves relevant context when needed.

---

## What This Does

Connects to Google Drive → Loads all docs/sheets/PDFs recursively → Embeds and stores in ChromaDB → Builds agentic retrieval system with LangGraph → Chat interface with markdown-formatted responses.

**Input:** Your Google Drive (with OAuth credentials)  
**Output:** Conversational AI that knows everything in your Drive

---

## Architecture

### LangGraph Agent Flow

```
START → query_or_respond → [Decision: Need retrieval?]
                          ↓ YES              ↓ NO
                      retrieve          Direct response
                          ↓
                      generate → END
```

**Components:**
1. **Document Loader** - Google Drive API integration
2. **Vector Store** - ChromaDB with OpenAI embeddings
3. **Retriever Tool** - Searches top-k relevant chunks
4. **LangGraph Agent** - Decides when to retrieve vs respond directly
5. **LLM** - Claude 3.5 Sonnet for generation

---

## Tech Stack

- **Python 3.10+** (Jupyter Notebook)
- **LangChain** - Document loading, text splitting, embeddings
- **LangGraph** - Agentic orchestration with tools
- **ChromaDB** - Local vector database
- **OpenAI API** - Text embeddings
- **Claude 3.5 Sonnet** - Response generation
- **Google Drive API** - Document access

---

## Setup

### 1. Install Dependencies

```bash
# Core dependencies
%pip install python-dotenv

# LangChain ecosystem
%pip install langchain
%pip install langchain-openai
%pip install langchain-chroma
%pip install langchain-google-community[drive]

# Vector store
%pip install chromadb

# LLM APIs
%pip install openai
%pip install anthropic

# LangGraph
%pip install langgraph
```

### 2. Google Drive API Setup

**Get OAuth Credentials:**

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create new project
3. Enable Google Drive API
4. Create OAuth 2.0 credentials
5. Download credentials as `credentials.json`
6. Place in project root

**Scopes needed:**
- `https://www.googleapis.com/auth/drive.readonly`

### 3. Configure API Keys

Create `.env` file:

```
OPENAI_API_KEY=your_openai_key
ANTHROPIC_API_KEY=your_anthropic_key
LANGCHAIN_API_KEY=your_langchain_key  # Optional - for tracing
```

### 4. Update Paths

⚠️ **IMPORTANT:** Change the hardcoded path in the code:

Find this line:
```python
persist_directory="C:...drive_vectors"
```

Change to your path:
```python
persist_directory="./drive_vectors"  # Or your preferred location
```

---

## Usage

### Step 1: Load Google Drive Documents

```python
loader = GoogleDriveLoader(
    folder_id="root",  # Or specific folder ID
    credentials_path="credentials.json",
    token_path="token.json",
    recursive=True,
    file_types=["pdf", "sheets", "documents"],
    scopes=['https://www.googleapis.com/auth/drive.readonly']
)

documents = loader.load()
print(f"Loaded {len(documents)} documents")
```

**First run:** Opens browser for OAuth authentication → Creates `token.json`

**Supported formats:**
- Google Docs
- Google Sheets
- PDFs

### Step 2: Create Vector Store

```python
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
split_docs = text_splitter.split_documents(documents)

embeddings = OpenAIEmbeddings()
vectorstore = Chroma.from_documents(
    documents=split_docs,
    embedding=embeddings,
    persist_directory="./drive_vectors"
)
```

**Output:** ChromaDB database saved to disk (reusable across sessions)

### Step 3: Run Chat Interface

```python
# Already configured in notebook - just run the cell
while True:
    user_input = input("User: ")
    if user_input.lower() in ["quit", "exit", "q"]:
        break
    stream_graph_updates(user_input)
```

**Agent decides automatically:**
- Simple questions → Direct response
- Complex questions → Retrieves context → Generates answer

---

## How It Works

### Document Loading Pipeline

1. **Connect to Google Drive** via OAuth
2. **Traverse folders recursively** from root (or specified folder)
3. **Load content** from docs, sheets, PDFs
4. **Add 2-second delay** between requests (avoid API quota limits)

### RAG Pipeline

1. **Text Splitting** - Chunks documents (1000 chars, 200 overlap)
2. **Embedding** - OpenAI text-embedding-ada-002
3. **Storage** - ChromaDB vector database
4. **Retrieval** - Top-12 similar chunks via cosine similarity

### LangGraph Agent Logic

**Node 1: query_or_respond**
- LLM (with tool access) decides if retrieval needed
- If needs context → Calls retriever tool
- If can answer directly → Responds

**Node 2: retrieve (ToolNode)**
- Automatically executes retriever tool
- Fetches top-k relevant document chunks
- Adds to context

**Node 3: generate**
- Takes user query + retrieved context
- Generates markdown-formatted response
- Returns final answer

### Markdown Formatting

System prompt enforces structured responses:
- **Bold** for key terms
- *Italics* for emphasis
- `Code blocks` for technical terms
- Bullet points for lists
- `##` Headers for sections
- `>` Quotes for important notes

---

## Configuration

### Retrieval Settings

```python
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 12}  # Number of chunks to retrieve
)
```

**Adjust k based on:**
- Document complexity (higher for technical docs)
- Response detail needed (higher for comprehensive answers)
- Token limits (lower to save tokens)

### LLM Model

Currently using: **Claude 3.5 Sonnet (claude-3-5-sonnet-20241022)**

Change model:
```python
llm = init_chat_model("claude-3-5-sonnet-latest", model_provider="anthropic")
```

**Other options:**
- `claude-3-opus-latest` - More capable, slower, expensive
- `claude-3-haiku-latest` - Faster, cheaper, less capable

### Google Drive Scope

**Current scope:** Read-only access to all files

**Restrict to specific folder:**
```python
loader = GoogleDriveLoader(
    folder_id="1abc123xyz",  # Specific folder ID
    ...
)
```

Get folder ID from Drive URL: `https://drive.google.com/drive/folders/[FOLDER_ID]`

---

## Token Usage & Cost

### Per Session

**Document Loading (One-time):**
- Embedding cost: ~$0.0001 per 1K tokens
- For 1000 documents (~500K words): ~$0.10

**Per Query:**
- Retrieval: Free (local vector search)
- Claude generation: ~$0.003 per query (with retrieved context)

**Estimated monthly cost for 100 queries/day:**
- ~$9/month (mostly Claude API calls)

---

## File Structure

```
.
├── googledriveassistant.ipynb  # Main notebook
├── credentials.json            # Google OAuth credentials (from GCP)
├── token.json                  # Generated after first auth
├── drive_vectors/              # ChromaDB database (auto-created)
│   └── chroma.sqlite3
├── .env                        # API keys
└── README.md
```

---

## Troubleshooting

### "Rate limit exceeded" from Google Drive API

**Solution:** Increase delay between requests:
```python
def load_sheet_with_delay(self, sheet_id):
    time.sleep(5)  # Increase from 2 to 5 seconds
    return original_load_sheet(self, sheet_id)
```

### "Invalid credentials" error

1. Delete `token.json`
2. Re-run loader (will re-authenticate)
3. Verify `credentials.json` is from correct GCP project

### "Documents not found" in vector store

**Vector store not persisting:**
```python
# After creating vectorstore, explicitly persist
vectorstore.persist()
```

**Loading existing vectorstore:**
```python
vectorstore = Chroma(
    persist_directory="./drive_vectors",
    embedding_function=OpenAIEmbeddings()
)
```

### Agent not retrieving when it should

**Lower retrieval threshold:**
- Agent might be overconfident
- Add explicit retrieval instruction in system prompt
- Reduce tool calling temperature (not available in current setup)

### Out of memory during document loading

**Load in batches:**
```python
# Instead of folder_id="root"
# Load specific folders one at a time
loader = GoogleDriveLoader(folder_id="folder1_id", ...)
docs1 = loader.load()

loader = GoogleDriveLoader(folder_id="folder2_id", ...)
docs2 = loader.load()

all_docs = docs1 + docs2
```

## Design Decisions

**Why LangGraph instead of simple RAG chain?**  
Agent decides when retrieval is needed vs direct response. Saves tokens and latency for simple queries.

**Why ChromaDB instead of Pinecone/Weaviate?**  
Local vector store - no additional API costs, works offline, faster for small-medium datasets.

**Why Claude 3.5 Sonnet instead of GPT-4?**  
Better instruction following, longer context window (200K), better markdown formatting, comparable quality.

**Why OpenAI embeddings instead of local models?**  
Higher quality retrieval, established reliability, worth the small cost (~$0.10 for 1000 docs).

**Why recursive Google Drive loading?**  
Get everything in one go - no manual folder navigation needed.

---

## Known Issues

1. **Google Sheets loading is slow** - 2-second delay per sheet to avoid quota limits - Free Tier :(
2. **Large PDFs may timeout** - Consider pre-processing large files
3. **Agent sometimes retrieves unnecessarily** - Over-cautious tool calling
4. **No file update detection** - Must reload entire Drive to get new docs

---

## Future Improvements

- Add file change detection (only reload modified docs)
- Implement document metadata filtering
- Add support for more file types (images via OCR, videos via transcription)
- Build Gradio web interface
- Add conversation memory (chat history)
- Implement hybrid search (keyword + semantic)

---

## Notes

- First load takes time (proportional to Drive size)
- Vector store persists - subsequent runs are instant
- Responses include source attribution in context
- All responses markdown-formatted for readability
- Works offline after initial document loading

---


**Questions?** Check LangGraph docs or adjust agent prompts in the notebook.
