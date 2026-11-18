# 🍳 Recipe Intelligence Chatbot

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red?logo=pytorch)](https://pytorch.org/)
[![LangChain](https://img.shields.io/badge/LangChain-Latest-green)](https://python.langchain.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**An intelligent conversational agent that understands recipes, matches ingredients, and provides evidence-based culinary recommendations**

[Features](#-features) • [Architecture](#-architecture) • [Installation](#-installation) • [Usage](#-usage) • [Documentation](#-documentation)

</div>

---

## 📖 Overview

The **Recipe Intelligence Chatbot** is an advanced conversational AI system that combines hybrid search, semantic understanding, and evidence-based reasoning to help users discover recipes, match ingredients, compute statistics, and suggest intelligent substitutions. Built on top of 125,000+ recipes from AllRecipes, Epicurious, and Food Network.

### 🎯 Key Capabilities

- **🔍 Semantic Recipe Search**: Natural language queries with HyDE (Hypothetical Document Embeddings) enhancement
- **📊 Recipe Analytics**: Statistical analysis including ingredient frequency, recipe counts, and rankings
- **🥘 Ingredient Matching**: Smart recipe suggestions based on available ingredients with match scoring
- **🔄 Evidence-Based Substitutions**: Data-driven ingredient replacement recommendations using enrichment analysis
- **💬 Conversational Context**: Stateful conversations with dietary restrictions and constraint tracking

---

## ✨ Features

### 1. **Hybrid Search Engine**
- **Dense Retrieval**: BGE embeddings with semantic similarity search
- **Sparse Retrieval**: BM25 keyword-based ranking
- **HyDE Enhancement**: LLM-generated hypothetical recipes improve search quality
- **Cross-Encoder Reranking**: Final refinement using BGE reranker
- **Ensemble Fusion**: Weighted combination of dense (60%) and sparse (40%) results

### 2. **Intelligent Ingredient Normalization**
- GPU-accelerated batch processing using Qwen2.5-1.5B-Instruct
- Automatic extraction of base ingredients from complex descriptions
- Caching mechanism for efficient repeated lookups

### 3. **Multi-Intent Classification**
- **SEARCH**: Recipe discovery with natural language understanding
- **STATS**: Analytical queries (counts, rankings, distributions)
- **MATCH**: Recipe recommendations from available ingredients
- **SUBSTITUTE**: Evidence-based ingredient replacement suggestions

### 4. **Conversation Management**
- Stateful context tracking across multiple turns
- Dynamic constraint extraction (dietary restrictions, step limits)
- Follow-up query support with context retention

---

## 🏗️ Architecture

### 📊 System Architecture Diagram

> **[🎯 View Interactive Version](.archify/architecture-recipe-chatbot-20261118-043523/system-architecture.html)** - Explore with live motion, dark/light themes, and detailed component cards

The Recipe Intelligence Chatbot follows a layered architecture with specialized processing engines, ML model components, and persistent data storage. The diagram below shows the complete system flow from user query to response.

<img src="docs/system-architecture.svg" alt="System Architecture" width="100%"/>

### Key Architecture Components

**Presentation Layer**
- **Chat Interface**: Jupyter Notebook / CLI with conversational context management
- **State Manager**: Tracks dietary restrictions, constraints, and query history

**Intent Routing**
- **Intent Classifier**: LLM-powered router directing queries to specialized engines
- **Groq LLM**: Fast classification for SEARCH, STATS, MATCH, or SUBSTITUTE intents

**Processing Engines**
- **Hybrid Search**: Dense + Sparse retrieval with HyDE enhancement and reranking
- **SQL Analytics**: Statistical queries on recipe database
- **Match Scorer**: Ingredient-based recipe matching with scoring
- **Substitution Engine**: Evidence-based ingredient replacements

**ML Model Layer**
- **BGE Embedder**: Dense semantic embeddings (384-dim vectors)
- **BM25 Index**: Sparse keyword-based retrieval
- **Reranker**: Cross-encoder for final result refinement
- **Normalizer**: Qwen2.5-1.5B for ingredient extraction

**Data Storage**
- **SQLite DB**: 125,164 recipes with normalized ingredients
- **Embedding Cache**: Pre-computed vectors for fast retrieval
- **Norm Cache**: GPU-generated ingredient normalizations

---

## 🚀 Installation

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (optional, for faster processing)
- 8GB+ RAM
- 10GB+ free disk space

### Step 1: Clone the Repository

```bash
git clone https://github.com/AtharvaMate/recipe-chatbot.git
cd recipe-chatbot
```

### Step 2: Create Virtual Environment

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Set Up Environment Variables

Create a `.env` file in the project root:

```env
# Required for LLM functionality
GROQ_API_KEY=your_groq_api_key_here
ANTHROPIC_API_KEY=your_anthropic_api_key_here  # Optional: for Claude fallback

# Optional: Local Omniroute API
OMNIROUTE_BASE_URL=http://localhost:20128/v1/chat/completions
```

### Step 5: Download Recipe Dataset

Place the following JSON files in `recipes_raw/` directory:
- `recipes_raw_nosource_ar.json` (AllRecipes)
- `recipes_raw_nosource_epi.json` (Epicurious)
- `recipes_raw_nosource_fn.json` (Food Network)

### Step 6: Initialize Database (First Run)

The database will be automatically created on first run. This may take 10-20 minutes for ingredient normalization.

---

## 💻 Usage

### Interactive Chat Mode

```python
from recipe_chatbot_final import chat

# Start interactive session
chat()
```

**Example Conversation:**

```
🧑 You: Find a quick chocolate dessert without eggs
🤖 Bot: Found 5 recipes. Showing top 5:

1. **Chocolate Ganache**
   Steps: 7
   Ingredients: semisweet chocolate, heavy cream, flavoring...

2. **Sorbetto di Cioccolato**
   Steps: 17
   Ingredients: bittersweet chocolate, water, sugar...
```

### Programmatic Usage

```python
from recipe_chatbot_final import (
    ConversationState,
    process_query,
    search_recipes,
    match_recipes,
    suggest_substitution
)

# Initialize conversation state
state = ConversationState()

# 1. Search for recipes
response = process_query("Find Italian pasta recipes", state)
print(response)

# 2. Get statistics
response = process_query("How many recipes contain garlic and onion?", state)
print(response)

# 3. Match ingredients
state.user_ingredients = ["eggs", "flour", "milk"]
response = process_query("What can I make?", state)
print(response)

# 4. Suggest substitutions
response = process_query("Replace eggs in this recipe", state)
print(response)
```

### Query Examples

#### Recipe Search
```
- "Find a vegan chocolate cake"
- "Quick breakfast recipes under 5 steps"
- "Italian pasta without dairy"
```

#### Analytics
```
- "How many recipes contain both garlic and onion?"
- "Show me the simplest recipes with chicken"
- "What percentage of recipes are vegetarian?"
```

#### Ingredient Matching
```
- "I have chicken, rice, and tomatoes"
- "What can I make with eggs and flour? No nuts, at most 6 steps"
```

#### Substitutions
```
- "Replace butter in this recipe"
- "Find an egg substitute for baking"
- "I'm allergic to dairy, what can I use instead?"
```

---

## 📊 Technical Details

### Models & Technologies

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **Ingredient Normalization** | Qwen2.5-1.5B-Instruct | Extract base ingredients |
| **Dense Embeddings** | BAAI/bge-small-en-v1.5 | Semantic search |
| **Sparse Retrieval** | BM25Okapi | Keyword matching |
| **Reranker** | BAAI/bge-reranker-base | Final ranking refinement |
| **Intent Classification** | Groq (GPT-OSS-20B) | Query routing |
| **Reasoning** | Claude Sonnet 4.6 / Omniroute | Substitution & HyDE |
| **Database** | SQLite | Recipe storage |

### Performance Metrics

- **Database**: 125,164 recipes indexed
- **Search Latency**: ~500ms (with reranking)
- **Ingredient Normalization**: ~200 ingredients/second (GPU)
- **Embedding Cache**: Persistent (embeddings.npy)
- **Memory Usage**: ~2GB (with models loaded)

### Database Schema

```sql
-- Recipes table
CREATE TABLE recipes (
    id TEXT PRIMARY KEY,
    source TEXT,
    title TEXT,
    ingredients_json TEXT,
    instructions TEXT,
    step_count INTEGER,
    ingredient_count INTEGER
);

-- Normalized ingredients
CREATE TABLE ingredients (
    recipe_id TEXT,
    ingredient_normalized TEXT,
    ingredient_original TEXT,
    FOREIGN KEY(recipe_id) REFERENCES recipes(id)
);

-- Indexes for fast lookups
CREATE INDEX idx_ing_norm ON ingredients(ingredient_normalized);
CREATE INDEX idx_recipe_id ON ingredients(recipe_id);
```

---

## 🧪 Evaluation Results

### Test Case 1: Search with Dietary Restrictions
**Query**: "Find a chocolate recipe" → "I cannot have sugar"

**Result**: ✅ Successfully filtered recipes and provided honey/maple syrup substitution with detailed adaptation instructions.

### Test Case 2: Statistical Query
**Query**: "How many recipes contain both garlic and onion?"

**Result**: ✅ Found 12,713 recipes (10.16% of dataset) with accurate percentage calculation.

### Test Case 3: Ingredient Matching
**Query**: "I have onions, tomatoes, and rice. No peanuts, at most 6 steps"

**Result**: ✅ Returned 5 recipes with match scores, correctly filtered by constraints.

### Test Case 4: Evidence-Based Substitution
**Query**: "Find banana bread" → "Replace butter in this recipe"

**Result**: ✅ Provided margarine as evidence-based substitute with enrichment score 7.52x, complete recipe adaptation.

### Test Case 5: No Match Scenario
**Query**: "Find chocolate cake" (with extreme constraints: no eggs/dairy/gluten/soy/nuts, max 1 step)

**Result**: ✅ Graceful handling with explanation of active filters and suggestion to relax constraints.

---

## 📁 Project Structure

```
recipe-chatbot/
├── recipe_chatbot_final.ipynb    # Main notebook with all functionality
├── README.md                      # This file
├── requirements.txt               # Python dependencies
├── .env                          # Environment variables (create this)
├── recipes.db                    # SQLite database (auto-generated)
├── embeddings.npy                # Cached embeddings (auto-generated)
├── recipes_raw/                  # Recipe JSON files
│   ├── recipes_raw_nosource_ar.json
│   ├── recipes_raw_nosource_epi.json
│   └── recipes_raw_nosource_fn.json
└── docs/                         # Documentation
    └── architecture_diagrams.md
```

---

## 🔧 Configuration

### Hybrid Search Weights

```python
search_engine = HybridRecipeSearchEngine(
    conn=conn,
    llm=groq_llm,
    dense_weight=0.6,  # Semantic similarity weight
    sparse_weight=0.4   # Keyword matching weight
)
```

### Conversation State Options

```python
state = ConversationState(
    exclude_ingredients=["dairy", "nuts"],  # Dietary restrictions
    max_steps=10,                           # Complexity limit
    user_ingredients=["chicken", "rice"]    # Available ingredients
)
```

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **Dataset**: Recipe data from AllRecipes, Epicurious, and Food Network
- **Models**: 
  - [BGE Embeddings](https://huggingface.co/BAAI/bge-small-en-v1.5) by Beijing Academy of AI
  - [Qwen2.5](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct) by Alibaba Cloud
  - [Claude](https://www.anthropic.com/claude) by Anthropic
- **Frameworks**: LangChain, PyTorch, Sentence Transformers

---

## 📞 Contact

**Project Maintainer**: Atharva Mate

- GitHub: [@AtharvaMate](https://github.com/AtharvaMate)
- Project: [Recipe Intelligence Chatbot](https://github.com/AtharvaMate/recipe-chatbot)

---

## 🗺️ Roadmap

- [ ] **v2.0**: Multi-modal support (recipe images)
- [ ] **v2.1**: Nutritional information integration
- [ ] **v2.2**: Meal planning and shopping list generation
- [ ] **v2.3**: Voice interface support
- [ ] **v2.4**: Multi-language support
- [ ] **v2.5**: User preference learning

