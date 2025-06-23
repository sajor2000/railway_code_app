# BioBERT Strategy for Railway Deployment

## Challenge
BioBERT requires heavy ML dependencies (torch, transformers, tensorflow) that exceed Railway's build limits and memory constraints, but we need BioBERT's medical domain accuracy.

## Solution: Hybrid Architecture

### 🏥 **Medical Domain Accuracy Strategy**

#### 1. **Pinecone Vector Database (Primary)**
- **Pre-computed BioBERT embeddings** for common medical terms stored in Pinecone
- Contains 768-dimensional BioBERT vectors for:
  - ICD-10-CM codes and descriptions
  - SNOMED CT concepts
  - RxNorm drug names
  - LOINC laboratory terms
  - Medical abbreviations and terminology

#### 2. **OpenAI Embeddings (Fallback)**
- Used only for new/rare medical terms not in Pinecone
- Provides consistent 768-dimensional vectors
- Maintains search functionality while BioBERT database grows

#### 3. **Local Development (Full BioBERT)**
- Use `requirements.txt` for local development with full BioBERT
- Use `requirements-minimal.txt` for Railway deployment
- Best of both worlds approach

### 🚀 **Railway Deployment Benefits**

1. **Fast Builds**: No heavy ML dependencies to download/compile
2. **Low Memory**: Minimal runtime memory usage
3. **Medical Accuracy**: Leverages pre-computed BioBERT embeddings
4. **Scalable**: Pinecone handles vector operations efficiently
5. **Cost Effective**: ~$7/month Railway + Pinecone free tier

### 📊 **Implementation Details**

#### Pinecone Index Structure
```
Index: biobert-medical
- Dimension: 768 (BioBERT standard)
- Metric: cosine
- Vectors: Pre-computed BioBERT embeddings for medical terminology
- Metadata: {code, system, description, type}
```

#### Code Behavior
```python
def get_medical_embedding(text):
    # 1. Search Pinecone for existing BioBERT embedding
    if found_in_pinecone:
        return biobert_vector_from_pinecone
    
    # 2. Fallback to OpenAI for new terms
    else:
        return openai_embedding(text)
        # Optionally: queue for BioBERT processing offline
```

### 🔧 **Setup Instructions**

1. **Populate Pinecone with BioBERT embeddings** (one-time setup):
   ```bash
   # Local environment with full BioBERT
   pip install -r requirements.txt
   python scripts/populate_biobert_embeddings.py
   ```

2. **Deploy to Railway**:
   ```bash
   # Uses lightweight requirements-minimal.txt
   # Leverages pre-computed embeddings in Pinecone
   ```

### 🎯 **Medical Use Cases Supported**

- ✅ Medical code similarity (ICD-10, SNOMED, etc.)
- ✅ Drug-disease relationship mapping
- ✅ Clinical text classification
- ✅ Medical entity extraction
- ✅ Biomedical literature search
- ✅ Medical abbreviation expansion

This architecture provides production-grade medical NLP capabilities while maintaining Railway deployment compatibility.