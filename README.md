# Driving Lens

**Driving Lens** is a multimodal autonomous-driving scene retrieval pipeline that combines **CLIP**, **FAISS**, and **RetinaNet with a ResNet-50 backbone**.

The system uses natural-language queries to retrieve semantically relevant driving scenes and then applies object detection to verify and rerank the retrieved results.

## Architecture

```text
                         Natural-Language Query
                                  │
                                  ▼
                         CLIP Text Encoder
                                  │
                           Query Embedding
                                  │
                                  ▼
Driving Images ──► CLIP Image Encoder ──► Image Embeddings
                                             │
                                             ▼
                                        FAISS Index
                                             │
                                  Semantic Top-K Retrieval
                                             │
                                             ▼
                                   RetinaNet + ResNet-50
                                             │
                                  Object Detection
                                             │
                                             ▼
                                   Target-Class Filter
                                             │
                                             ▼
                         ┌──────────────────────────┐
                         │ CLIP Similarity          │
                         │ Detection Confidence     │
                         └────────────┬─────────────┘
                                      │
                                      ▼
                              Score Fusion
                                      │
                                      ▼
                           Final Ranked Scenes
```

## Workflow

### 1. Object Detection

A **RetinaNet** detector with a pretrained **ResNet-50** backbone is trained using the COCO-style annotations from the autonomous-driving dataset.

```text
Driving Image
      │
      ▼
ResNet-50 Backbone
      │
      ▼
RetinaNet
      │
      ├── Object Class
      ├── Bounding Box
      └── Detection Confidence
```

The detector recognizes 12 driving-related categories including cars, pedestrians, bikers, trucks, traffic lights, and obstacles.

---

### 2. Multimodal Embedding Generation

A pretrained **OpenCLIP ViT-B/32** model maps driving images into a shared vision-language embedding space.

```text
Driving Image
      │
      ▼
CLIP Image Encoder
      │
      ▼
Normalized Image Embedding
```

The image embeddings are stored in a **FAISS** vector index for efficient similarity search.

---

### 3. Zero-Shot Semantic Retrieval

A natural-language query is encoded using the CLIP text encoder.

```text
"a pedestrian crossing the road"
                │
                ▼
        CLIP Text Encoder
                │
                ▼
        Query Embedding
                │
                ▼
             FAISS
                │
                ▼
        Top-K Similar Scenes
```

Because the pretrained CLIP model is used without task-specific fine-tuning, this stage performs **zero-shot multimodal retrieval**.

---

### 4. Object-Level Verification

The Top-K scenes retrieved by FAISS are passed through RetinaNet.

```text
Top-K CLIP Results
        │
        ▼
RetinaNet
        │
        ▼
Detected Objects
        │
        ▼
Target-Class Filtering
```

For example, for the query:

```text
"a pedestrian crossing the road"
```

the semantic retrieval stage finds relevant scenes, while the detector verifies whether a `pedestrian` is actually present and provides its bounding box and confidence.

This combines **global semantic understanding** from CLIP with **localized object-level evidence** from RetinaNet.

---

### 5. Multimodal Reranking

Each verified candidate contains two signals:

```text
CLIP Similarity
      +
Detection Confidence
```

The current ranking function combines them as:

```text
Final Score =
0.7 × CLIP Similarity
+
0.3 × Detection Confidence
```

Candidates are sorted by the fused score to produce the final ranked scenes.

---

## End-to-End Flow

```text
                       OFFLINE
                         │
        ┌────────────────┴────────────────┐
        │                                 │
Driving Images                    COCO Annotations
        │                                 │
        ▼                                 ▼
 CLIP Image Encoder              RetinaNet Training
        │                                 │
        ▼                                 ▼
 Image Embeddings                Trained Detector
        │
        ▼
    FAISS Index


                       ONLINE
                         │
                  Natural-Language Query
                         │
                         ▼
                  CLIP Text Encoder
                         │
                         ▼
                    FAISS Search
                         │
                         ▼
                  Top-K Candidates
                         │
                         ▼
                RetinaNet Detection
                         │
                         ▼
                 Object Verification
                         │
                         ▼
        CLIP Similarity + Detection Confidence
                         │
                         ▼
                     Reranking
                         │
                         ▼
                 Final Driving Scenes
```

## Module Responsibilities

```text
clip_retrieval.py
    CLIP text encoding
    FAISS semantic retrieval

detection.py
    RetinaNet inference
    object localization
    detection confidence

ranking.py
    target-object filtering
    score fusion
    reranking

pipeline.py
    end-to-end pipeline orchestration

visualization.py
    final scene and bounding-box visualization

config.py
    model and artifact configuration

main.py
    pipeline entry point
```

## Summary

Driving Lens implements a two-stage multimodal retrieval architecture:

```text
Semantic Retrieval
CLIP + FAISS
       │
       ▼
Perception Verification
RetinaNet + ResNet-50
       │
       ▼
Multimodal Score Fusion
       │
       ▼
Final Ranked Scenes
```

The architecture combines the flexible natural-language search capabilities of **CLIP** with the spatial and object-level predictions of **RetinaNet**, enabling semantic search and verification over autonomous-driving scenes.
