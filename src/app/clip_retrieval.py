import torch
import faiss
import pandas as pd
import open_clip

from config import (
    CLIP_MODEL_NAME,
    CLIP_PRETRAINED,
    FAISS_INDEX_FILE,
    IMAGE_METADATA_FILE,
)


class CLIPRetriever:
    """Text-to-image retrieval using OpenCLIP embeddings and a FAISS index."""

    def __init__(self, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            CLIP_MODEL_NAME,
            pretrained=CLIP_PRETRAINED,
        )
        self.tokenizer = open_clip.get_tokenizer(CLIP_MODEL_NAME)

        self.model = self.model.to(self.device)
        self.model.eval()

        self.index = faiss.read_index(str(FAISS_INDEX_FILE))

        metadata = pd.read_csv(IMAGE_METADATA_FILE)
        self.valid_paths = metadata["image_path"].tolist()

    @torch.no_grad()
    def encode_text(self, text):
        tokens = self.tokenizer([text]).to(self.device)

        features = self.model.encode_text(tokens)
        features = features / features.norm(dim=-1, keepdim=True)

        return features.cpu().numpy().astype("float32")

    def retrieve_images(self, query, k=10):
        query_embedding = self.encode_text(query)

        scores, indices = self.index.search(query_embedding, k)

        results = []

        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue

            results.append(
                {
                    "image_path": self.valid_paths[idx],
                    "clip_score": float(score),
                }
            )

        return results
