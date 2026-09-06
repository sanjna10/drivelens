from clip_retrieval import CLIPRetriever
from detection import ObjectDetector
from ranking import filter_by_object, rerank_results


class DrivingLensPipeline:
    """End-to-end CLIP retrieval + RetinaNet verification pipeline."""

    def __init__(self, device=None):
        self.retriever = CLIPRetriever(device=device)
        self.detector = ObjectDetector()

    def retrieve_and_detect(
        self,
        query,
        retrieval_k=8,
        confidence_threshold=0.5,
    ):
        candidates = self.retriever.retrieve_images(
            query,
            k=retrieval_k,
        )

        results = []

        for candidate in candidates:
            detections = self.detector.detect_objects(
                candidate["image_path"],
                confidence_threshold=confidence_threshold,
            )

            results.append(
                {
                    "image_path": candidate["image_path"],
                    "clip_score": candidate["clip_score"],
                    "detections": detections,
                }
            )

        return results

    def search(
        self,
        query,
        target_class,
        retrieval_k=20,
        final_k=5,
        confidence_threshold=0.5,
        alpha=0.7,
        beta=0.3,
    ):
        # 1. Semantic retrieval + object detection
        results = self.retrieve_and_detect(
            query,
            retrieval_k=retrieval_k,
            confidence_threshold=confidence_threshold,
        )

        # 2. Verify the requested object
        results = filter_by_object(
            results,
            target_class,
        )

        # 3. Fuse CLIP similarity + detector confidence
        results = rerank_results(
            results,
            alpha=alpha,
            beta=beta,
        )

        # 4. Return final Top-K
        return results[:final_k]
