from pipeline import DrivingLensPipeline
from visualization import show_final_results


def main():
    pipeline = DrivingLensPipeline()

    results = pipeline.search(
        query="a pedestrian crossing the road",
        target_class="pedestrian",
        retrieval_k=10,
        final_k=5,
        confidence_threshold=0.5,
    )

    for i, result in enumerate(results, start=1):
        print(f"\nIMAGE {i}")
        print("Path:", result["image_path"])
        print("CLIP score:", result["clip_score"])
        print("Final score:", result["final_score"])
        print("Best detection:", result["best_detection"])

    show_final_results(results)


if __name__ == "__main__":
    main()
