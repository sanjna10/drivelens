def filter_by_object(results, target_class):
    """Keep retrieved images containing the requested detector class."""

    filtered = []
    target_class = target_class.lower()

    for result in results:
        matching = [
            detection
            for detection in result["detections"]
            if detection["class_name"].lower() == target_class
        ]

        if not matching:
            continue

        best_detection = max(
            matching,
            key=lambda detection: detection["confidence"],
        )

        result["best_detection"] = best_detection
        filtered.append(result)

    return filtered


def rerank_results(results, alpha=0.7, beta=0.3):
    """Fuse CLIP similarity with the strongest matching detector confidence."""

    for result in results:
        clip_score = result["clip_score"]
        detection_score = result["best_detection"]["confidence"]

        result["final_score"] = (
            alpha * clip_score
            + beta * detection_score
        )

    return sorted(
        results,
        key=lambda result: result["final_score"],
        reverse=True,
    )
