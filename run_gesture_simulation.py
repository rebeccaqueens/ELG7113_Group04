"""Send confident gesture predictions to the original two-arm playback."""

from contextlib import closing

MIN_CONFIDENCE_PERCENT = 30.0 # what percent confidence is sent to the robotic simulation 


def gesture_requests(predictions, minimum_confidence=MIN_CONFIDENCE_PERCENT):
    """Yield a new label as soon as its smoothed confidence meets the cutoff."""
    if not 0 <= minimum_confidence <= 100:
        raise ValueError("The confidence threshold must be between 0 and 100.")

    last_sent = None
    for label, frame, seconds, confidence in predictions:
        if not confidence >= minimum_confidence or label == last_sent:
            continue

        print(
            f"[Controller] START {label} from video frame {frame} "
            f"({seconds:.2f} s, confidence {confidence:.1f}%).",
            flush=True,
        )
        last_sent = label
        yield label


def report_result(label, processed, total, rejected):
    if processed < total:
        status = "INTERRUPTED"
    elif rejected:
        status = "FINISHED WITH SKIPPED FRAMES"
    else:
        status = "COMPLETE"
    print(
        f"[Controller] {status} {label}: {processed - rejected}/{total} "
        f"paired frames played, {rejected} rejected.",
        flush=True,
    )


def main():
    from gesture_recognizer import iter_predictions
    from gesture_simulator import run

    with closing(iter_predictions()) as predictions:
        with closing(gesture_requests(predictions)) as requests:
            run(requests, report_result)


if __name__ == "__main__":
    main()
