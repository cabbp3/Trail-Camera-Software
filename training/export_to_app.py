"""
Copy trained ONNX artifacts into the main app models folder.
This is a convenience script; no training runs here.
"""
import argparse
import os
import shutil


def copy_if(src, dst):
    if src:
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy(src, dst)
        print(f"Copied {src} -> {dst}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--detector", help="Path to detector ONNX to install as species model")
    parser.add_argument("--detector-labels", help="Path to detector labels.txt")
    parser.add_argument("--classifier", help="Path to classifier ONNX to install as species model")
    parser.add_argument("--classifier-labels", help="Path to classifier labels.txt")
    parser.add_argument("--reid", help="Path to reid ONNX")
    args = parser.parse_args()

    models_dir = os.path.join(os.path.dirname(__file__), "..", "models")
    os.makedirs(models_dir, exist_ok=True)

    # Prefer classifier if provided; otherwise detector labels can drive suggested_tag
    if args.classifier:
        copy_if(args.classifier, os.path.join(models_dir, "species.onnx"))
        if args.classifier_labels:
            copy_if(args.classifier_labels, os.path.join(models_dir, "labels.txt"))
    elif args.detector:
        copy_if(args.detector, os.path.join(models_dir, "species.onnx"))
        if args.detector_labels:
            copy_if(args.detector_labels, os.path.join(models_dir, "labels.txt"))

    if args.reid:
        copy_if(args.reid, os.path.join(models_dir, "reid.onnx"))

    print("Done. Launch the app to use the newly installed models.")


if __name__ == "__main__":
    main()
