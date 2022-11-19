import json
from pathlib import Path

import pandas as pd
from fastai.vision.all import *

f1_macro = F1Score(average="macro")


if Path("/kaggle").exists():
    # This is in Kaggle
    initial_path = Path(
        "/kaggle", "input", "deforestation", "train_test_data", "train_test_data"
    )
    output_path = Path("/kaggle", "working")
    df_train = pd.read_csv(Path("/kaggle", "input", "deforestation") / "train.csv")
else:
    initial_path = Path("data") / "train_test_data"
    df_train = pd.read_csv(Path("data") / "train.csv")
    output_path = Path("artifacts")
    output_path.mkdir(exist_ok=True)

# Preprocess data
df_train["short_path"] = df_train["example_path"].apply(lambda x: Path(x).name)
train_labels = {row["short_path"]: row["label"] for i, row in df_train.iterrows()}
def label(file_name):
    return train_labels.get((Path(file_name)).name, 0)


def main():

    # Get image data
    base_path = initial_path / "train"
    test_path = initial_path / "test"

    all_files = get_image_files(str(base_path))

    # Image loader
    all_dls = ImageDataLoaders.from_path_func(
        base_path,
        all_files,
        label_func=label,
        item_tfms=Resize(224),
        batch_tfms=aug_transforms(size=224),
        valid_pct=0,
    )

    # Create learner
    learn = vision_learner(
        all_dls,
        resnet50,
        metrics=[error_rate, f1_macro],
        model_dir=output_path,
    )

    # Train learner
    lr = learn.lr_find()
    learn.fine_tune(15, lr.valley)

    # Save learner
    learn.export(output_path / "model.pkl")

    # Get predictions
    test_predictions = {}
    test_scores = {}
    for file in test_path.glob("*.png"):
        prediction = learn.predict(file)
        test_predictions[file.name.replace(".png", "")] = int(prediction[0])
        test_scores[file.name] = prediction[2].numpy()

    test_predictions = {k: test_predictions[k] for k in sorted(test_predictions.keys())}

    # Save scores
    pd.DataFrame(test_scores).T.reset_index().rename(
        columns={"index": "short_path"}
    ).to_csv(output_path / "test_scores.csv", index=False)

    full_test_predictions = {"target": test_predictions}

    with open(output_path / "test_predictions.json", "w") as f:
        json.dump(full_test_predictions, f)


if __name__ == "__main__":
    main()