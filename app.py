import pathlib
import sys
from pathlib import Path

# Patch WindowsPath to PosixPath when running on Linux (needed for fastai models trained on Windows)
if sys.platform != "win32":
    pathlib.WindowsPath = pathlib.PosixPath
    try:
        import pathlib._local
        pathlib._local.WindowsPath = pathlib.PosixPath
    except ImportError:
        pass

import gradio as gr
from fastai.vision.all import PILImage, load_learner



BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "my_cat_classifier.pkl"


THRESHOLD = 0.90


learn = load_learner(MODEL_PATH, cpu=True)


def classify_image(image):
    """Predict whether the uploaded image contains my cat."""

    if image is None:
        return "Please upload an image first.", {}

    try:
        fastai_image = PILImage.create(image)

        _, _, probabilities = learn.predict(fastai_image)

        vocab = list(learn.dls.vocab)

        class_probabilities = {
            class_name: float(probability)
            for class_name, probability in zip(vocab, probabilities)
        }

        my_cat_probability = class_probabilities["my_cat"]

        if my_cat_probability >= THRESHOLD:
            result = "This is probably my cat 🐈"
        else:
            result = "This is probably not my cat"

        result_text = (
            f"{result}\n\n"
            f"My cat probability: {my_cat_probability:.2%}\n"
            f"Acceptance threshold: {THRESHOLD:.0%}"
        )

        return result_text, class_probabilities

    except Exception as error:
        return f"Could not process the image: {error}", {}


css = """
.gradio-container {
    max-width: 1050px !important;
    margin: 0 auto !important;
}

#title-section {
    text-align: center;
    margin-bottom: 20px;
}

#classify-button {
    min-height: 48px;
    font-size: 16px;
    font-weight: 600;
}

#result-box textarea {
    font-size: 16px;
}
"""


with gr.Blocks(
    title="Is This My Cat?"
) as demo:

    gr.Markdown(
        """
        # Is This My Cat?

        Upload a photo and the model will estimate whether it is my cat.
        """,
        elem_id="title-section"
    )

    with gr.Row(equal_height=True):

        with gr.Column(scale=1):
            image_input = gr.Image(
                type="pil",
                label="Upload an image",
                height=420
            )

            classify_button = gr.Button(
                "Is this my cat?",
                variant="primary",
                elem_id="classify-button"
            )

        with gr.Column(scale=1):
            result_output = gr.Textbox(
                label="Result",
                lines=5,
                interactive=False,
                elem_id="result-box"
            )

            probability_output = gr.Label(
                label="Class probabilities",
                num_top_classes=2
            )

    gr.Examples(
        examples=[
            str(BASE_DIR / "examples" / "5357079487188769839.jpg"),
            str(BASE_DIR / "examples" / "5359331287002455078.jpg"),
            str(BASE_DIR / "examples" / "5359331287002455087.jpg"),
            str(BASE_DIR / "examples" / "5359331287002455096.jpg"),
            str(BASE_DIR / "examples" / "0_0090.jpg"),
            str(BASE_DIR / "examples" / "0_0091.jpg"),
            str(BASE_DIR / "examples" / "0_0092.jpg"),
            str(BASE_DIR / "examples" / "0_0093.jpg"),
        ],
        inputs=image_input,
        label="Test examples"
    )

    classify_button.click(
        fn=classify_image,
        inputs=image_input,
        outputs=[
            result_output,
            probability_output
        ]
    )


if __name__ == "__main__":
    demo.launch(allowed_paths=[str(BASE_DIR / "examples")], css=css)