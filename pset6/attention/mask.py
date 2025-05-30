import sys
import tensorflow as tf

from PIL import Image, ImageDraw, ImageFont
from transformers import AutoTokenizer, TFBertForMaskedLM

# Pre-trained masked language model
MODEL = "bert-base-uncased"

# Number of predictions to generate
K = 3

# Constants for generating attention diagrams
FONT = ImageFont.truetype("assets/fonts/OpenSans-Regular.ttf", 28)
GRID_SIZE = 40
PIXELS_PER_WORD = 200


def main():
    # دریافت جمله از کاربر
    text = input("Text: ")

    # بارگذاری توکنایزر
    tokenizer = AutoTokenizer.from_pretrained(MODEL)

    # توکن‌سازی جمله‌ی ورودی
    inputs = tokenizer(text, return_tensors="tf")

    # پیدا کردن موقعیت توکن [MASK]
    mask_token_index = get_mask_token_index(tokenizer.mask_token_id, inputs)
    if mask_token_index is None:
        sys.exit(f"Input must include mask token {tokenizer.mask_token}.")

    # بارگذاری مدل BERT و انجام پیش‌بینی
    model = TFBertForMaskedLM.from_pretrained(MODEL)
    result = model(**inputs, output_attentions=True)

    # گرفتن logits مربوط به توکن [MASK]
    mask_token_logits = result.logits[0, mask_token_index]

    # انتخاب K توکن محتمل‌تر
    top_tokens = tf.math.top_k(mask_token_logits, K).indices.numpy()
    print("\nPredictions:")
    for token_id in top_tokens:
        predicted_token = tokenizer.decode([token_id])
        print(text.replace(tokenizer.mask_token, predicted_token))

    # نمایش نمودارهای Attention
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    visualize_attentions(tokens, result.attentions)


def get_mask_token_index(mask_token_id, inputs):
    input_ids = inputs["input_ids"][0].numpy().tolist()
    try:
        return input_ids.index(mask_token_id)
    except ValueError:
        return None


def get_color_for_attention_score(attention_score):
    intensity = int(attention_score * 255)
    return (intensity, intensity, intensity)


def visualize_attentions(tokens, attentions):
    num_layers = len(attentions)
    num_heads = attentions[0].shape[1]
    for layer in range(num_layers):
        for head in range(num_heads):
            attention = attentions[layer][0][head]
            diagram = generate_diagram(
                layer + 1,
                head + 1,
                tokens,
                attention
            )
            diagram.save(f"attention_layer{layer+1}_head{head+1}.png")



def generate_diagram(layer_number, head_number, tokens, attention_weights):
    """
    Generate a diagram representing the self-attention scores for a single
    attention head. The diagram shows one row and column for each of the
    `tokens`, and cells are shaded based on `attention_weights`, with lighter
    cells corresponding to higher attention scores.

    The diagram is saved with a filename that includes both the `layer_number`
    and `head_number`.
    """
    # Create new image
    image_size = GRID_SIZE * len(tokens) + PIXELS_PER_WORD
    img = Image.new("RGBA", (image_size, image_size), "black")
    draw = ImageDraw.Draw(img)

    # Draw each token onto the image
    for i, token in enumerate(tokens):
        # Draw token columns
        token_image = Image.new("RGBA", (image_size, image_size), (0, 0, 0, 0))
        token_draw = ImageDraw.Draw(token_image)
        token_draw.text(
            (image_size - PIXELS_PER_WORD, PIXELS_PER_WORD + i * GRID_SIZE),
            token,
            fill="white",
            font=FONT
        )
        token_image = token_image.rotate(90)
        img.paste(token_image, mask=token_image)

        # Draw token rows
        _, _, width, _ = draw.textbbox((0, 0), token, font=FONT)
        draw.text(
            (PIXELS_PER_WORD - width, PIXELS_PER_WORD + i * GRID_SIZE),
            token,
            fill="white",
            font=FONT
        )

    # Draw each word
    for i in range(len(tokens)):
        y = PIXELS_PER_WORD + i * GRID_SIZE
        for j in range(len(tokens)):
            x = PIXELS_PER_WORD + j * GRID_SIZE
            color = get_color_for_attention_score(attention_weights[i][j])
            draw.rectangle((x, y, x + GRID_SIZE, y + GRID_SIZE), fill=color)

    # Save image
    img.save(f"Attention_Layer{layer_number}_Head{head_number}.png")


if __name__ == "__main__":
    main()
