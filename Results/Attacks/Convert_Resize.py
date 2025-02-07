from PIL import Image

def convert_and_resize(input_file, output_file, new_size=(512, 512)):
    """Converts a JPEG image to BMP and resizes it to the specified size.

    Args:
        input_file: The path to the input JPEG file.
        output_file: The path to the output BMP file.
        new_size: The desired size of the output image as a tuple (width, height).
    """

    try:
        # Open the JPEG image
        img = Image.open(input_file)

        # Resize the image
        resized_img = img.resize(new_size)

        # Save the resized image as BMP
        resized_img.save(output_file, format='BMP')

        print(f"Image converted and resized to {new_size[0]}x{new_size[1]} pixels.")

    except Exception as e:
        print(f"Error: {e}")

# Example usage:
input_image = "watermarked_B1_CROP_75.bmp"
output_image = "watermarked_B1_CROP_75.bmp"

convert_and_resize(input_image, output_image)