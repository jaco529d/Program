from PIL import Image

def resize_image(image_path):
    # Open the image file
    with Image.open(image_path) as img:
        # Get the original size of the image
        width, height = img.size
        
        # Determine the size of the new image
        if width > height:
            new_width = 32
            new_height = int(height * (new_width / width))
        else:
            new_height = 32
            new_width = int(width * (new_height / height))
        
        # Resize the image
        resized_img = img.resize((new_width, new_height))
        
        # Create a new blank image with the target size
        target_img = Image.new('RGB', (32, 32), (0, 0, 0))
        
        # Calculate the position to paste the resized image
        x_offset = (32 - new_width) // 2
        y_offset = (32 - new_height) // 2
        
        # Paste the resized image onto the target image
        target_img.paste(resized_img, (x_offset, y_offset))

        #convert to greyscale
        target_img = target_img.convert("L") 
        
            #target_img.save('resized_image.png')

        #finds pixelvalues, and stores in list
        pixelValues = list(target_img.getdata())

        return pixelValues

# Example usage:
a = resize_image('C:/Users/jacob/OneDrive - Syddansk Erhvervsskole/Programering/Eksamensprojekt/dataset/training_set/cats/cat.1.jpg')
print('DONE', a)