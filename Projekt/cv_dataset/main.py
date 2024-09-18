import os
from PIL import Image, ImageOps, ExifTags
"""
# bikeee ##############################################################

# alte folder path for bike
source_folder = "C:/Users/Soyoung/Desktop/cvss-24-gruppe-1/Projekt/cv_dataset/bike_neu"

# new folder path for bike 
destination_folder = "C:/Users/Soyoung/Desktop/cvss-24-gruppe-1/Projekt/cv_dataset/converted_bike"

# prefix bike
new_name_prefix = "bike"

"""
"""
# schilderr ##############################################################

# alte folder path for shild
source_folder = "C:/Users/Soyoung/Desktop//cvss-24-gruppe-1/Projekt/cv_dataset/shield_neu"

# new folder path for shild 
destination_folder = "C:/Users/Soyoung/Desktop/cvss-24-gruppe-1/Projekt/cv_dataset/converted_shield"

# prefix shield
new_name_prefix = "shield"

"""
# bin ##############################################################

# alte folder path for bin
source_folder = "C:/Users/Soyoung/Desktop/cvss-24-gruppe-1/Projekt/cv_dataset/bin_neuneu"


# new folder path for bin 
destination_folder = "C:/Users/Soyoung/Desktop/cvss-24-gruppe-1/Projekt/cv_dataset/converted_bin"

# prefix bin
new_name_prefix = "bin"

##############################################################



if not os.path.exists(destination_folder):
    os.makedirs(destination_folder)


def get_existing_numbers(base_path, prefix):
    existing_numbers = []
    for filename in os.listdir(base_path):
        if filename.startswith(prefix) and filename.endswith(".png"):
            # index only nr. (ie: bin074.png -> 074)
            num_part = filename[len(prefix):].split(".")[0]
            try:
                existing_numbers.append(int(num_part))
            except ValueError:
                continue
    return sorted(existing_numbers)


def get_next_filename(prefix, existing_numbers):
    counter = 1
    if existing_numbers:
        counter = max(existing_numbers) + 1  # starts from biggest nr. from existing things
    new_filename = f"{prefix}{counter:03}.png"
    return new_filename, counter

existing_numbers = get_existing_numbers(destination_folder, new_name_prefix)


for filename in os.listdir(source_folder):
    if filename.endswith((".jpg", ".jpeg", ".bmp", ".png")):
        img_path = os.path.join(source_folder, filename)
        new_filename, file_counter = get_next_filename(new_name_prefix, existing_numbers)
        new_img_path = os.path.join(destination_folder, new_filename)
        
        img = Image.open(img_path)

        try:
            for orientation in ExifTags.TAGS.keys():
                if ExifTags.TAGS[orientation] == 'Orientation':
                    break
            exif = img._getexif()
            
            if exif is not None:
                orientation = exif.get(orientation)
                
                if orientation == 3:
                    img = img.rotate(180, expand=True)
                elif orientation == 6:
                    img = img.rotate(270, expand=True)
                elif orientation == 8:
                    img = img.rotate(90, expand=True)
        except (AttributeError, KeyError, IndexError):

            pass

        width, height = img.size
        
        if width > height:
            print(f"{filename}: Landscape")
        elif height > width:
            print(f"{filename}: Portrait")
        else:
            print(f"{filename}: Square")

        size = min(img.size)
        
        img = ImageOps.fit(img, (size, size), Image.Resampling.LANCZOS)
        
        img.save(new_img_path, "PNG")
        existing_numbers.append(file_counter)
        print(f"changed: {filename} -> {new_filename}")

print("convert done!")