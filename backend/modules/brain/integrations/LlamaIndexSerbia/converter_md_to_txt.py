import os
import cyrtranslit

def convert_md_to_txt_and_delete(directory):
    # Check if the directory is valid
    if not os.path.isdir(directory):
        print(f"{directory} is not a valid directory.")
        return
    
    # Iterate over all files in the directory
    for filename in os.listdir(directory):
        # Check if the file has a .md extension
        if filename.endswith(".md"):
            # Form the path to the .md file
            md_file_path = os.path.join(directory, filename)
            # Form the path for the .txt file
            txt_file_path = os.path.join(directory, filename.replace(".md", ".txt"))
            
            # Read the content of the .md file
            with open(md_file_path, 'r', encoding='utf-8') as md_file:
                content = md_file.read()
            
            # Convert the content from Cyrillic to Latin
            content_latin = cyrtranslit.to_latin(content, "sr")
            
            # Write the content to the .txt file
            with open(txt_file_path, 'w', encoding='utf-8') as txt_file:
                txt_file.write(content_latin)
            
            # Delete the original .md file
            os.remove(md_file_path)
            
            print(f"Converted {md_file_path} to {txt_file_path} and deleted the original .md file")

if __name__ == "__main__":
    # Set the path to the directory with .md files
    directory_path = "C:/Users/rakin/OneDrive/Documents/Luccid/quivr/luccid-data/data/Documents/SerbiaGemini"
    convert_md_to_txt_and_delete(directory_path)
