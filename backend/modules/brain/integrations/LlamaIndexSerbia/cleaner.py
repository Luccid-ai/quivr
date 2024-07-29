import os

from converter_md_to_txt import convert_md_to_txt_and_delete

# Function to load a Markdown file
def load_md_file(file_path):
    # Open the file in read mode with utf-8 encoding
    with open(file_path, 'r', encoding='utf-8') as file:
        # Read the content of the file
        text = file.read()
    return text

# Function to save the modified content back to the Markdown file
def save_md_file(file_path, content):
    # Open the file in write mode with utf-8 encoding
    with open(file_path, 'w', encoding='utf-8') as file:
        # Write the modified content to the file
        file.write(content)

# Function to check lines starting with '|'
def check_lines_starting_with_pipe(content):
    # Split the content into lines
    lines = content.split('\n')
    
    # Initialize a list to store lines that start with '|'
    modified_lines = []

    # Iterate through each line in the content with index
    for index, line in enumerate(lines):
        # Call the function to process lines starting with '|'
        modified_line = process_line(index, line)

        if not "---" in line:
            modified_lines.append(modified_line)
    
    # Join the modified lines back into a single string
    modified_content = '\n'.join(modified_lines)
    return modified_content

# Function to process a line and check if it starts with '|'
def process_line(index, line):
    # Check if the line starts with '|'
    if line.startswith('|'):
        print(line)
        
        # Replace all commas in the line with an empty string
        line = line.replace(',', '')
        # Replace the first '|' with an empty string
        line = line.replace('|', '', 1)
        # Replace all subsequent '|' with commas
        line = line.replace('|', ',')
        
        # Replace specific characters with an empty string
        line = line.replace('*', '')
        line = line.replace('<sup>', '')
        line = line.replace('</sup>', '')
        line = line.replace('<br>', ' ')

        print(line)
        print()
    return line

# Function to process all Markdown files in a directory
def process_all_md_files_in_directory(directory_path):
    # Iterate through all files in the given directory
    for file_name in os.listdir(directory_path):
        # Check if the file has a .md extension
        if file_name.endswith('.md'):
            file_path = os.path.join(directory_path, file_name)
            
            print(f"Processing file: {file_path}")

            # Load the content of the Markdown file
            md_content = load_md_file(file_path)

            # Check and modify lines starting with '|'
            modified_content = check_lines_starting_with_pipe(md_content)

            # Save the modified content back to the Markdown file
            save_md_file(file_path, modified_content)

            print("Modifications saved successfully for file:", file_name)

# Example usage of the functions
directory_path = 'C:/Users/rakin/OneDrive/Documents/Luccid/quivr/luccid-data/data/Documents/SerbiaGemini'

if __name__ == "__main__":
    # Process all Markdown files in the specified directory
    process_all_md_files_in_directory(directory_path)
    convert_md_to_txt_and_delete(directory_path)
