import pdfplumber
import pandas
import os

# Specify the folder path
folder_path = 'Bill Project/Rajehswari rice'

# List all files in the folder
files_in_folder = os.listdir(folder_path)
output_dir = 'nlp/dataset'
# Print the list of files
for x in range(20):
    
    file_path = os.path.join(folder_path,files_in_folder[x] )

    #file = "C:/Users/paritosh/Downloads/Cosoha/ANKHI.pdf609cb1b86834247a628960c8_20230801_19_17_53_PM._TCAKBc_pstUWN.pdf"
    with pdfplumber.open(file_path) as pdf:
        first_page = pdf.pages[0]
        layout_text1 = first_page.extract_text(x_tolerance=3, y_tolerance=3, layout=False, x_density=7.25, y_density=13)
    layout_text = first_page.extract_text(x_tolerance=3, y_tolerance=3, layout=True, x_density=7.25, y_density=13)
    cleaned_text = ' '.join(layout_text.split())
    
    output_file = os.path.join(output_dir, files_in_folder[x].replace('.pdf', '.txt'))
    
        # Write the extracted text into the text file
    with open(output_file, 'w', encoding='utf-8') as txt_file:
        txt_file.write(cleaned_text[:750])

    print("Text saved to:", output_file)