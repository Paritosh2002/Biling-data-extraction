import gradio as gr 
from pdf2image import convert_from_path
import os
import shutil
#import easyocr
from yolo_predict import yolo_prediction
import cv2
from paddleocr import PaddleOCR
from nlp.head_ner import ner 
import numpy as np
import pandas as pd
import re
import datetime
# Function to process the uploaded files
image_folder = 'file_images'
def process_files(files):
    if not os.path.exists(image_folder):
        os.makedirs(image_folder)
    combined_df = pd.DataFrame(columns=['timestamp', 'document','description', 'amount', 'qty', 'invoice number', 'invoice date',
                                        'buyer', 'seller', 'gst no', 'fssai no'])
    for file in files:
        if file.name.endswith(('.jpg', '.jpeg', '.png')):
            # Save the image directly
            path = f"{image_folder}/" + os.path.basename(file)  #NB*
            shutil.copyfile(file, path)
            #image_path = os.path.join('file_images', file.name)
            """with open(image_path, 'wb') as f:
                f.write(file.getvalue()) """ # Use getvalue() to get the file content
            #print(f"Image saved: {image_path}")
        elif file.name.endswith('.pdf'):
            # Convert PDF to images
            images = convert_from_path(file.name)
            #print(images)
            
            # Save each image in the folder
            file_name = os.path.basename(file).split('.pdf')
            #print(file_name)
            for i, image in enumerate(images):
                
                path = "file_images/" + os.path.basename(f'{file_name[0]}_{i}.jpg')  #NB*
                #shutil.copyfile(file, path)
                image.save(path,'JPEG')
                
        else:
            print(f"Unsupported file type: {file.name}")
    
    files_list = os.listdir(image_folder)
    for file_name in files_list:
        file_path = os.path.join(image_folder, file_name)
        final_dic = ocr(file_path)
        final_dic_keys = [key for key in final_dic]
        temp = [datetime.datetime.now(),file_name ]
        for columns in combined_df.columns[2:]:
            flag = False
            for col in final_dic_keys:
                if columns == col:
                    if type(final_dic[col]) is list:
                        if len(final_dic[col]) !=0:
                            temp.append(final_dic[col][0])
                        else :
                            temp.append(' ')
                    else :
                        temp.append(final_dic[col])
                    flag = True 
                    break
            if not flag:
                temp.append(' ')
        final = []
        
        final.append(temp)
        df = pd.DataFrame(final,columns=['timestamp', 'document','description', 'amount', 'qty', 'invoice number', 'invoice date',
                                        'buyer', 'seller', 'gst no', 'fssai no'])
        

                


    # Save the DataFrame to an Excel file
        combined_df = pd.concat([combined_df, df], ignore_index=True)

    #deleting file after ocr is performed 
    for file_name in files_list:
        file_path = os.path.join(image_folder, file_name)
        if os.path.isfile(file_path):
            os.remove(file_path)
    excel_path = 'output_combined.xlsx'
    
    # Save the combined DataFrame to an Excel file
    combined_df.to_excel(excel_path, index=False)

    return excel_path
def head(final_text):
    dic = {}
    ner_result = ner(final_text)
    #print(ner_result)
    for labels in ner_result:
        if labels['label'] == 'FSSAI_DATA':
            dic['fssai no'] = labels['text']
        elif labels['label'] == 'INVOICE_DATA':
            dic['invoice number'] = labels['text']
        elif labels['label'] == 'GST_DATA':
            dic['gst no'] = labels['text']
        elif labels['label'] == 'INVOICE_DATE':
            dic['invoice date'] = labels['text']
        elif labels['label'] == 'BUYER':
            dic['buyer'] = labels['text']
        elif labels['label'] == 'SELLER':
            dic['seller'] = labels['text']
    #print(dic)
    return dic 

def table(result):
    sorted_boxes = sorted(result, key=lambda box: box[0][0][1])
    table_data = []
    for line in sorted_boxes[0]:
        text = line[1][0]
        bbox = line[0]
        table_data.append({'text': text, 'bbox': bbox})

# Sort table data based on y-coordinate to maintain row order
    table_data.sort(key=lambda x: ( x['bbox'][0][1],x['bbox'][0][0]))
    #print(table_data)
    # Organize table data into rows and columns
    rows = []
    current_row = []
    prev_y_max = table_data[0]['bbox'][2][1]
    for item in table_data:
        y_min = item['bbox'][0][1]
        y_max = item['bbox'][2][1]
        if y_min > prev_y_max +2:  # Assuming 5 pixels tolerance for row grouping
            current_row.sort(key=lambda x: x['bbox'][0][0])
            #print(f'current row : {current_row}')
            rows.append(current_row)
            current_row = []
            prev_y_max = y_max
        current_row.append(item)
    
    current_row.sort(key=lambda x: x['bbox'][0][0])
    rows.append(current_row)  # Add the last row
    """for x in rows :
        prev=0
        store = ''
        for z in x :
            if prev > z['bbox'][0][0] and prev < z['bbox'][1][0]:
                store += z['text']
                print(store)
                
            else:
                prev = (z['bbox'][0][0] + z['bbox'][1][0])//2
                store = z['text']
            #print(z['bbox'], z['bbox'][0][0] ,z['bbox'][1][0] )"""
    final_row = []
    for x in rows :
        temp=[x[0]['text']]
        prev=0
        store = ''
        
        for y in range(1,len(x)):
            prev_box = x[y-1]
            prev = (prev_box['bbox'][0][0] + prev_box['bbox'][1][0])//2
            if prev > x[y]['bbox'][0][0] and prev < x[y]['bbox'][1][0]:
                store += x[y]['text']
                
                
                
            else:
                if store != '' :
                    temp.pop()
                    temp.append(store)
                
                store = ''
                store = x[y]['text']
                temp.append(store)
                
        final_row.append(temp)
    print(final_row)
    # Create DataFrame
    if 'Sr' in final_row[0]:
        final_row[0].remove('Sr')

    # Create a dictionary from the data
    table_data = dict(zip(final_row[0], final_row[1]))
    #print(table_data)
    # Create DataFrame
    df = pd.DataFrame(final_row)
    df.iloc[0] = df.iloc[0].apply(lambda x: x.lower() if isinstance(x, str) else x)
    print(df)
    def extract_cells_below_header(df, headers):
        cells = []
        header_cols = []
        
        # Find the columns corresponding to the headers
        for header in headers:
            for idx, cell in enumerate(df.iloc[0]):
                if str(header).lower() in str(cell).lower():
                    header_cols.append(idx)
                    break
        
        # Extract cells below each header column
        for header_col in header_cols:
            for cell in df.iloc[1:, header_col]:
                if pd.notna(cell) and str(cell).strip() not in ['nan', 'None', '0']:
                    cells.append(cell)
                else:
                    break  # Stop iterating if encounter null, nan, None, or 0
        return cells

    # Example usage
    # Set of headers to extract cells below
    headers_set = {'description', 'qty', 'amount','bags'}

    # Extract cells below the set of headers
    dic = {}
    for header in headers_set:
        cells = extract_cells_below_header(df, {header})
        dic[header] = cells
    for header, cells in dic.items():
        if header == 'bags' and len(dic['bags']) !=0 and len(dic['qty'])==0:
            dic['qty'] = dic['bags']
        #print(f"Cells below '{header}': {cells}")
    """print("Description of Goods Cells:", description_cells)
    print("Qty Cells:", qty_cells)
    print("Amount Cells:", amount_cells)"""
    if len(dic['amount'])==0  :
        column_index = df.apply(lambda col: col.astype(str).str.contains('amount').idxmax(), axis=1).values[0]
        print(column_index)
        dic['amount'].append(df.iloc[1, column_index-1])
    print(dic)
    return dic
        
    
def ocr(image_path):
    data_dict = yolo_prediction(image_path)
    #print(data_dict)
    ocr = PaddleOCR(show_log=False)
    #reader = easyocr.Reader(['en'])  # You can specify multiple languages if needed
    image = cv2.imread(image_path)
# Load the image
    #image_path = 'path/to/your/image.jpg'
    head_dic,table_data = {},{}
    for label in data_dict:
        x1 = data_dict[label]['x']
        y1 = data_dict[label]['y']
        width = data_dict[label]['width']
        height = data_dict[label]['height']
        x2 = x1 + width
        y2 = y1 + height  # Crop the image based on the bounding box
        cropped_image = image[y1:y2, x1:x2]
        """cv2.imshow('Cropped Image', cropped_image)
        cv2.waitKey(0)  # Wait for a key press to continue
        cv2.destroyAllWindows()"""
        #print(label)
        extracted_text = []
        # paddle ocr
      
        
        if label == 'head':
            result = ocr.ocr(cropped_image, det=True, cls=True)
        #print(result)
            for x in result:
                for line in x:
                    if len(line) > 1 and len(line[1]) > 0:
                        text = line[1][0]
                        extracted_text.append(text)
            final_text =''
            """# Perform OCR on the cropped image using EasyOCR
            result = reader.readtext(cropped_image)

            # Extract text from the OCR result
            final_text = ''
            for detection in result:
                text = detection[1]
                final_text += text + ' '"""
            
            for y in extracted_text:
                final_text += y + ' '
            #print(final_text)
            head_dic = head(final_text)
            """directory = 'nlp/dataset1'
            image_name = os.path.splitext(os.path.basename(image_path))[0]  # Remove extension
            file_name = image_name + '.txt'

            file_path = os.path.join(directory, file_name)
    
    # Open the file in write mode and write the data
            with open(file_path, 'w', encoding='utf-8') as file:
                file.write(final_text)"""
        else :
            original_height, original_width = cropped_image.shape[:2]

# Calculate the new height
            new_height = int(0.6 * original_height)

            # Perform cropping
            cropped_image_0_6 = cropped_image[:new_height, :]

            # Now you can use cropped_image_0_6 for further processing
            result = ocr.ocr(cropped_image_0_6, det=True, cls=True)
            #result = ocr.ocr(cropped_image, det=True, cls=True)
            #print(result)
            table_data =  table(result)
            

            # Display DataFrame
            
            #print(sorted_boxes)
# Initialize variables to store the vertical groups
            
    final_dict = table_data.copy()
    final_dict.update(head_dic)
    #print(final_dict)
    return final_dict
    """
    # Perform OCR on the cropped image using EasyOCR
    result = reader.readtext(cropped_image)

    # Extract text from the OCR result
    total_text = ''
    for detection in result:
        text = detection[1]
        total_text += text + ' '

    # Perform OCR on the image
    result = reader.readtext(image_path)
    total_text = ''
    # Print the extracted text
    for detection in result:
        text = detection[1]
        total_text += text + ' '
    print(total_text)"""


# Create the Gradio interface
demo = gr.Interface(
    process_files,
    inputs=['files'],
    outputs='file',
    title='Invoice Extractor'
)

# Launch the interface
demo.launch()
