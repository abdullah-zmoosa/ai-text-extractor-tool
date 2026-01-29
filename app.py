import streamlit as st
import os
import re
import csv
import time
import glob
import tempfile
import pandas as pd
from google import genai
from google.genai import types
from dotenv import load_dotenv
import json
import zipfile
import io

# Page config
st.set_page_config(page_title="AI Text Extractor", page_icon="-📝-")

def get_label_from_filename(filename):
    return os.path.splitext(filename)[0]

def main():
    st.title("AI Text Extractor")
    st.markdown("Extract text from images to CSV using Gemma 3 27b.")

    # Load environment variables
    load_dotenv()
    env_api_key = os.getenv("GEMINI_API_KEY", "")

    with st.sidebar:
        st.subheader("Settings")
        with st.expander("API & Model Config", expanded=True):
            api_key = st.text_input("Enter your Gemini API Key", value=env_api_key, type="password")
            save_key = st.checkbox("Save API Key locally", value=False)
            
            # Hardcoded model
            model_name = "gemma-3-27b-it" 
            st.info(f"Using Model: {model_name}")
        
        project_name = st.text_input("Project Name", value="extraction_results")
        
        # Prompt Templates
        st.subheader("Instructions")
        prompt_templates = {
            "Default (Extract Text)": "",
            "Translate to English": "Translate the extracted text to English.",
            "Format as JSON": "Format the output as valid JSON.",
            "Summarize Content": "Provide a summary of the extracted text."
        }
        selected_template = st.selectbox("Quick Template", list(prompt_templates.keys()))
        
        custom_prompt = st.text_area(
            "Custom Instructions", 
            value=prompt_templates[selected_template], 
            help="Add specific instructions like 'Translate to English' or 'Format as JSON'."
        )

    uploaded_files = st.file_uploader("Upload Images", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
    
    if st.button("Start Extraction"):
        if not api_key:
            st.error("Please provide an API Key.")
            return
        if not uploaded_files:
            st.error("Please upload some images.")
            return

        if save_key and api_key:
            with open(".env", "w") as f:
                f.write(f"GEMINI_API_KEY={api_key}")
            st.sidebar.success("API Key saved to .env file!")

        client = genai.Client(api_key=api_key)
        
        # Temp directory for processing
        with tempfile.TemporaryDirectory() as tmp_dir:
            csv_filename = f"{project_name}.csv" if project_name else "extracted_text.csv"
            csv_path = os.path.join(tmp_dir, csv_filename)
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            results = []
            last_call_time = 0
            
            for i, uploaded_file in enumerate(uploaded_files):
                # Continuous Rate Limiting
                # limit is 20 RPM -> 1 request every 3 seconds. We use 4s to be safe.
                current_time = time.time()
                time_since_last = current_time - last_call_time
                if time_since_last < 4:
                    wait_time = 4 - time_since_last
                    status_text.text(f"Throttling for {wait_time:.1f}s to respect rate limits...")
                    time.sleep(wait_time)
                
                last_call_time = time.time()
                
                filename = uploaded_file.name
                label = get_label_from_filename(filename)
                status_text.text(f"Processing {i+1}/{len(uploaded_files)}: {label}...")
                
                try:
                    # Extract Text
                    image_bytes = uploaded_file.read()
                    # Revised Prompt Logic (Multimodal Categorization + Extraction)
                    base_prompt = """Analyze this image and:
1. Categorize it (e.g., Receipt, Business Card, Handwritten Note, Invoice, or Other).
2. Extract all text exactly as it appears.

Format your response exactly as:
Category: [Type]
Text: [Extracted Content]"""
                    final_prompt = f"{base_prompt}\n\nUser Instructions: {custom_prompt}" if custom_prompt else base_prompt
                    
                    response = client.models.generate_content(
                        model=model_name,
                        contents=[
                            types.Content(
                                parts=[
                                    types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"),
                                    types.Part(text=final_prompt)
                                ]
                            )
                        ]
                    )
                    response_text = response.text.strip()
                    
                    # Parse Category and Text
                    category = "Other"
                    extracted_text = response_text
                    
                    if "Category:" in response_text and "Text:" in response_text:
                        try:
                            parts = response_text.split("Text:", 1)
                            cat_part = parts[0].replace("Category:", "").strip()
                            text_part = parts[1].strip()
                            category = cat_part
                            extracted_text = text_part
                        except:
                            pass

                    results.append([filename, label, category, extracted_text])

                    # Live Preview
                    with st.expander(f"✅ Result: {label} ({category})", expanded=True):
                        col_img, col_text = st.columns([1, 2])
                        with col_img:
                            st.image(image_bytes, caption=filename, width="stretch")
                        with col_text:
                            st.text_area("Extracted", value=extracted_text, height=200, key=f"text_{i}")
                        
                except Exception as e:
                    st.error(f"Error processing {filename}: {e}")
                
                progress_bar.progress((i + 1) / len(uploaded_files))
                time.sleep(1) # Small delay

            status_text.success("Processing Complete!")
            
            # Create DataFrame
            df_full = pd.DataFrame(results, columns=["Filename", "Label", "Category", "Extracted Text"])
            
            # Display Editable Data Table
            st.subheader("Extracted Data (Editable)")
            st.info("💡 You can click on cells to edit and correct any errors before downloading.")
            edited_df = st.data_editor(df_full, width="stretch", num_rows="dynamic")

            st.subheader("Download Results")
            col1, col2, col3 = st.columns(3)

            # 1. CSV Download
            csv = edited_df.to_csv(index=False).encode('utf-8')
            col1.download_button(
                label="📄 Download CSV",
                data=csv,
                file_name=csv_filename,
                mime="text/csv",
            )

            # 2. JSON Download
            json_str = edited_df.to_json(orient="records", indent=2)
            col2.download_button(
                label="​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​{​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​} Download JSON",
                data=json_str,
                file_name=f"{project_name}.json" if project_name else "extracted_text.json",
                mime="application/json",
            )

            # 3. ZIP of Text Files Download
            # Create a zip file in memory
            zip_buffer = io.BytesIO()
            with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
                for idx, row in edited_df.iterrows():
                    # Create a safe filename for inside the zip
                    safe_name = f"{row['Label']}.txt"
                    zf.writestr(safe_name, row['Extracted Text'])
            
            col3.download_button(
                label="📦 Download ZIP (Txt)",
                data=zip_buffer.getvalue(),
                file_name=f"{project_name}_txts.zip" if project_name else "extracted_text_files.zip",
                mime="application/zip",
            )

if __name__ == "__main__":
    main()
