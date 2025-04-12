import gradio as gr
import json
import os
from main import run_job_matching
import dotenv
import fitz  # PyMuPDF

# Load environment variables from .env
dotenv.load_dotenv()

def run_job_matching_logic(resume_text: str, resume_file):
    """
    Processes resume, runs job matching, and returns the result string or error.
    """
    try:
        # Prioritize file input if provided
        if resume_file is not None:
            try:
                # --- Debugging Start ---
                print(f"DEBUG: Received file object: {resume_file}")
                if hasattr(resume_file, 'name'):
                    print(f"DEBUG: Received filename: {resume_file.name}")
                    file_extension = os.path.splitext(resume_file.name)[1].lower()
                    print(f"DEBUG: Detected extension: {file_extension}")
                else:
                    print("DEBUG: Uploaded object has no 'name' attribute.")
                    return "Error: Uploaded object is not a valid file."
                # --- Debugging End ---

                # Handle PDF files
                if file_extension == '.pdf':
                    with fitz.open(resume_file.name) as pdf:
                        resume_text = ""
                        for page in pdf:
                            resume_text += page.get_text()
                # Handle text files
                else:
                    with open(resume_file.name, "r") as file:
                        resume_text = file.read()
            except Exception as file_e:
                print(f"DEBUG: Received filename: {resume_file.name}")
                return f"Error reading file: {str(file_e)}"
        # If no file is uploaded, use the text input
        elif not resume_text or not resume_text.strip():
            # Return error if both are empty
            return "Please provide a resume either by pasting text or uploading a file."

        # Run the job-matching with the resume text
        result_object = run_job_matching(resume_text)
        
        # Convert the result to a string
        result_string = str(result_object)
        
        # Return success values
        return result_string

    except Exception as e:
        # Log the exception
        print(f"Error during job matching: {e}")
        return f"An error occurred during processing: {str(e)}"

# Build the Gradio interface using Blocks
with gr.Blocks(theme=gr.themes.Soft()) as iface:
    gr.Markdown("# Job Matching System")
    gr.Markdown("Provide your resume below (paste text or upload a file) to find matching job positions.")

    with gr.Row():  # Main row for side-by-side layout
        with gr.Column(scale=1):  # Left Column for Inputs and Button
            resume_text_input = gr.Textbox(
                lines=15,
                placeholder="Paste your resume here...",
                label="Resume Text"
            )
            resume_file_input = gr.File(
                label="Or upload your resume file (PDF or TXT)"
            )
            submit_btn = gr.Button("Find Matching Jobs", variant="primary")

        with gr.Column(scale=2):  # Right Column for Results
            # We'll use a Textbox for status (showing processing state)
            status_text = gr.Textbox(
                label="Status",
                placeholder="Ready to process resume...",
                interactive=False
            )
            
            # Main results display using Textbox with a box appearance
            results_output = gr.Textbox(
                label="Results", 
                lines=20,
                max_lines=50,
                interactive=False,
                show_copy_button=True
            )
    

    # First click handler to update status text (guaranteed to be fast)
    submit_btn.click(
        fn=lambda: "Processing your resume... Please wait...",
        inputs=None,
        outputs=status_text,
        queue=False  # Don't queue this step
    ).then(  # Then run the actual processing
        fn=run_job_matching_logic,
        inputs=[resume_text_input, resume_file_input],
        outputs=results_output
    ).then(  # Finally update status when done
        fn=lambda: "Processing complete!",
        inputs=None,
        outputs=status_text
    )

# Launch the Gradio app
if __name__ == "__main__":
    iface.launch()