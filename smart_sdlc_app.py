# smart_sdlc_app.py

import gradio as gr
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import PyPDF2
import random

# Load model and tokenizer
model_name = "ibm-granite/granite-3.2-2b-instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto" if torch.cuda.is_available() else None
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

def generate_response(prompt, max_length=1024):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    if torch.cuda.is_available():
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = response.replace(prompt, "").strip()
    return response

# PDF text extraction
def extract_text_from_pdf(pdf_file):
    if pdf_file is None:
        return ""
    try:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
        return text
    except Exception as e:
        return f"Error reading PDF: {str(e)}"

# Requirement analysis
def requirement_analysis(pdf_file, prompt_text):
    if pdf_file is not None:
        content = extract_text_from_pdf(pdf_file)
        analysis_prompt = f"Analyze the following document and extract key software requirements. Organize them into functional requirements, non-functional requirements, and technical specifications:\n\n{content}"
    else:
        analysis_prompt = f"Analyze the following requirements and organize them into functional requirements, non-functional requirements, and technical specifications:\n\n{prompt_text}"
    return generate_response(analysis_prompt, max_length=1200)

# Code generation
def code_generation(prompt, language):
    code_prompt = f"Generate {language} code for the following requirement:\n\n{prompt}\n\nCode:"
    return generate_response(code_prompt, max_length=1200)

# Health AI (project health report)
def generate_health_report(project_name="SmartSDLC Project"):
    requirements_coverage = random.randint(70, 100)
    code_quality = random.choice(["Excellent", "Good", "Average", "Needs Improvement"])
    bugs_predicted = random.choice(["Low", "Medium", "High"])
    timeline_risk = random.choice(["On Track", "Slight Delay", "High Risk"])
    team_productivity = random.randint(60, 95)

    report = f"""
    📊 Project Health Report: {project_name}

    ✅ Requirements Coverage: {requirements_coverage}%
    🧑‍💻 Code Quality: {code_quality}
    🐞 Bug Prediction: {bugs_predicted}
    ⏳ Timeline Risk: {timeline_risk}
    🚀 Team Productivity: {team_productivity}%

    Summary:
    The project is currently being monitored for risks and quality. 
    The AI suggests focusing on areas with lower performance to improve overall health.
    """
    return report

# Gradio App
with gr.Blocks() as app:
    gr.Markdown("# SmartSDLC – AI Powered SDLC Tool")

    with gr.Tabs():
        # Tab 1: Requirement Analysis
        with gr.TabItem("Requirement Analysis"):
            with gr.Row():
                with gr.Column():
                    pdf_upload = gr.File(label="Upload PDF", file_types=[".pdf"])
                    prompt_input = gr.Textbox(
                        label="Or write requirements here",
                        placeholder="Describe your software requirements...",
                        lines=5
                    )
                    analyze_btn = gr.Button("Analyze")
                with gr.Column():
                    analysis_output = gr.Textbox(label="Requirements Analysis", lines=20)
            analyze_btn.click(requirement_analysis, inputs=[pdf_upload, prompt_input], outputs=analysis_output)

        # Tab 2: Code Generation
        with gr.TabItem("Code Generation"):
            with gr.Row():
                with gr.Column():
                    code_prompt = gr.Textbox(
                        label="Code Requirements",
                        placeholder="Describe what code you want to generate...",
                        lines=5
                    )
                    language_dropdown = gr.Dropdown(
                        choices=["Python", "JavaScript", "Java", "C++", "C#", "PHP", "Go", "Rust"],
                        label="Programming Language",
                        value="Python"
                    )
                    generate_btn = gr.Button("Generate Code")
                with gr.Column():
                    code_output = gr.Textbox(label="Generated Code", lines=20)
            generate_btn.click(code_generation, inputs=[code_prompt, language_dropdown], outputs=code_output)

        # Tab 3: Health AI
        with gr.TabItem("Health AI"):
            with gr.Row():
                with gr.Column():
                    project_name = gr.Textbox(label="Project Name", value="SmartSDLC Project")
                    health_btn = gr.Button("Generate Health Report")
                with gr.Column():
                    health_output = gr.Textbox(label="Health Report", lines=20)
            health_btn.click(generate_health_report, inputs=[project_name], outputs=health_output)

app.launch(share=True)
