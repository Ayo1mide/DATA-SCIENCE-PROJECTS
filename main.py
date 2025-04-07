from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel, Field
from typing import Literal
import joblib
import pandas as pd
from io import StringIO, BytesIO
import os
import logging
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet

# Set up logging
logger = logging.getLogger("uvicorn")
logging.basicConfig(level=logging.INFO)

# Initialize the FastAPI app with a file size limit
app = FastAPI(max_upload_size=100 * 1024 * 1024)  # 100MB limit

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
) 

# Load the trained model and preprocessor
model_path = "decision_tree_model.joblib"
preprocessor_path = "column_transformer.joblib"

try:
    model = joblib.load(model_path)
    preprocessor = joblib.load(preprocessor_path)
except Exception as e:
    model = None
    preprocessor = None
    logger.error(f"Error loading model or preprocessor: {e}")

# Serve the static HTML file and any other assets
app.mount("/static", StaticFiles(directory=".", html=True), name="static")

# Define the structure of the input data using Pydantic
class PredictionInput(BaseModel):
    gender: Literal['Male', 'Female'] = Field(..., example="Male")
    age_in_years: int = Field(..., example=30)
    type_of_testing: Literal[
        'Oral Fluid (Drugs Only)', 'Oral Fluid (Back to Lab & Breath)', 'Oral Fluid (Back to Lab Only)', 
        'Urine (Drugs Only)', 'Breath Only', 'Oral Fluid (Drugs & Breath)', 'Urine (Back to Lab & Breath)', 
        'Urine (with Alcohol)', 'Urine (Drugs & Breath)', 'Urine (Back to Lab Only)'
    ] = Field(..., example="Oral Fluid (Drugs Only)")
    declared_medication: Literal['yes', 'no'] = Field(..., example="no")
    alcohol_result: Literal['Pass', 'Fail'] = Field(..., example="Pass")
    breath_test_result: float = Field(..., example=0.0000)
    company_cut_off_level: float = Field(..., example=0.35)
    drug_result: Literal['Negative', 'Non-Negative'] = Field(..., example="Non-Negative")
    drugs_non_negative: Literal[
        'None', 'Methadone', 'Cocaine', 'Other', 'Amphetamines', 'Methamphetamine', 'Ecstasy', 'Cannabis', 
        'Barbiturates', 'Benzodiazepine', 'Tri-cyclic Antidepressants', 'Opiates', 'Tramadol'
    ] = Field(..., example="Methadone")
    year: int = Field(..., example=2023)
    month: int = Field(..., example=5)
    day_of_week: int = Field(..., example=2)
    weekOfYear: int = Field(..., example=20)
    hour: int = Field(..., example=14)
    minutes: int = Field(..., example=30)
    seconds: int = Field(..., example=45)

# Serve the index.html file at the root URL
@app.get("/", response_class=HTMLResponse)
async def get_index():
    if os.path.exists("index.html"):
        with open("index.html", "r") as file:
            html_content = file.read()
        return HTMLResponse(content=html_content)
    else:
        raise HTTPException(status_code=404, detail="HTML file not found")

@app.post("/predict/")
def predict(input_data: PredictionInput):
    # Ensure model and preprocessor are loaded
    if model is None or preprocessor is None:
        raise HTTPException(status_code=500, detail="Model or preprocessor is not loaded.")
    
    try:
        # Convert input data to a DataFrame
        input_dict = input_data.dict() 
        input_df = pd.DataFrame([input_dict])

        # Process the data using the ColumnTransformer
        try:
            processed_features = preprocessor.transform(input_df)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Error during preprocessing: {str(e)}")

        # Predict using the trained model
        try:
            prediction = model.predict(processed_features)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error during model prediction: {str(e)}")

        # Return the prediction as a list
        return {"result": prediction.tolist()}

    except HTTPException as http_error:
        raise http_error
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")

# Generate a PDF report
def generate_report(prediction: str, input_data: dict) -> BytesIO:
    pdf_buffer = BytesIO()
    doc = SimpleDocTemplate(pdf_buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    title_style = styles["Title"]
    normal_style = styles["BodyText"]
    title = Paragraph("Lab Result Report", title_style)

    table_data = [["Field", "Value"]]
    for key, value in input_data.items():
        table_data.append([key, str(value)])

    table = Table(table_data, colWidths=[200, 300])
    table.setStyle(
        TableStyle([ 
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ])
    )

    if prediction == "positive":
        advice_text = (
            "Based on the test results, a positive result has been recorded. It is important to take the following steps to prevent further complications and protect your health:\n\n"
            "1. Consult with a Healthcare Provider: Schedule an appointment with your doctor to discuss the results, determine any necessary treatments, and follow their recommendations closely.\n\n"
            "2. Adhere to Prescribed Medications or Treatments: If a medication or treatment plan has been prescribed, follow it as directed to manage or mitigate the condition effectively.\n\n"
            "3. Practice Preventive Measures:\n"
            "- Hygiene: Wash your hands regularly, avoid touching your face, and use hand sanitizer frequently.\n"
            "- Social Distancing or Quarantine: If the result relates to a communicable disease, stay at home and avoid contact with others as advised by your healthcare provider.\n"
            "- Vaccination: In some cases, vaccination can reduce the risk of transmission or recurrence. Speak to your healthcare provider about whether this is an option for you.\n"
            "- Healthy Lifestyle: Maintain a healthy diet, exercise regularly, and manage stress to boost your immune system and overall well-being.\n\n"
            "4. Monitor Symptoms: Pay attention to any changes in your health and seek medical attention if symptoms worsen or new symptoms arise.\n\n"
            "5. Educate Yourself: Understanding your condition and how it can spread or progress is key to preventing future risks. Keep up with reliable sources of information and recommendations from healthcare professionals."
        )
    else:
        advice_text = (
            "The test result is negative, indicating no immediate concerns. However, if symptoms persist or new symptoms arise, it is advisable to seek medical attention. "
            "Regular health check-ups are recommended to monitor your health. Should you have any questions, please contact your healthcare provider for further guidance."
        )
    advice = Paragraph(advice_text, normal_style)
    elements = [title, table, advice]
    doc.build(elements)
    pdf_buffer.seek(0)
    return pdf_buffer

@app.post("/generate_report/") 
def generate_report_endpoint(input_data: PredictionInput):
    if model is None or preprocessor is None:
        raise HTTPException(status_code=500, detail="Model or preprocessor is not loaded.")
    
    try:
        input_dict = input_data.dict()
        input_df = pd.DataFrame([input_dict])
        processed_features = preprocessor.transform(input_df)
        prediction = model.predict(processed_features)[0]

        pdf_buffer = generate_report(prediction, input_dict)
        return StreamingResponse(pdf_buffer, media_type="application/pdf", headers={ 
            "Content-Disposition": "attachment; filename=lab_result_report.pdf"
        })
    except Exception as e:
        logger.error(f"Error generating report: {e}")
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")

@app.post("/upload_batch")
async def upload_batch(file: UploadFile = File(...)):
    if model is None or preprocessor is None:
        raise HTTPException(status_code=500, detail="Model or preprocessor is not loaded.")
    
    try:
        # Read and decode the uploaded file
        contents = await file.read()
        csv_data = StringIO(contents.decode('utf-8'))
        df = pd.read_csv(csv_data)

        # Validate that required columns exist
        required_columns = [
            'gender', 'age_in_years', 'type_of_testing', 'declared_medication', 
            'alcohol_result', 'breath_test_result', 'company_cut_off_level', 
            'drug_result', 'drugs_non_negative', 'year', 'month', 
            'day_of_week', 'weekOfYear', 'hour', 'minutes', 'seconds'
        ]
        for col in required_columns:
            if col not in df.columns:
                raise HTTPException(status_code=400, detail=f"Missing required column: {col}")
        
        # Replace None values only in 'drugs_non_negative' column with 'None'
        if 'drugs_non_negative' in df.columns:
            df['drugs_non_negative'] = df['drugs_non_negative'].fillna('None')
        
        # Preprocess other columns for consistency
        df.fillna({
            'gender': 'Unknown',
            'type_of_testing': 'Unknown',
            'declared_medication': 'no',
            'alcohol_result': 'Pass',
        }, inplace=True)

        # Replace numeric NaN with 0
        numeric_columns = ['age_in_years', 'breath_test_result', 'company_cut_off_level', 'year', 'month', 
                           'day_of_week', 'weekOfYear', 'hour', 'minutes', 'seconds']
        df[numeric_columns] = df[numeric_columns].fillna(0)

        # Process predictions
        predictions = []
        for _, row in df.iterrows():
            input_data = row.to_dict()
            input_df = pd.DataFrame([input_data])
            processed_features = preprocessor.transform(input_df)
            prediction = model.predict(processed_features)[0]
            predictions.append(prediction)

        # Add predictions to the DataFrame
        df['lab_result'] = predictions

        # Return the updated CSV
        output = StringIO()
        df.to_csv(output, index=False)
        output.seek(0)
        return StreamingResponse(output, media_type="text/csv", headers={ 
            "Content-Disposition": f"attachment; filename={file.filename}"
        })

    except Exception as e:
        logger.error(f"Error processing CSV: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing CSV: {str(e)}")
