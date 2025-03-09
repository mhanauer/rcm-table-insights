import streamlit as st
import pandas as pd
import numpy as np
import anthropic
from langchain_anthropic import ChatAnthropicMessages
from langchain_community.vectorstores import FAISS
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.schema import SystemMessage, HumanMessage

# Load environment variables
api_key = st.secrets["ANTHROPIC_API_KEY"]

# Initialize the Anthropic Chat Model (Claude)
llm = ChatAnthropicMessages(
    model_name="claude-3-7-sonnet-20250219",  # Using Claude 3.7 Sonnet
    anthropic_api_key=api_key,
    temperature=0.7,
)

# Sample healthcare knowledge base
healthcare_text = """
**Healthcare Analytics and Revenue Cycle Management**

**Key Metrics in Healthcare Analytics:**
- **Total Medical Paid**: Total amount paid for medical services.
- **Total Pharmacy Paid**: Total amount paid for pharmacy services.
- **Total Allowed Amount**: Maximum amount a plan will pay for a covered service.
- **ER Visits**: Number of emergency room visits.
- **Hospital Admissions**: Number of inpatient hospital admissions.
- **Length of Stay**: Average number of days patients stay in the hospital.
- **Readmission Rate**: Percentage of patients who are readmitted within 30 days.

**Key Dimensions for Analysis:**
- **Incurr Date (Service Date)**: When healthcare services were provided.
- **Chronic Conditions**: Including Diabetes, COPD, CHF, Hypertension, etc.
- **Employer Group**: The employer providing the healthcare benefits.
- **Age Group**: Patient age brackets for demographic analysis.
- **Gender**: Patient gender for demographic analysis.
- **Provider Specialty**: Type of healthcare provider (PCP, Specialist, etc.).

**Healthcare Cost Management Strategies:**
- **Care Management Programs**: Target high-risk patients with chronic conditions.
- **Network Optimization**: Steer patients to high-quality, cost-effective providers.
- **Pharmacy Benefit Management**: Implement step therapy and formulary management.
- **ER Avoidance Programs**: Promote urgent care and telehealth alternatives.
- **Value-Based Care Models**: Shift from fee-for-service to outcomes-based payment.

**Challenges in Healthcare Cost Management:**
- Rising specialty drug costs.
- Increasing chronic disease prevalence.
- Inappropriate emergency room utilization.
- Uncoordinated care across providers.
- Lack of price transparency.
"""

# Split the text into manageable chunks
text_splitter = CharacterTextSplitter(
    separator="\n",
    chunk_size=500,
    chunk_overlap=100,
    length_function=len,
)
texts = text_splitter.split_text(healthcare_text)

# Create embeddings for the texts
embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
docsearch = FAISS.from_texts(texts, embeddings)

# Create a retriever
retriever = docsearch.as_retriever()

# Define the custom prompt template
prompt_template = """
You are an expert healthcare analytics assistant helping customers make data-driven decisions.

Based on the selected metrics and dimensions in the table, analyze the data patterns and trends.

First, provide a brief summary of what the data is showing.

Then, your output should include: 
1. A primary area of focus based on the data patterns
2. Three specific initiatives to address the identified focus area
3. Two actionable steps for each initiative

Use the following context to inform your answer:
{context}

Question: {question}

Answer:
"""

PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)

# Create the RetrievalQA chain with the custom prompt
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    chain_type_kwargs={"prompt": PROMPT},
)

# Function to generate fake healthcare data
def generate_fake_healthcare_data(num_samples=100):
    # Define date range for the past 12 months
    end_date = pd.Timestamp.today()
    start_date = end_date - pd.DateOffset(months=12)
    incurr_dates = pd.date_range(start=start_date, end=end_date, periods=num_samples)
    
    # Define employer groups
    employer_groups = ['Acme Corp', 'TechGiant Inc', 'HealthFirst', 'Metro Utilities', 'Education System']
    
    # Create base dataframe
    data = pd.DataFrame({
        'PatientID': np.arange(1, num_samples + 1),
        'IncurrDate': np.random.choice(incurr_dates, num_samples),
        'Diabetes': np.random.choice([True, False], num_samples, p=[0.15, 0.85]),
        'COPD': np.random.choice([True, False], num_samples, p=[0.10, 0.90]),
        'Hypertension': np.random.choice([True, False], num_samples, p=[0.25, 0.75]),
        'AgeGroup': np.random.choice(['18-34', '35-50', '51-64', '65+'], num_samples),
        'Gender': np.random.choice(['Male', 'Female'], num_samples),
        'EmployerGroup': np.random.choice(employer_groups, num_samples)
    })
    
    # Pre-calculate chronic condition counts
    data['DiabetesCount'] = data['Diabetes'].astype(int)
    data['COPDCount'] = data['COPD'].astype(int)
    data['HypertensionCount'] = data['Hypertension'].astype(int)
    
    # Generate metric values with some correlation to conditions
    # Base values
    data['TotalMedicalPaid'] = np.random.normal(1000, 500, num_samples)
    data['TotalPharmacyPaid'] = np.random.normal(300, 150, num_samples)
    data['TotalAllowedAmount'] = np.random.normal(1500, 700, num_samples)
    data['ERVisits'] = np.random.poisson(0.2, num_samples)
    data['HospitalAdmissions'] = np.random.poisson(0.1, num_samples)
    
    # Adjust metrics based on conditions
    # Diabetes increases costs
    data.loc[data['Diabetes'], 'TotalMedicalPaid'] *= np.random.uniform(1.5, 2.5, data['Diabetes'].sum())
    data.loc[data['Diabetes'], 'TotalPharmacyPaid'] *= np.random.uniform(2.0, 3.0, data['Diabetes'].sum())
    
    # COPD increases ER visits and hospitalizations
    data.loc[data['COPD'], 'ERVisits'] += np.random.poisson(1, data['COPD'].sum())
    data.loc[data['COPD'], 'HospitalAdmissions'] += np.random.poisson(0.5, data['COPD'].sum())
    
    # Hypertension increases pharmacy costs
    data.loc[data['Hypertension'], 'TotalPharmacyPaid'] *= np.random.uniform(1.3, 2.0, data['Hypertension'].sum())
    
    # Age affects costs
    data.loc[data['AgeGroup'] == '65+', 'TotalMedicalPaid'] *= np.random.uniform(1.5, 2.0, (data['AgeGroup'] == '65+').sum())
    data.loc[data['AgeGroup'] == '51-64', 'TotalMedicalPaid'] *= np.random.uniform(1.3, 1.7, (data['AgeGroup'] == '51-64').sum())
    
    # Ensure all metrics are positive and round appropriately
    data['TotalMedicalPaid'] = np.round(np.maximum(data['TotalMedicalPaid'], 0), 2)
    data['TotalPharmacyPaid'] = np.round(np.maximum(data['TotalPharmacyPaid'], 0), 2)
    data['TotalAllowedAmount'] = np.round(data['TotalMedicalPaid'] + data['TotalPharmacyPaid'] + np.random.normal(100, 50, num_samples), 2)
    data['TotalAllowedAmount'] = np.maximum(data['TotalAllowedAmount'], data['TotalMedicalPaid'] + data['TotalPharmacyPaid'])
    
    # Ensure whole numbers for visit counts
    data['ERVisits'] = data['ERVisits'].astype(int)
    data['HospitalAdmissions'] = data['HospitalAdmissions'].astype(int)
    
    return data

# Function to aggregate data based on selected dimensions and metrics
def aggregate_data(data, dimensions, metrics):
    # Group by selected dimensions
    if dimensions:
        grouped = data.groupby(dimensions)
        agg_dict = {metric: 'sum' for metric in metrics}
        
        # Include count of patients
        agg_dict['PatientID'] = 'count'
        
        # Aggregate the data
        result = grouped.agg(agg_dict).reset_index()
        
        # Rename the count column
        result = result.rename(columns={'PatientID': 'PatientCount'})
        
        return result
    else:
        # If no dimensions selected, just sum up the metrics
        result = pd.DataFrame({metric: [data[metric].sum()] for metric in metrics})
        result['PatientCount'] = len(data)
        return result

# Function to generate insights based on table data
def generate_insights(data, selected_metrics, selected_dimensions, additional_context=""):
    # Convert the dataframe to a string representation for the prompt
    data_str = data.to_string()
    
    # Create a question based on the selected metrics and dimensions
    if additional_context:
        question = f"""
        Analyze the healthcare data table showing {', '.join(selected_metrics)} by {', '.join(selected_dimensions)}.
        
        Here's the data:
        {data_str}
        
        Additional context provided by user:
        {additional_context}
        
        What patterns or trends do you observe and what strategies would you recommend?
        """
    else:
        question = f"""
        Analyze the healthcare data table showing {', '.join(selected_metrics)} by {', '.join(selected_dimensions)}.
        
        Here's the data:
        {data_str}
        
        What patterns or trends do you observe and what strategies would you recommend?
        """
    
    # Get insights using the QA chain
    response = qa_chain.run(question)
    return response

# Streamlit application
st.title("Healthcare Analytics Dashboard")
st.write("Analyze healthcare metrics across different dimensions")

# Generate the fake data
data = generate_fake_healthcare_data(1000)

# Define available metrics and dimensions
available_metrics = [
    'TotalMedicalPaid', 
    'TotalPharmacyPaid', 
    'TotalAllowedAmount', 
    'ERVisits', 
    'HospitalAdmissions',
    'DiabetesCount',
    'COPDCount',
    'HypertensionCount'
]

available_dimensions = [
    'IncurrDate',
    'AgeGroup', 
    'Gender', 
    'EmployerGroup'
]

# Sidebar for selections
st.sidebar.header("Select Data to Analyze")

# Select metrics (default to medical paid, ER visits, and diabetes count)
default_metrics = ['TotalMedicalPaid', 'ERVisits', 'DiabetesCount']
selected_metrics = st.sidebar.multiselect(
    "Select Metrics",
    available_metrics,
    default=default_metrics
)

# Select dimensions (default to employer group)
default_dimensions = ['EmployerGroup']
selected_dimensions = st.sidebar.multiselect(
    "Select Dimensions",
    available_dimensions,
    default=default_dimensions
)

# Button to apply selections
apply_button = st.sidebar.button("Apply Selections")

# Show data sample
if st.checkbox("Show Raw Data Sample", value=False):
    st.write("Sample of raw data (first 10 rows):")
    st.write(data.head(10))

# Process data based on selections
if selected_metrics and apply_button:
    # Special handling for date dimension
    if 'IncurrDate' in selected_dimensions:
        # Group by month for better visualization
        data['Month'] = data['IncurrDate'].dt.to_period('M')
        selected_dimensions[selected_dimensions.index('IncurrDate')] = 'Month'
    
    # No need to convert chronic conditions as they're now pre-calculated as count metrics
    
    # Aggregate the data
    aggregated_data = aggregate_data(data, selected_dimensions, selected_metrics)
    
    # Display the aggregated data
    st.header("Metrics by Dimensions")
    st.write(aggregated_data)
    
    # Generate insights
    st.header("Automated Playbook")
    
    # Additional context input for customizing the prompt
    st.subheader("Additional Context (Optional)")
    additional_context = st.text_area(
        "Add information about your hospital system, insurance company, or specific programs to better tailor recommendations:",
        height=100
    )
    
    with st.spinner("Generating insights..."):
        # Include additional context in the prompt if provided
        if additional_context:
            custom_question = f"""
            Analyze the healthcare data table showing {', '.join(selected_metrics)} by {', '.join(selected_dimensions)}.
            
            Here's the data:
            {aggregated_data.to_string()}
            
            Additional context provided by user:
            {additional_context}
            
            What patterns or trends do you observe and what strategies would you recommend?
            """
        else:
            custom_question = f"""
            Analyze the healthcare data table showing {', '.join(selected_metrics)} by {', '.join(selected_dimensions)}.
            
            Here's the data:
            {aggregated_data.to_string()}
            
            What patterns or trends do you observe and what strategies would you recommend?
            """
            
        insights = generate_insights(aggregated_data, selected_metrics, selected_dimensions, additional_context)
    st.write(insights)
elif not apply_button:
    st.info("Select metrics and conditions, then click 'Apply Selections'.")
else:
    st.warning("Please select at least one metric.")

# Custom question section
st.header("Ask a Custom Question")
custom_question = st.text_input("Enter your question about the healthcare data:")

if custom_question:
    with st.spinner("Generating response..."):
        response = qa_chain.run(custom_question)
    st.write(response)
