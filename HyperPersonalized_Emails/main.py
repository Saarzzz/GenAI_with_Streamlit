import os
import re
import json
import streamlit as st
import pandas as pd
from io import BytesIO
from anthropic import Anthropic
from dotenv import load_dotenv
from pydantic import BaseModel, Field, ValidationError, create_model
from langchain_core.output_parsers import JsonOutputParser
from time import sleep

# Set the page configuration to wide mode
st.set_page_config(layout="wide",page_icon=":rocket:",page_title="Excel/CSV Column Manager and Hyper Personalized Email Generator")

load_dotenv()

client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

st.title(":blue[Hyper Personalized Email Generator ]:rocket:")

# Sidebar for file upload
uploaded_file = st.sidebar.file_uploader('Upload an Excel or CSV file', type=['xlsx', 'csv'])

if uploaded_file:
    # Determine the file type
    file_type = uploaded_file.name.split('.')[-1]

    # Radio buttons for selecting the operation
    operation = st.sidebar.radio("Select the operation:", ("Generate Hyperpersonalized Emails", "Delete Columns from Excel/CSV Files"))

    if operation == "Delete Columns from Excel/CSV Files":
        if file_type == 'xlsx':
            # Read the uploaded Excel file
            excel_data = pd.ExcelFile(uploaded_file)

            # List all the sheet names
            sheet_names = excel_data.sheet_names

            # Select sheet
            selected_sheet = st.selectbox('Select a sheet', sheet_names)
            if selected_sheet:
                df = pd.read_excel(uploaded_file, sheet_name=selected_sheet)
                columns = df.columns.tolist()
                
                # Multiselect for columns to delete
                columns_to_delete = st.multiselect('Select columns to delete', columns)
                
                if st.button('Delete Selected Columns'):
                    if columns_to_delete:
                        df.drop(columns=columns_to_delete, inplace=True)
                        st.write('Updated DataFrame:', df)
                        
                        # Convert the updated DataFrame to a downloadable Excel file
                        output = BytesIO()
                        with pd.ExcelWriter(output, engine='openpyxl') as writer:
                            df.to_excel(writer, index=False, sheet_name=selected_sheet)
                        output.seek(0)
                        
                        # Provide download link
                        st.download_button(label='Download Updated Excel', data=output, file_name='updated_excel_file.xlsx', mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
                    else:
                        st.warning('No columns selected for deletion')

        elif file_type == 'csv':
            # Read the uploaded CSV file
            df = pd.read_csv(uploaded_file)
            columns = df.columns.tolist()

            # Multiselect for columns to delete
            columns_to_delete = st.multiselect('Select columns to delete', columns)
            
            if st.button('Delete Selected Columns'):
                if columns_to_delete:
                    df.drop(columns=columns_to_delete, inplace=True)
                    st.write('Updated DataFrame:', df)
                    
                    # Convert the updated DataFrame to a downloadable CSV file
                    output = BytesIO()
                    df.to_csv(output, index=False)
                    output.seek(0)
                    
                    # Provide download link
                    st.download_button(label='Download Updated CSV', data=output, file_name='updated_csv_file.csv', mime='text/csv')
                else:
                    st.warning('No columns selected for deletion')

    elif operation == "Generate Hyperpersonalized Emails":
        # Hyper Personalized Email Generator Section
        st.subheader("Hyper Personalized Email Generator")

        # Input field for the number of emails
        num_emails = st.number_input("Number of Emails", min_value=1, max_value=10, value=5)

        try:
            if file_type == 'csv':
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)

            if df.empty:
                st.write("The uploaded file is empty. Please upload a file with data.")
            else:
                # Add email and subject columns if they don't exist
                email_columns = [f"Email {i+1}" for i in range(num_emails)]
                subject_columns = [f"Subject {i+1}" for i in range(num_emails)]
                for col in email_columns + subject_columns:
                    if col not in df.columns:
                        df[col] = ""
                st.session_state['df'] = df  # Save the DataFrame to session state

        except pd.errors.EmptyDataError:
            st.error("The uploaded file is empty or not properly formatted. Please upload a valid file.")

        def preprocess_json_string(json_string):
            # Remove invalid control characters
            json_string = re.sub(r'[\x00-\x1f\x7f]', '', json_string)
            # Escape backslashes
            json_string = json_string.replace('\\', '\\\\')
            return json_string

        email_fields = {f'email{i+1}': (str, Field(description=f"Email {i+1}")) for i in range(num_emails)}
        Email = create_model('Email', **email_fields)

        class EmailResponse(BaseModel):
            emails: list[Email]
            subjects: list[str]

        parser = JsonOutputParser(pydantic_object=EmailResponse)
        format_instructions = parser.get_format_instructions()

        # Initialize session state for inputs if not already present
        for i in range(1, num_emails + 1):
            if f'Input_E{i}' not in st.session_state:
                st.session_state[f'Input_E{i}'] = ""
            if f'Format_E{i}' not in st.session_state:
                st.session_state[f'Format_E{i}'] = ""
            if 'user_details' not in st.session_state:
                st.session_state['user_details'] = (
                    "Name: \n"
                    "Designation: \n"
                    "Company: \n"
                    "Service Offering: \n"
                    "Reason for Outreach: \n"
                )

        # Create two columns for layout
        col0, col2 = st.columns([3, 3], gap="large")

        # Collect inputs in the second column (adjacent to the sidebar)
        with col0:
            st.subheader("VF Market Plan Inputs")
            for i in range(1, num_emails + 1):
                with st.expander(f"Email {i}"):
                    st.session_state[f'Input_E{i}'] = st.text_area(f"Enter VF Market Plan for Email {i}", st.session_state[f'Input_E{i}'], height=None)
                    st.session_state[f'Format_E{i}'] = st.text_area(f"Enter Format for Email {i}", st.session_state[f'Format_E{i}'], height=None)

        # Collect user details in the third column
        with col2:
            st.subheader("User Details")
            st.session_state['user_details'] = st.text_area("Enter User Details (Name, Designation, Company, Service Offering, Reason for Outreach)", st.session_state['user_details'], height=None)

        if st.button("Generate Emails"):
            if all([st.session_state[f'Input_E{i}'] for i in range(1, num_emails + 1)]) and st.session_state['user_details']:
                progress_bar = st.progress(0)
                total_rows = len(st.session_state['df'])

                try:
                    for index, row in st.session_state['df'].iterrows():
                        prompt = f"""
                        You are tasked with generating {num_emails} hyper-personalized cold email outreach messages based on a LinkedIn profile and other provided information. Follow these instructions carefully to create effective, tailored emails.

                        First, you will be provided with the following information:

                        Name: {row['first_name']} {row['last_name']}
                        Title: {row['headline']}
                        Company: {row['current_company']}
                        Location: {row['location_name']}
                        Summary: {row['summary']}
                        Designation: {row['current_company_position']}
                        Company: {row['organization_description_1']}
                        Skills: {row['skills']}

                        {st.session_state['user_details']}

                        Use the Mini Brief Market Plan given for Each Email.
                        """

                        for i in range(1, num_emails + 1):
                            prompt += f"""
                            For Email {i}:
                            {st.session_state[f'Input_E{i}']}
                            Format: {st.session_state[f'Format_E{i}']}
                            """

                        prompt += f"""
                        To generate each email, always use this formula:
                        1. Find a unique opening line tied to the prospect from the given information.
                        2. Introduce your company's relevant specialty.
                        3. Show understanding of prospect's world.
                        4. Explain 3 key benefits of your offering from the case studies.
                        5. Share a bit about your process and speed.
                        6. Provide a customer example if possible.
                        7. End with a clear meeting request.
                        8. Close with full signature.

                        Additional guidelines:
                        - Each email should be approximately 200 words long.
                        - Ensure each email is unique and tailored to the specific Mini Brief Market Plan provided for that email.
                        - Use the prospect's information to personalize the content and make it relevant to their role and company.

                        JSON Format for the output will be:-
                        {format_instructions}.
                        Ensure that your output contains only the JSON structure with the generated emails. Do not include any additional text or explanations outside of the JSON format.
                        """

                        response = client.messages.create(
                            model="claude-3-haiku-20240307",
                            max_tokens=3096,
                            temperature=0.4,
                            messages=[{"role": "user", "content": prompt}]
                        )

                        response_text = response.content[0].text
                        emails_json = preprocess_json_string(response_text)

                        try:
                            emails_json = json.loads(emails_json)
                            for email_dict in emails_json['emails']:
                                for key, value in email_dict.items():
                                    if key.startswith('email') and value:
                                        email_index = int(key.replace('email', '')) - 1
                                        st.session_state['df'].at[index, f"Email {email_index + 1}"] = value
                            for i, subject in enumerate(emails_json.get('subjects', [])):
                                st.session_state['df'].at[index, f"Subject {i + 1}"] = subject
                        except (json.JSONDecodeError, ValidationError) as e:
                            print(f"Error parsing JSON or validation error for row {index}: {e}")
                            st.error(f"Error parsing JSON or validation error for row {index}: {e}")
                            continue  # Skip to the next row if there's an error

                        progress_bar.progress((index + 1) / total_rows)
                        sleep(0.1)  # Simulate a delay for demonstration purposes

                    st.write("Emails generated successfully!")
                except Exception as e:
                    st.error(f"An error occurred: {e}")

                st.dataframe(st.session_state['df'])

                csv = st.session_state['df'].to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name='output.csv',
                    mime='text/csv'
                )
            else:
                st.error("Please fill in all the input fields.")
else:
    st.write("Please upload a file to get started.")
