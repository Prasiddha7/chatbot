
## Setup

1. **Google API Key:**
   - Obtain a Google API key from the [Google Cloud Console](https://console.cloud.google.com/).
   - Save the API key in a file named `.env` in the project directory:
     ```
     GOOGLE_API_KEY=your_api_key_here
     ```

2. **Prepare Documents:**
   - Ensure you have the documents (PDF or TXT) from which you want the chatbot to extract information.

## Usage

1. **Run the Application:**
   - Open a terminal or command prompt.
   - Navigate to the directory containing the Python files (`main.py` and other related files).
   - Run the `main.py` file using Python:
     ```
     python main.py
     ```

2. **Interact with the Chatbot:**
   - Follow the on-screen instructions to interact with the chatbot.
   - Ask questions, and the chatbot will search through the provided documents to provide relevant answers.
   - To exit the application, type 'exit' when prompted.

3. **Providing Input Documents:**
   - If you want to use your own documents instead of the default ones provided in the code, update the file paths in the `get_pdf_raw_text()` and `get_text_file_raw_text()` functions accordingly.