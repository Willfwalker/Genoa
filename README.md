1. Clone this repository
2. Create a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```
3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
4. Create a `.env` file with your API keys:
   ```
   CANVAS_URL=your_canvas_url
   Canvas_API_KEY=your_canvas_api_key
   GOOGLE_API_KEY=your_google_api_key
   ```

## Usage

```
streamlit run app.py
```

