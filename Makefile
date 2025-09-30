run: 
	uvicorn api:app --reload --port 8000 &
	streamlit run app.py --server.port 8501